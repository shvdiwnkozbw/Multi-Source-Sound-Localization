import argparse
import numpy as np
import torch
import extract_feature
import utils
import pandas as pd
import librosa
# from models import resnet34, crnn2, resnet101, vgg11, mobilecrnn_v1, mobilecrnn_v2
from models import mobilecrnn_v1, mobilecrnn_v2
from sklearn.preprocessing import binarize


if __name__ == "__main__":
    # model = mobilecrnn_v1().to('cpu').eval()
    model = mobilecrnn_v2().to('cpu').eval()
    encoder = torch.load('encoder.pth')
    feature = extract_feature.main()
    K = 5
    THRESHOLD = 0.5
    np.set_printoptions(precision=3)
    with torch.no_grad():
        feature = torch.as_tensor(feature).unsqueeze(0)  # BatchDim
        prediction_tag, prediction_time = model(feature)

        if prediction_time is not None:  # Some models do not predict timestamps
            filtered_pred = utils.double_threshold(prediction_time,
                                                   high_thres=0.5,
                                                   low_thres=0.05)

            # filtered_pred = binarize(prediction_time[0], 0.5)[None, :, :]
            labelled_predictions = utils.decode_with_timestamps(
                encoder, filtered_pred)
            pred_label_df = pd.DataFrame(labelled_predictions[0],
                                         columns=['event', 'start',
                                                  'end']).sort_values('start')
            if not pred_label_df.empty:
                # Print frame level predictions
                # Frame is on 80ms scale
                pred_label_df['start'] *= 0.08
                pred_label_df['end'] *= 0.08
                print("Frame-Level predictions\n\n{}\n".format(pred_label_df))
        top_k_prob, top_k_index = torch.topk(prediction_tag, K)
        print(torch.max(prediction_tag))
        prediction_tag = binarize(prediction_tag, threshold=THRESHOLD)
        prediction = encoder.inverse_transform(prediction_tag)[0]
        top_k_labels = encoder.classes_[top_k_index[0]]
        print("Top-{:<10}:\t{}".format(K, ",".join(top_k_labels)))
        print("Prob-{:<9}:\t{}".format(K, top_k_prob[0].numpy()))
        print("Threshold {:<4}:\t{} ".format(THRESHOLD, ",".join(prediction)))


