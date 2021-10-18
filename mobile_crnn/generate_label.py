import numpy as np
import torch
import extract_feature
from models import mobilecrnn_v2


if __name__ == "__main__":
    labels = []
    model = mobilecrnn_v2().to('cpu').eval()
    aggre = np.load('../utils/cluster_a.npy')
    audios = h5py.File('path/to/spec_h5', 'r')['audio']
    with torch.no_grad():
        for feature in audios:
            feature = torch.as_tensor(feature).unsqueeze(0).unsqueeze(0)
            prediction_tag, _ = model(feature)
            prediction = prediction_tag.detach().numpy() * aggre
            prediction = np.max(prediction, 1)
            #to make the predicted probability more discriminate
            prediction[prediction>0.3] =  prediction[prediction>0.3]*0.4 + 0.6
            labels.append(prediction)
    np.save('labels_a', labels)
