import numpy as np
import torch

if __name__ == "__main__":
    labels = []
    model = resnet()
    model.eval()
    aggre = np.load('../utils/cluster_v3.npy')
    images = h5py.File('path/to/video_h5', 'r')['video']
    with torch.no_grad():
        for im in images:
            im = (im/255.0-np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
            im = torch.FloatTensor(im).unsqueeze(0)
            im = im.permute(0, 3, 1, 2)
            prob = model(im).detach()
            prob[torch.topk(prob, dim=1, k=990)] = 0
            prediction = prob.numpy() * aggre
            prediction = np.sum(prediction, 1)
            prediction = prediction / np.max(prediction)
            labels.append(prediction)
    np.save('labels_v', labels)
