import torch
import torch.utils.data as data
import os
import numpy as np
import random
import h5py

def DataAllocate(batch):   
    specs = []
    imgs = []
    labels_a = []
    labels_v = []
    for sample in batch:
        specs.append(sample[0])
        imgs.append(sample[1])
        labels_a.append(sample[2])
        labels_v.append(sample[3])
    specs = torch.stack(specs, 0).unsqueeze(1)
    imgs = torch.stack(imgs, 0).permute(0, 3, 1, 2).contiguous()
    labels_a = torch.stack(labels_a, 0)
    labels_v = torch.stack(labels_v, 0)
    return specs, imgs, labels_a, labels_v
    
    
class AudioVisualData(data.Dataset):
    
    def __init__(self):
        self.audio = '/media/ruiq/Data/AudioVisual/data/Flickr/Spec.h5'
        self.video = '/media/ruiq/Data/AudioVisual/data/Flickr/Video.h5'
        self.label_a = '/media/ruiq/Data/AudioVisual/data/Flickr/labels_a.npy'
        self.label_v = '/media/ruiq/Data/AudioVisual/data/Flickr/labels_v.npy'
        self.audio = h5py.File(self.audio, 'r')['audio']
        self.video = h5py.File(self.video, 'r')['video']
        self.label_a = np.load(self.label_a)
        self.label_v = np.load(self.label_v)

    def __len__(self):
        return self.label_a.shape[0]
    
    def __getitem__(self, idx):
        spec = self.audio[idx]
        img = self.video[idx]
        label_a = self.label_a[idx]
        label_v = self.label_v[idx]
        if np.random.rand() > 0.5:
            img = img[:, ::-1]
        img = (img/255.0-np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
        return torch.FloatTensor(spec), torch.FloatTensor(img), torch.FloatTensor(label_a), torch.FloatTensor(label_v)
