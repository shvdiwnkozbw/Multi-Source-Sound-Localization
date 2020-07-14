import torch
import torch.utils.data as data
import os
import numpy as np
import cv2
import librosa
import pickle

EPS = np.spacing(1)

def TestAllocate(batch):
    specs = []
    imgs = []
    gtmaps = []
    boxes = []
    for sample in batch:
        specs.append(sample[0])
        imgs.append(sample[1])
        gtmaps.append(sample[2])
        boxes.append(sample[3])
    specs = torch.stack(specs, 0).unsqueeze(1)
    imgs = torch.stack(imgs, 0).permute(0, 3, 1, 2).contiguous()
    gtmaps = torch.stack(gtmaps, 0)
    
    return specs, imgs, gtmaps, boxes
    
class TestData(data.Dataset):
    
    def __init__(self):
        self.file = pickle.load(open('/media/ruiq/Data/yuxi_data/ruiq/AudioVisual/v8/gtdata.pkl', 'rb'))
        f = open('/media/ruiq/Data/yuxi_data/ruiq/AudioVisual/v8/gtest.txt')
        self.pool = []
        while True:
            num = f.readline()
            if len(num) == 0:
                break
            self.pool.append(num.split('.')[0])
        self.search()

    def __len__(self):
        return len(self.number)

    def search(self):
        self.number = []
        for i in range(len(self.file)):
            img = self.file[i]['visual']
            if img.split('/')[-1].split('.')[0] in self.pool:
                self.number.append(i)
        np.save('file', np.array(self.number))

    def __getitem__(self, idx):
        boxes = []
        img = self.file[self.number[idx]]['visual']
        img = os.path.join('/media/ruiq/Data/SuperAVC', img)
        wav = self.file[self.number[idx]]['audio']
        wav = os.path.join('/media/ruiq/Data/SuperAVC', wav)
        anno = self.file[self.number[idx]]['anno']
        img = cv2.imread(img)[:, :, ::-1]
        img = (img/255.0-np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
        audio, sr = librosa.load(wav)
        audio = audio / np.max(np.abs(audio) + np.spacing(1))
        spec = librosa.feature.melspectrogram(audio, n_mels=64, n_fft=882, hop_length=441)
        spec = np.log(spec+EPS).T
        gtmap = torch.zeros((256, 256))
        for coordinate in anno:
            coordinate = coordinate.astype(np.int)
            boxes.append(coordinate)
            (xmin, ymin, xmax, ymax) = coordinate
            gtmap[ymin-1:ymax-1, xmin-1:xmax-1] += 1.0
        gtmap = torch.min(gtmap/2.0, torch.ones(256, 256))
        return torch.FloatTensor(spec), torch.FloatTensor(img), gtmap, boxes
