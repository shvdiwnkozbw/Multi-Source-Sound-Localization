import torch
import torch.utils.data as data
import os
import numpy as np
import cv2
import librosa

EPS = np.spacing(1)

def DataAllocate(batch):
    
    audios = []
    visuals = []
    rois = []
    gtmaps = []
    boxes = []
    for sample in batch:
        audios.append(sample[0])
        visuals.append(sample[1])
        rois.append(sample[2])
        gtmaps.append(sample[3])
        boxes.append(sample[4])
    
    audios = torch.stack(audios, dim=0)
    visuals = torch.stack(visuals, dim=0).permute(0, 1, 4, 2, 3).contiguous()
    rois = torch.stack(rois, dim=0)
    gtmaps = torch.stack(gtmaps, dim=0)
    
    return audios, visuals, rois, gtmaps, boxes
    
class AudioVisualData(data.Dataset):
    
    def __init__(self, file, mix, pool, training):
        self.sr = 22050
        self.mix = mix
        self.file = file
        self.pool = pool
        self.training = training
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
        imgs = []
        rois = []
        specs = []
        gtmaps = []
        boxes = []
        for i in range(idx*self.mix, idx*self.mix+self.mix):
            img = self.file[self.number[i]]['visual']
            img = os.path.join('/media/yuxi/Data/SuperAVC', img)
            wav = self.file[self.number[i]]['audio']
            wav = os.path.join('/media/yuxi/Data/SuperAVC', wav)
            roi = self.file[self.number[i]]['roi'] * 256
            anno = self.file[self.number[i]]['anno']
            img = cv2.imread(img)
            audio, sr = librosa.load(wav)
            audio = audio / np.max(np.abs(audio) + np.spacing(1))
            spec = librosa.feature.melspectrogram(audio, n_mels=64, n_fft=882, hop_length=441)
            spec = np.log(spec+EPS).T
            img = img[:, :, ::-1]
            img = img / 255.0
            img = img - np.array([0.485, 0.456, 0.406])
            img = img / np.array([0.229, 0.224, 0.225])
            gtmap = torch.zeros((256, 256))
            for coordinate in anno:
                coordinate = coordinate.astype(np.int)
                boxes.append(coordinate)
                (xmin, ymin, xmax, ymax) = coordinate
                gtmap[ymin-1:ymax-1, xmin-1:xmax-1] += 1.0
            gtmap = torch.min(gtmap/2.0, torch.ones(256, 256))
            gtmaps.append(gtmap)
            imgs.append(img)
            specs.append(spec)
            rois.append(roi)
        return torch.FloatTensor(specs), torch.FloatTensor(imgs), torch.FloatTensor(rois), torch.stack(gtmaps, 0), boxes
