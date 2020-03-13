import torch
import torch.utils.data as data
import os
import numpy as np
import random
import h5py
import librosa

EPS = np.spacing(1)

def DataAllocate(batch):
    
    audios = []
    visuals = []
    rois = []
    for sample in batch:
        audios.append(sample[0])
        visuals.append(sample[1])
        rois.append(sample[2])
    
    audios = torch.stack(audios, dim=0)#(batchsize, mix, T, F)
    visuals = torch.stack(visuals, dim=0).permute(0, 1, 2, 5, 3, 4).contiguous()#(batchsize, mix, 9, 256, 256, 3)
    rois = torch.stack(rois, dim=0)#(batchsize, mix, 9, 8, 4)

    return audios, visuals, rois
    
class AudioVisualData(data.Dataset):
    
    def __init__(self, audio, video, rois, mix, frame, dataset, training):
        self.sr = 22050
        self.mix = mix
        self.frame = frame
        self.dataset = dataset
        self.training = training
        self.base = 3339 if self.dataset == 'AVE_C' else 0
        if self.training:
            self.samples = 3339 if self.dataset == 'AVE_C' else 10000
        else:
            self.samples = 402 if self.dataset == 'AVE_C' else 500

        self.audio = h5py.File(audio, 'r')
        self.video = h5py.File(video, 'r')
        self.rois = h5py.File(rois, 'r')

    def __len__(self):
        return self.samples
    
    def get_train(self, idx):
        train_idx = np.random.permutation(self.samples)[:self.mix]
        audio = [None] * self.mix
        video = [None] * self.mix
        rois = [None] * self.mix
        spec = [None] * self.mix
        for i in range(self.mix):
            if self.dataset == 'AVE_C':#AVE
                start = np.random.randint(0, 10-self.frame)
                spec[i] = self.audio['audio'][train_idx[i], start: start+self.frame].reshape(-1, 64)
                rois[i] = self.rois['roi'][train_idx[i], start: start+self.frame] * 256
                video[i] = (self.video['video'][train_idx[i], start: start+self.frame]/255.0 - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])

            if self.dataset == 'Flickr':#Flickr
                spec[i] = self.audio['audio'][train_idx[i]]
                rois[i] = self.rois['roi'][train_idx[i]] * 256
                rois[i] = np.expand_dims(rois[i], 0)
                video[i] = (self.video['video'][train_idx[i]] / 255.0 - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
                video[i] = np.expand_dims(video[i], 0)

        spec = torch.FloatTensor(spec)
        rois = torch.FloatTensor(rois)
        video = torch.FloatTensor(video)
        
        return spec, video, rois
    
    def get_val(self, idx):
        val_idx = np.random.permutation(self.samples-1)
        val_idx[val_idx>=idx] += 1
        val_idx = np.hstack([np.array(idx), val_idx])
        val_idx = val_idx + self.base
        audio = [None] * self.mix
        video = [None] * self.mix
        rois = [None] * self.mix
        spec = [None] * self.mix
        for i in range(self.mix):
            if self.dataset == 'AVE_C':#AVE
                start = np.random.randint(0, 10-self.frame)
                spec[i] = self.audio['audio'][val_idx[i], start: start+self.frame].reshape(-1, 64)
                rois[i] = self.rois['roi'][val_idx[i], start: start+self.frame] * 256
                video[i] = (self.video['video'][val_idx[i], start: start+self.frame]/255.0 - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
            if self.dataset == 'Flickr':#Flickr
                spec[i] = self.audio['audio'][val_idx[i]]
                rois[i] = self.rois['roi'][val_idx[i]] * 256
                rois[i] = np.expand_dims(rois[i], 0)
                video[i] = (self.video['video'][val_idx[i]]/255.0 - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
                video[i] = np.expand_dims(video[i], 0)

        spec = torch.FloatTensor(spec)
        rois = torch.FloatTensor(rois)
        video = torch.FloatTensor(video)
        
        return spec, video, rois
    
    def __getitem__(self, idx):

        if self.training:
            return self.get_train(idx)
        else:
            return self.get_val(idx)
            
