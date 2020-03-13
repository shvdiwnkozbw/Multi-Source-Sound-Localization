import torch
import torch.utils.data as data
import os
import numpy as np
import pickle
import cv2
import librosa

EPS = np.spacing(1)


def DataAllocate(batch):
    audios = []
    visuals = []
    labels = []
    for sample in batch:
        audios.append(sample[0])
        visuals.append(sample[1])
        labels.append(sample[2])

    audios = torch.stack(audios, dim=0)
    visuals = torch.stack(visuals, dim=0)
    labels = torch.stack(labels, dim=0)

    return audios, visuals, labels


class AudioVisualData(data.Dataset):

    def __init__(self, mix, frame, training):
        self.sr = 22050
        self.mix = mix
        self.frame = frame
        self.training = training
        self.mean = np.array([0.485, 0.456, 0.406])
        self.std = np.array([0.229, 0.224, 0.225])
        self.prefix = '/media/yuxi/Data/AudioVisual/data/Audioset/'
        if training:
            file = self.prefix + 'train_audio.pkl'
            self.prefix = self.prefix + 'train'
            self.effidx = np.load('audioset_train_effv2.npy')
        else:
            file = self.prefix + 'val_audio.pkl'
            self.prefix = self.prefix + 'val'
            self.effidx = np.load('audioset_val.npy')
        with open(file, 'rb') as f:
            self.anno = pickle.load(f, encoding='bytes')
        self.keys = list(self.anno.keys())
        self.keys.sort()

    def __len__(self):
        if self.training:
            return len(self.effidx)
        else:
            return len(self.keys)-1

    def get_train(self):
        specs = []
        visuals = []
        labels = []
        audio_len = self.frame * self.sr
        visual = np.zeros((self.frame, 256, 256, 3))
        vids = np.random.permutation(len(self.effidx))[:self.mix]
        vids = self.effidx[vids]
        for vid in vids:
            frame = np.random.randint(0, 11-self.frame)

            im = os.path.join(self.prefix, self.keys[vid])
            for f in range(frame, frame+self.frame):
                img = os.path.join(im, str(f).zfill(4)+'.jpg')
                visual[f-frame] = cv2.imread(img)
            visual = (visual / 255.0 - self.mean) / self.std

            audio = os.path.join(self.prefix, 'audio', self.keys[vid]+'.wav')
            audio, _ = librosa.load(audio, self.sr, offset=frame, duration=self.frame)
            if len(audio) < audio_len:
                wav = np.zeros(audio_len)
                wav[:len(audio)] = audio
                audio = wav
            spec = librosa.feature.melspectrogram(audio[:audio_len], self.sr,
                                                  n_fft=882, hop_length=441, n_mels=64)
            spec = np.log(spec+EPS).T

            specs.append(spec)
            visuals.append(visual)
            labels.append(self.anno[self.keys[vid]])

        return specs, visuals, labels

    def get_val(self, idx):
        specs = []
        visuals = []
        labels = []
        audio_len = self.frame * self.sr
        visual = np.zeros((self.frame, 256, 256, 3))
        vids = np.random.permutation(len(self.effidx-1))[:self.mix-1]
        vids[vids>=idx] += 1
        vids = np.hstack([np.array(idx), vids])
        for vid in vids:
            frame = np.random.randint(0, 11-self.frame)

            im = os.path.join(self.prefix, self.keys[vid])
            for f in range(frame, frame+self.frame):
                img = os.path.join(im, str(f).zfill(4)+'.jpg')
                visual[f-frame] = cv2.resize(cv2.imread(img)[:, :, ::-1], (256, 256))
            visual = (visual / 255.0 - self.mean) / self.std

            audio = os.path.join(self.prefix, 'audio', self.keys[vid]+'.wav')
            audio, _ = librosa.load(audio, self.sr, offset=frame, duration=self.frame)
            if len(audio) < audio_len:
                wav = np.zeros(audio_len)
                wav[:len(audio)] = audio
                audio = wav
            spec = librosa.feature.melspectrogram(audio[:audio_len], self.sr,
                                                  n_fft=882, hop_length=441, n_mels=64)
            spec = np.log(spec+EPS).T

            specs.append(spec)
            visuals.append(visual)
            labels.append(self.anno[self.keys[vid]])

        return specs, visuals, labels

    def __getitem__(self, idx):

        if self.training:
            audio, visual, label = self.get_train()
            audio = torch.FloatTensor(audio)
            visual = torch.FloatTensor(visual).permute(0, 1, 4, 2, 3).contiguous()
            label = torch.FloatTensor(label)
            return audio, visual, label

        else:
            audio, visual, label = self.get_val(idx)
            audio = torch.FloatTensor(audio)
            visual = torch.FloatTensor(visual).permute(0, 1, 4, 2, 3).contiguous()
            label = torch.FloatTensor(label)
            return audio, visual, label

