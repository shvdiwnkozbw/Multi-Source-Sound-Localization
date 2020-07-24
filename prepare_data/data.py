import numpy as np
import cv2
import librosa
import os
import pickle
import h5py

#load audio-image pairs index as a list
pairs = pickle.load(open('path/to/pairs', 'rb'))

imgs = []
audios = []

for pair in pairs:
    img_dir = pair[0]
    audio_dir = pair[1]
    #load image data
    img = cv2.imread(img_dir)
    img = cv2.resize(img, (256, 256))
    img = img [:, :, ::-1]
    #load audio data and transform into spectrogram
    wav, sr = librosa.load(audio_dir, sr=22050)
    logmel_spec = np.log(librosa.feature.melspectrogram(wav, sr, n_fft=int(0.04*sr), 
                                                        hop_length=int(0.02*sr), n_mels=64) + np.spacing(1)).T
    imgs.append(img)
    audios.append(logmel_spec)
    
with h5py.File('path/to/img_h5', 'w') as f:
    f['video'] = np.array(imgs)
with h5py.File('path/to/audio_h5', 'w') as f:
    f['audio'] = np.array(logmel_spec)
