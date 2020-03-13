import argparse
import librosa
import sys
import numpy as np

SAMPLE_RATE = 22050
EPS = np.spacing(1)


def extract_feature(wavefilepath, **kwargs):
    wav, sr = librosa.load(wavefilepath, sr=SAMPLE_RATE)
    return np.log(librosa.feature.melspectrogram(wav, sr, **kwargs) + EPS).T


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-wavefilepath', default='../sample.wav')
    parser.add_argument('-n_mels', default=64, type=int)
    parser.add_argument('-n_fft',
                        default=0.04,
                        type=float,
                        help='window size for fft, default 40ms')
    parser.add_argument('-hop_length', default=0.02, type=float, help='帧移')
    args = parser.parse_args()
    args.n_fft = int(args.n_fft * SAMPLE_RATE)
    args.hop_length = int(args.hop_length * SAMPLE_RATE)
    feature = extract_feature(**vars(args))
    return feature


if __name__ == "__main__":
    main()
