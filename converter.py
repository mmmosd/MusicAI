import librosa
import numpy as np
import soundfile as sf
import os

from sklearn.preprocessing import MinMaxScaler
from PIL import Image

def Sound_To_Spectrogram(FilePath, duration):
    y = librosa.load(FilePath, duration=duration, sr=44100)
    mel_spec = librosa.feature.melspectrogram(y=y[0], sr=44100)
    return mel_spec

def Save_Spectrogram_Image(Spectrogram, filename):
    arr = MinMaxScaler().fit_transform(Spectrogram)*255
    Image.fromarray(arr).convert('RGB').save(filename, 'JPEG')

def Save_Spectrogram_Audio(Spectrogram, filename):
    audio = librosa.feature.inverse.mel_to_audio(Spectrogram, sr=44100)
    sf.write(filename, audio, 44100)