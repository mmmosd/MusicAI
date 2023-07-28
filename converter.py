import librosa
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from PIL import Image

def Sound_To_Spectrogram(Sound, duration):
    y = librosa.load(Sound, duration=duration)
    return librosa.amplitude_to_db(np.abs(librosa.stft(y[0])), ref=np.max)

def Spectrogram_To_Sound(Spectrogram, duration):
    return librosa.feature.melspectrogram(Spectrogram, duration=duration)

def Save_Spectrogram_Image(Spectrogram, filename):
    arr = MinMaxScaler().fit_transform(Spectrogram)*255
    Image.fromarray(arr).convert('RGB').save(filename, 'JPEG')

spg = Sound_To_Spectrogram(librosa.ex('choice'), 15)
Save_Spectrogram_Image(spg, 'spgTest.jpg')