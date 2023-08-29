import librosa
import numpy as np
import soundfile as sf

from PIL import Image
from sklearn.preprocessing import MinMaxScaler

def Audio_To_Spectrogram(Audio):
    S = librosa.feature.melspectrogram(y=Audio, sr=44100, n_mels=128)

    log_mel = librosa.power_to_db(S, ref=np.max)
    norm_spec = (log_mel/40)+1

    return norm_spec

def Save_Spectrogram_To_Image(Spectrogram, filename, write=True):
    arr = MinMaxScaler().fit_transform(Spectrogram)*255

    if (write == True):
        Image.fromarray(arr).convert('RGB').save('./Result_Image/'+filename+'.jpg', 'JPEG')

def Save_Spectrogram_To_Audio(Spectrogram, filename, volume=15, write=True):
    Spectrogram = (Spectrogram-1)*40
    Spectrogram = librosa.db_to_power(Spectrogram)

    S = librosa.feature.inverse.mel_to_stft(Spectrogram, sr=44100)
    y = librosa.griffinlim(S)*volume

    if (write == True):
        sf.write('./Result_Audio/'+filename+'.wav', y, 44100)

    return y