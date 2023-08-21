import librosa
import numpy as np
import soundfile as sf

from PIL import Image
from sklearn.preprocessing import MinMaxScaler

def Audio_To_Spectrogram(Audio):
    mel_spec = librosa.feature.melspectrogram(y=Audio, sr=44100, n_mels=128)

    log_mel = librosa.power_to_db(np.abs(mel_spec))
    norm_spec = librosa.util.normalize(log_mel)

    # norm_spec = MinMaxScaler().fit_transform(mel_spec)*2 - 1

    return norm_spec

def Save_Spectrogram_To_Image(Spectrogram, filename, write=True):
    arr = MinMaxScaler().fit_transform(Spectrogram)*255

    if (write == True):
        Image.fromarray(arr).convert('RGB').save('./Result_Image/'+filename+'.jpg', 'JPEG')

def Save_Spectrogram_To_Audio(Spectrogram, filename, write=True):
    Spectrogram = np.array(Spectrogram*5, np.float32)
    audio = librosa.feature.inverse.mel_to_audio(Spectrogram, sr=44100)

    if (write == True):
        sf.write('./Result_Audio/'+filename+'.wav', audio, 44100, format='WAV')