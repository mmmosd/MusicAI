import librosa
import numpy as np
import soundfile as sf

from PIL import Image

def Audio_To_Spectrogram(Audio):
    S = librosa.feature.melspectrogram(y=Audio, sr=44100, n_mels=128)

    log_mel = librosa.power_to_db(S, ref=np.max) # normalize -80 ~ 0
    norm_spec = (log_mel/40)+1 # normalize -1 ~ 1

    return norm_spec

def Save_Spectrogram_To_Image(Spectrogram, filename, write=True):
    arr = ((Spectrogram+1)/2)*255 # convert 0 ~ 255

    if (write == True):
        Image.fromarray(arr).convert('RGB').save('./Result_Image/'+filename+'.jpg', 'JPEG')

    return arr

def Save_Spectrogram_To_Audio(Spectrogram, filename, volume=25, write=True):
    Spectrogram = (Spectrogram-1)*40 # convert -80 ~ 0
    Spectrogram = librosa.db_to_power(Spectrogram)

    y = librosa.feature.inverse.mel_to_audio(Spectrogram, sr=44100)*volume

    if (write == True):
        sf.write('./Result_Audio/'+filename+'.wav', y, 44100)

    return y