import math
import librosa
import numpy as np
import glob
import converter
import soundfile as sf

def Save_Cut_Audio(data_path_list, cut_length, save_path):
    resultList = []

    for num, data_path in enumerate(data_path_list):
        print('loading file: '+data_path)
        audio, sr = librosa.load(data_path, sr=44100)

        for j in range(int(math.floor(len(audio)/(cut_length*sr)))):
            cut = audio[j*cut_length*sr : j*cut_length*sr + cut_length*sr]
            sf.write(save_path+'data{}_{}.wav'.format(num, j), cut, sr)
            resultList.append(cut)

    return resultList


def Load_Data_As_Spectrogram(audio_length):
    Data_list = Save_Cut_Audio(glob.glob('./Sample_Data/*'), audio_length, './data/')
    fileList = []

    for i in range(len(Data_list)):
        spg = converter.Audio_To_Spectrogram(Data_list[i])
        np.resize(spg, (128, int(spg.shape[1]/128) * 128))
        fileList.append(spg)

    fileList = np.array(fileList)
    y, x = fileList[0].shape

    print('fileShape: {}, {}'.format(x, y))

    return fileList, x, y