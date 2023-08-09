import math
import librosa
import numpy as np
import glob
import converter
import soundfile as sf


def Cut_Audio(data_path_list, cut_length, save_path, write=True):
    resultList = []

    for num, data_path in enumerate(data_path_list):
        print('loading file: '+data_path)
        audio, sr = librosa.load(data_path, sr=44100)

        for j in range(int(math.floor(len(audio)/(cut_length*sr)))):
            cut = audio[j*cut_length*sr : j*cut_length*sr + cut_length*sr]
            
            if (write): sf.write(save_path+'data{}_{}.wav'.format(num, j), cut, sr)
            
            resultList.append(cut)

    return resultList


def Load_Data_As_Spectrogram(audio_length):
    Data_list = Cut_Audio(glob.glob('./Sample_Data/*'), audio_length, './data/', write=False)
    fileList = []

    for i in range(len(Data_list)):
        spg = converter.Audio_To_Spectrogram(Data_list[i])
        # spg = np.resize(spg, (1, 128, int(spg.shape[1]/32) * 32))

        h, w = spg.shape

        fileList.append(spg)

    spg = fileList[78]

    converter.Save_Spectrogram_To_Image(spg, 'sample_image')
    converter.Save_Spectrogram_To_Audio(spg, 'sample_audio')

    return fileList, w, h