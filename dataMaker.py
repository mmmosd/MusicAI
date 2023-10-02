import librosa
import glob
import converter
import soundfile as sf
import random


def Cut_Audio(data_path_list, cut_length, write=True, save_path=None, shuffle=True, count=None):
    resultList = []

    if (shuffle):
        random.shuffle(data_path_list)

    for num, data_path in enumerate(data_path_list):
        print('loading file {}: '.format(num) + data_path)
        audio, sr = librosa.load(data_path, sr=44100)
        tempo, beats = librosa.beat.beat_track(y=audio, sr=sr)
        beats = librosa.frames_to_time(beats, sr=sr)

        for idx, start in enumerate(beats):
            if (int(start*sr) + int(cut_length*sr) < len(audio)):
                cut = audio[int(start*sr) : int(start*sr) + int(cut_length*sr)]
                resultList.append(cut)

                if (write): sf.write(save_path+'data{}_{}.wav'.format(num, idx), cut, sr)

                if (count != None): 
                    if (len(resultList) >= count): return resultList

    return resultList


def Load_Data_As_Spectrogram(audio_length, shuffle=True, max_count=None):
    Data_list = Cut_Audio(glob.glob('./Sample_Data/*'), audio_length, write=False, save_path='./data/', shuffle=shuffle, count=max_count)
    fileList = []

    for idx, value in enumerate(Data_list):
        spg = converter.Audio_To_Spectrogram(value)

        spg = spg[0 : int(spg.shape[0] / 32) * 32, 0 : int(spg.shape[1] / 32) * 32]

        h, w = spg.shape

        fileList.append(spg)

    spg = fileList[random.randint(0, len(fileList)-1)]

    converter.Save_Spectrogram_To_Image(spg, 'sample_image')
    converter.Save_Spectrogram_To_Audio(spg, 'sample_audio')

    print('data_count: {}'.format(len(fileList)))

    return fileList, w, h