import math
import librosa
import soundfile as sf

def Save_Cut_Audio(data_path_list, cut_length, save_path):
    resultList = []

    for num, data_path in enumerate(data_path_list):
        print(data_path)
        audio, sr = librosa.load(data_path, sr=44100)

        for j in range(int(math.floor(len(audio)/(cut_length*sr)))):
            cut = audio[j*cut_length*sr : j*cut_length*sr + cut_length*sr]
            sf.write(save_path+'data_{}.wav'.format(j), cut, sr)
    
    return resultList