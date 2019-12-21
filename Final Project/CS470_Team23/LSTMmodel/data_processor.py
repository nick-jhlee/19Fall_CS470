import librosa
import matplotlib.pyplot as plt
import pathlib, os, csv, re, math
import numpy as np
import pylab

genres = 'blues classical country disco hiphop jazz metal pop reggae rock'.split()
sets= '_test _train _validation'.split()

def one_hot(Y_genre_strings):
    y_one_hot = np.zeros((Y_genre_strings.shape[0], len(genres)))
    for i, genre_string in enumerate(Y_genre_strings):
        index = genres.index(genre_string)
        y_one_hot[i, index] = 1
    return y_one_hot

def extract_audio_features_from_wholedata():
    
    for g in sets: #_train, _test, _validation
        #print(g)
        if g == '_train':
            n_samples = 7000
        elif g == '_test':
            n_samples = 1000
        elif g == '_validation':
            n_samples = 2000

        data = np.zeros(
            (n_samples, 128, 33), dtype=np.float64

        )

        mel_data = np.zeros(
            (n_samples, 128, 161), dtype=np.float64
        )

        target = []
        j = 0
        for filename in os.listdir(f'dataset/{g}'):
            songname = f'dataset/{g}/{filename}'

            splits = re.split("[ .]", filename)
            genre = splits[0]
            
            for i in range(1):
                print (j, i)
                y, sr = librosa.load(songname, sr=None, offset=3*i, duration=3)

                mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13) #
                spectral_center = librosa.feature.spectral_centroid(y=y, sr=sr)
                chroma = librosa.feature.chroma_stft(y=y, sr=sr)
                spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
                mel = librosa.feature.melspectrogram(y=y, sr=sr)
                target.append(genre)
                
                idx= 10*j + i

                data[idx, :mfcc.T[0:128, :].shape[0], 0:13] = mfcc.T[0:128, :]
                data[idx, :spectral_center.T[0:128, :].shape[0], 13:14] = spectral_center.T[0:128, :]
                data[idx, :chroma.T[0:128, :].shape[0], 14:26] = chroma.T[0:128, :]
                data[idx, :spectral_contrast.T[0:128, :].shape[0], 26:33] = spectral_contrast.T[0:128, :]

                mel_data[idx, :mfcc.T[0:128, :].shape[0], 0:13] = mfcc.T[0:128, :]
                mel_data[idx, :spectral_center.T[0:128, :].shape[0], 13:14] = spectral_center.T[0:128, :]
                mel_data[idx, :chroma.T[0:128, :].shape[0], 14:26] = chroma.T[0:128, :]
                mel_data[idx, :spectral_contrast.T[0:128, :].shape[0], 26:33] = spectral_contrast.T[0:128, :]
                mel_data[idx, :mel.T[0:128, :].shape[0], 33:161] = mel.T[0:128, :]

                print("Extracted features audio track %i of %i." % (idx+1, n_samples))

            j = j + 1

        X, mel_X, Y = data, mel_data, np.expand_dims(np.asarray(target), axis=1)

        if g == '_train':
            with open('./dataset/data_train_input.npy', "wb") as f:
                np.save(f, X)

            with open('./dataset/data_train_target.npy', "wb") as f:
                Y = one_hot(Y)
                np.save(f, Y)

            with open('./dataset/mel_data_train_input.npy', "wb") as f:
                np.save(f, mel_X)

            with open('./dataset/mel_data_train_target.npy', "wb") as f:
                np.save(f, Y)


        elif g == '_test':
            with open('./dataset/data_test_input.npy', "wb") as f:
                np.save(f, X)

            with open('./dataset/data_test_target.npy', "wb") as f:
                Y = one_hot(Y)
                np.save(f, Y)

            with open('./dataset/mel_data_test_input.npy', "wb") as f:
                np.save(f, mel_X)

            with open('./dataset/mel_data_test_target.npy', "wb") as f:
                np.save(f, Y)


        elif g == '_validation':
            with open('./dataset/data_valid_input.npy', "wb") as f:
                np.save(f, X)

            with open('./dataset/data_valid_target.npy', "wb") as f:
                Y = one_hot(Y)
                np.save(f, Y)

            with open('./dataset/mel_data_valid_input.npy', "wb") as f:
                np.save(f, mel_X)

            with open('./dataset/mel_data_valid_target.npy', "wb") as f:
                np.save(f, Y)





if __name__ == '__main__':
    extract_audio_features_from_wholedata()
