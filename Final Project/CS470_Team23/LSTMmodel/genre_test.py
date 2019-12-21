import librosa
import logging
import sys
import numpy as np
import keras
from keras.models import model_from_json
#from GenreFeature import GenreFeature
#from melGenreFeature import melGenreFeature

logging.getLogger("tensorflow").setLevel(logging.ERROR)

def get_features(input_path): # return both data for normal and mel version
    features = np.zeros((1,128,33), dtype=np.float64)
    melfeatures = np.zeros((1,128,161), dtype=np.float64)\

    y, sr = librosa.load(input_path)

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    spectral_center = librosa.feature.spectral_centroid(y=y, sr=sr)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    mel = librosa.feature.melspectrogram(y=y, sr=sr)

    features[0, :, 0:13] = mfcc.T[0:128, :]
    features[0, :, 13:14] = spectral_center.T[0:128, :]
    features[0, :, 14:26] = chroma.T[0:128, :]
    features[0, :, 26:33] = spectral_contrast.T[0:128, :]

    melfeatures[0, :, 0:13] = mfcc.T[0:128, :]
    melfeatures[0, :, 13:14] = spectral_center.T[0:128, :]
    melfeatures[0, :, 14:26] = chroma.T[0:128, :]
    melfeatures[0, :, 26:33] = spectral_contrast.T[0:128, :]
    melfeatures[0, :, 33:161] = mel.T[0:128, :]

    return features, melfeatures

def get_model(json_path, weights_path):

    with open(json_path, "r") as model_json:
        model = model_from_json(model_json.read())

    model.load_weights(weights_path)
    model.compile(loss='categorical_crossentropy', optimizer="adam", metrics=['accuracy'])
    
    return model

def predict_genre(model, input):
    genre_list = 'blues classical country disco hiphop jazz metal pop reggae rock'.split()
    pred = model.predict(input)
    return genre_list[np.argmax(pred)]

if __name__=="__main__":

    path = sys.argv[1] if len(sys.argv) == 2 else print("Just one music file at once")
    net = get_model("./dataset/ckpt/LSTM_adam.json","./dataset/ckpt/LSTM_adam--86--1.600.hdf5")
    melnet =  get_model("./dataset/ckpt/melLSTM_adam.json","./dataset/ckpt/mel_LSTM_ada--224--2.271.hdf5")
    mel3net = get_model("./dataset/ckpt/trilayer_melLSTM_adam.json","./dataset/ckpt/trilayer_mel_LSTM_adam--174--2.119.hdf5")
    data, meldata = get_features(path)

    ans1 = predict_genre(net, data)
    ans2 = predict_genre(melnet, meldata)
    ans3 = predict_genre(mel3net, meldata)

    print("Bi-layer LSTM predict with MFCC, centriod, contrast, chroma: {}".format(ans1))
    print("Bi-layer LSTM predict with melspectrogram MFCC, centroid, contrast, chroma: {}".format(ans2))
    print("Tri-layer LSTM predict with melspectrogram MFCC, centroid, contrast, chroma: {}".format(ans3))