"""
generatecsv.py

Generate the csv file as needed (useful for creating datasets)
"""

import librosa
import librosa.display
import matplotlib.pyplot as plt
import pathlib, os, csv
import numpy as np

genres = 'blues classical country disco hiphop jazz metal pop reggae rock'.split()

# Generate csv files for the datasets

header = 'filename imgname melname label rms spectral_centroid spectral_bandwidth spectral_flatness spectral_rolloff zero_crossing_rate'
header = header.split()

file = open('metadata_train.csv', 'w', newline='')

with file:
    writer = csv.writer(file)
    writer.writerow(header)

for g in genres:
    for filename in os.listdir(f'dataset/{g}'):
        songname = f'dataset/{g}/{filename}'
        # The first 8 3-sec sections of the song will be put in the training set
        for i in range(8):
            imgname = f'dataimg/{g}/{filename[:-3].replace(".", "")}_{i}.png'
            melname = f'datamel/{g}/{filename[:-3].replace(".", "")}_{i}.png'
            y, sr = librosa.load(songname, sr=None, mono=True, offset=3*i, duration=3)
            rms = librosa.feature.rms(y=y)
            spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
            spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
            spec_fl = librosa.feature.spectral_flatness(y=y)
            spec_ro = librosa.feature.spectral_rolloff(y=y, sr=sr)
            zcr = librosa.feature.zero_crossing_rate(y)
            to_append = f'{songname} {imgname} {melname} {g} {np.mean(rms)} {np.mean(spec_cent)} {np.mean(spec_bw)} {np.mean(spec_fl)} {np.mean(spec_ro)} {np.mean(zcr)}'    
            
            file = open('metadata_train.csv', 'a', newline='')
            with file:
                writer = csv.writer(file)
                writer.writerow(to_append.split())

file = open('metadata_test.csv', 'w', newline='')

with file:
    writer = csv.writer(file)
    writer.writerow(header)

for g in genres:
    for filename in os.listdir(f'dataset/{g}'): 
        songname = f'dataset/{g}/{filename}'
        # The last 2 3-sec sections of the song will be put in the test set
        for i in range(8, 10):
            imgname = f'dataimg/{g}/{filename[:-3].replace(".", "")}_{i}.png'
            melname = f'datamel/{g}/{filename[:-3].replace(".", "")}_{i}.png'
            y, sr = librosa.load(songname, sr=None, mono=True, offset=3*i, duration=3)
            rms = librosa.feature.rms(y=y)
            spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
            spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
            spec_fl = librosa.feature.spectral_flatness(y=y)
            spec_ro = librosa.feature.spectral_rolloff(y=y, sr=sr)
            zcr = librosa.feature.zero_crossing_rate(y)
            to_append = f'{songname} {imgname} {melname} {g} {np.mean(rms)} {np.mean(spec_cent)} {np.mean(spec_bw)} {np.mean(spec_fl)} {np.mean(spec_ro)} {np.mean(zcr)}'    
            
            file = open('metadata_test.csv', 'a', newline='')
            with file:
                writer = csv.writer(file)
                writer.writerow(to_append.split())