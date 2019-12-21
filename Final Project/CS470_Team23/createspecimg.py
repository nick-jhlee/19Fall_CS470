"""
createspecimg.py

Process the .wav file data accordingly.
"""

import librosa
import librosa.display
import matplotlib.pyplot as plt
import pathlib, os, csv
import numpy as np

genres = 'blues classical country disco hiphop jazz metal pop reggae rock'.split()

# Generate spectrograms for the wav files in the dataset

cmap = plt.get_cmap('magma')
ax = plt.axes()
ax.set_axis_off()

# 256 x 256 (borders account for 30% -> 256 * 1.3 = 333)
plt.figure(figsize=(3.33,3.33), dpi=100)

for g in genres:
    pathlib.Path(f'dataimg/{g}').mkdir(parents=True, exist_ok=True)     
    for filename in os.listdir(f'dataset/{g}'):
        songname = f'dataset/{g}/{filename}'
        for i in range(10):
            songdata, sr = librosa.load(songname, sr=None, mono=True, offset=3*i, duration=3)
            plt.specgram(songdata, NFFT=2048, Fs=2, Fc=0, noverlap=128, cmap=cmap, sides='default', mode='default', scale='dB')
            plt.axis('tight')
            plt.axis('off')
            plt.savefig(f'dataimg/{g}/{filename[:-3].replace(".", "")}_{i}.png', bbox_inches='tight', transparent='true', pad_inches=0)
            plt.clf()
