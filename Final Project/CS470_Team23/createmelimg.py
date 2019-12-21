"""
createmelimg.py

Process the .wav file data accordingly.
(Make a melspectrogram)
"""

import librosa
import librosa.display
import matplotlib.pyplot as plt
import pathlib, os, csv
import numpy as np
import pylab

genres = 'blues classical country disco hiphop jazz metal pop reggae rock'.split()

# Generate melspectrograms for the wav files in the dataset

for g in genres:
    pathlib.Path(f'datamel/{g}').mkdir(parents=True, exist_ok=True)     
    for filename in os.listdir(f'dataset/{g}'):
        songname = f'dataset/{g}/{filename}'
        for i in range(10):
            songdata, sr = librosa.load(songname, sr=None, mono=True, offset=3*i, duration=3)
            pylab.axis('off') # no axis
            pylab.axes([0., 0., 1., 1.], frameon=False, xticks=[], yticks=[]) # Remove the white edge
            mel = librosa.feature.melspectrogram(y=songdata, sr=22050)
            librosa.display.specshow(librosa.power_to_db(mel, ref=np.max))
            pylab.savefig(f'datamel/{g}/{filename[:-3].replace(".", "")}_{i}.png', bbox_inches=None, pad_inches=0)
            pylab.close()
