'''

cropimg.py

The image data created from createdata is not fully square.
This file modifies the image so that it is in 1024 * 1024 resolution,
and gets rid of alpha (from RGBa to RGB format)

'''

import os
from PIL import Image

genres = 'blues classical country disco hiphop jazz metal pop reggae rock'.split()

for g in genres:
    for filename in os.listdir(f'dataimg/{g}'):
        
        # For spectrogram images
        imgpath = f'dataimg/{g}/{filename}'
        img = Image.open(imgpath)
        img = img.convert('RGB')
        w, h = img.size

        img.crop((0, 0, w-2, h)).save(imgpath)

        # For mel spectrogram images

        melpath = f'datamel/{g}/{filename}'
        mel = Image.open(melpath)
        mel = mel.convert('RGB').save(melpath)
