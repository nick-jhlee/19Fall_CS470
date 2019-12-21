## File/Folder description
```
dataset/_train
       /_valid
       /_test
       /ckpt
```
 GTZAN dataset is divided into train, validation, and test dataset.
 - ./_train: contains 70 .wav files
 - ./_valid: contains 20 .wav files
 - ./_test: contains 10 .wav files
 - ./ckpt: contains model architecture(.json) and wieghts(.hdf5) files.

 - data_processor.py: extract MFCC, Spectral_center, Chroma, Spectral_contrast, Melspectrogram from .wav files and convert into numpy files (.npy). 
 - GenreFeature.py: load preprocessed .npy data on GenreFeature class (MFCC, s_center, chroma, s_contrast)
 - melGenreFeature.py: load preprocessed .npy data on melGenreFeature class (MFCC, s_center, chroma, s_contrast, mel)
 
 - LSTM_training.py: train bi-layer LSTM model on training set (with 4 features MFCC, s_center, chroma, s_contrast)
 - melLSTM_training.py: train bi-layer LSTM model on training set (with 5 features MFCC, s_center, chroma, s_contrast, mel)
 - melLSTM_training_3lyr.py: train 3-layer LSTM model on training set (with 5 features MFCC, s_center, chroma, s_contrast, mel)
 
 - LSTM_training_history.csv: training history of LSTM_training.py
 - melLSTM_training_history.csv: training history of melLSTM_training.py
 - 3layer_melLSTM_training_history.csv: training history of melLSTM_training_3lyr.py
 
## Usage
To preprocess .wav data
```
python data_processor.py
```

Test 3 models with real music file. Architecture(.json) and pretrained weight parameters(.hdf5) for models are available
```
python genre_test.py <any .wav file for test>
```

You can train each models on given dataset
```
python LSTM_training.py
python melLSTM_training.py
python melLSTM_training_3lyr.py
```
For each epoch, save model weights if the model is improved based on validation test.
Training histories are saved as .csv files
