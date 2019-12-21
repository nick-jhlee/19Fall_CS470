# AI_Team23_Proj
Youngjin Jin  
Minsung Park  
Junghyun Lee  

## Dependencies
libblas-3.8.0        
pyparsing-2.4.5      
joblib-0.14.0        
kiwisolver-1.1.0     
matplotlib-base-3.1. 
decorator-4.4.1      
python-dateutil-2.8. 
cycler-0.10.0        
matplotlib-3.1.2     
llvmlite-0.30.0      
audioread-2.1.8      
librosa-0.6.3        
numba-0.46.0         
resampy-0.2.2        
pyqt-5.9.2           
sip-4.19.8           
libcblas-3.8.0      
tornado-6.0.3        
scikit-learn-0.21.3  
pygame-1.9.6  
pytz-2019.3     
pandas-0.25.3     
torch-1.3.1  
torchvision-0.4.2  
keras-2.3.1   
tensorflow-1.14.0

Note that installing librosa, torch, torchvision, and pandas should take care of most of these dependencies.  

## Running Environment
The models are run using anaconda in a Windows environment with CUDA 10.2.  
```
conda activate cuda10
```
The models have been trained with an RTX 2070 SUPER.  

## List of files and directories (in directory / alphabetical order)
- dataimg: contains the image dataset for spectrogram based on the GTZAN dataset, labeled by genre. (256 x 256 Resolution)
- datamel: contains the image dataset for Mel-spectrogram based on the GTZAN dataset, labeled by genre. (640 x 480 Resolution)
- dataset: contains the original wav files from the GTZAN dataset, labeled by genre.
- net: contains the trained neural network model.
- cnn_test.py: script used to test the network on the test dataset
- cnn_train.py: script used to train the network on the training dataset
- cnn.py: contains the neural network architectures used on the dataset
- createmelimg.py: script that generates mel spectrogram images from the GTZAN dataset
- createspecimg.py: script that generates spectrogram images from the GTZAN dataset
- cropimg.py: script that is used to crop the dataset to a preferred resolution and convert from RGBa to RGB from
- generatecsv.py: generates two csv files (test and train data) which contain many properties available from the librosa library
- gtzanmelspec.py: contains the dataset class for mel spectrogram images
- gtzanspecgram.py: contains the dataset class for spectrogram images

## How to use
There are two datasets based on the GTZAN dataset and a model for each dataset. One is a spectrogram, and the other is a mel-spectrogram. The datasets are contained in dataimg and datamel directories, respectively. The neural network used to train these models is a Convolutional Neural Network. If this repository is cloned directly, all the datasets will have been prepared so many of the python scripts will not need to be called.  

To train the models, call the following (in a Windows environment):
```
python cnn_train.py
```

Once the model has been trained, call the following to test the trained model:
```
python cnn_test.py
```

There are two models available and you can change the model being used by commenting out one of the lines in cnn_train.py and cnn_test.py:
```
net = Model_CNN()
net = Model_CNN_Mel()
```

## Things to note
The LSTM model description can be found in the README file in the LSTM directory.
