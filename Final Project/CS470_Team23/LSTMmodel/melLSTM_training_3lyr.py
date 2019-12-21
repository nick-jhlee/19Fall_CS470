import logging
import os
import pandas
from keras.models import Sequential
from keras.layers.recurrent import LSTM
from keras.layers import Dense
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint

from melGenreFeature import melGenreFeature

logging.getLogger("tensorflow").setLevel(logging.ERROR)

genre_features = melGenreFeature()

# Check preprocessed data
genre_features.load_deserialize_data()
print("Training X shape: " + str(genre_features.train_X.shape))
print("Training Y shape: " + str(genre_features.train_Y.shape))
print("Dev X shape: " + str(genre_features.valid_X.shape))
print("Dev Y shape: " + str(genre_features.valid_Y.shape))
print("Test X shape: " + str(genre_features.test_X.shape))
print("Test Y shape: " + str(genre_features.test_Y.shape))

input_shape = (genre_features.train_X.shape[1], genre_features.train_X.shape[2])

print("LSTM model ...")
model = Sequential()
model.add(LSTM(units=128, dropout=0.05, recurrent_dropout=0.35, return_sequences=True, input_shape=input_shape))
model.add(LSTM(units=64,  dropout=0.05, recurrent_dropout=0.35, return_sequences=True))
model.add(LSTM(units=32,  dropout=0.05, recurrent_dropout=0.35, return_sequences=False))
model.add(Dense(units=genre_features.train_Y.shape[1], activation="softmax"))

print("Compiling ...")
#es = EarlyStopping(monitor='val_loss', min_delta=0.00001, patience=4, mode='min', verbose=1)
ckpt = ModelCheckpoint(filepath='./dataset/ckpt/trilayer_mel_LSTM_adam--{epoch:02d}--{val_loss:.3f}.hdf5', monitor='val_accuracy', verbose=1, mode='max', save_best_only=True)
#ckpt = ModelCheckpoint(filepath='./dataset/ckpt/mel_LSTM_adam--{epoch:02d}--{val_loss:.3f}.hdf5', monitor='val_loss', verbose=1, mode='min', period=1)
opt = Adam()

##Save model structure as json file
model_json=model.to_json()
with open("./dataset/ckpt/trilayer_melLSTM_adam.json", "w") as json_file:
    json_file.write(model_json)

model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
model.summary()

print("Training ...")
batch_size = 35  # num of training examples per minibatch
n_epochs = 300
history = model.fit(
    genre_features.train_X,
    genre_features.train_Y,
    validation_data=(genre_features.valid_X, genre_features.valid_Y),
    batch_size=batch_size,
    epochs=n_epochs,
    callbacks=[ckpt],
    shuffle=True
)
pandas.DataFrame(history.history).to_csv("3layer_melLSTM_training_history.csv")

print("\nValidating ...")
score, accuracy = model.evaluate(
    genre_features.valid_X, genre_features.valid_Y, batch_size=batch_size, verbose=1
)
print("Validation loss:  ", score)
print("Validation accuracy:  ", accuracy)

print("\nTesting ...")
score, accuracy = model.evaluate(
    genre_features.test_X, genre_features.test_Y, batch_size=batch_size, verbose=1
)
print("Test loss:  ", score)
print("Test accuracy:  ", accuracy)
