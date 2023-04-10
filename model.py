import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

def modle() :
    model = Sequential()
    model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(15,126)))
    model.add(LSTM(128, return_sequences=True, activation='relu'))
    model.add(LSTM(64, return_sequences=False, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(4, activation='softmax'))
    model.load_weights('action1.h5')
    return model