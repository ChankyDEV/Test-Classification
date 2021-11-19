from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.text import Tokenizer
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np


def create_model(vocab_size, maxlen):
    model = Sequential()
    model.add(layers.Embedding(input_dim=vocab_size,                      
                               output_dim=5,
                               input_length=maxlen))
    model.add(layers.Conv1D(32,16,activation='relu'))
    model.add(layers.AvgPool1D())
    model.add(layers.Conv1D(24,8,activation='relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(10, activation='relu'))
    model.add(layers.Dense(2, activation='relu'))
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model
