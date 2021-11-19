from tensorflow.keras.models import Sequential
from tensorflow.keras import layers


def create_model(vocab_size, maxlen):
    model = Sequential()
    model.add(layers.Embedding(input_dim=vocab_size+1,                      
                               output_dim=64,
                               input_length=maxlen))
    model.add(layers.Conv1D(32,16,activation='relu'))
    model.add(layers.MaxPool1D())
    model.add(layers.Flatten())
    model.add(layers.Dense(16, activation='relu'))
    model.add(layers.Dense(2, activation='relu'))
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model
