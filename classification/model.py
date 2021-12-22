from tensorflow.keras.models import Sequential
from tensorflow.python.keras.backend import dropout


def create_model(layers, optimizer, loss):
    model = Sequential(layers)
    model.compile(optimizer=optimizer,
                  loss=loss,
                  metrics=['accuracy'])
    return model



 # model = Sequential()
    # model.add(layers.Embedding(input_dim=vocab_size+1,                      
    #                            output_dim=16, input_length=maxlen))
    # model.add(layers.Dropout(0.35))
    # model.add(layers.Conv1D(3,5,activation='relu'))
    # model.add(layers.Dropout(0.33))
    # model.add(layers.Conv1D(3,3,activation='relu'))
    # model.add(layers.Dropout(0.31))
    # model.add(layers.Conv1D(3,3,activation='relu'))
    # model.add(layers.Flatten())
    # model.add(layers.Dense(2, activation='sigmoid'))
    # model.compile(optimizer='adam',
    #               loss='binary_crossentropy',
    #               metrics=['accuracy'])