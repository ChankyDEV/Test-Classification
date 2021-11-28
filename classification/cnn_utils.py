from tensorflow.keras.utils import to_categorical
import processing.processing as processing
import random as r
import io
import json
import matplotlib.pyplot as plt
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.preprocessing.text import Tokenizer
from processing.utils import get_max_sentence_length
import numpy as np
from datetime import date
from sklearn.metrics import ConfusionMatrixDisplay


def process(sentences, maxlen, tokenizer):
    proccessed_sentences = processing.process_all(sentences, tokenizer, maxlen)
    return proccessed_sentences

def label(arr,label):
    tuples = []
    for sentence in arr:
        tuples.append((sentence,label))
    return tuples

def join(firstArr,secondArr):
    return [*firstArr, *secondArr]

def get_max_length(x,y):
    if x > y:
        return x
    else:
        return y

def split(arr, percentage):
    arr_length = len(arr)-1
    train_set_length = int(len(arr)*percentage)
    train = []
    test = []
    for i in range(train_set_length):
        random_index = random(0,arr_length)
        train.append(arr[random_index])
        del arr[random_index]
        arr_length = len(arr)-1
    test = arr
    return train,test    

def random(min,max):
    return r.randint(min,max)


def shuffle(arr):
    r.shuffle(arr)
    return arr

def get_data_and_labels(tuples):
    data = []
    labels = []
    for tuple in tuples:
        data.append(tuple[0])
        labels.append(tuple[1])
    return data, labels

def expand_labels(first_arr, second_arr):
    first_arr = to_categorical(first_arr, num_classes=2)
    second_arr = to_categorical(second_arr, num_classes=2)
    return first_arr, second_arr

def save(model: Sequential, tokenizer_to_save: Tokenizer):
    print('SAVING...')
    model.save(f'G:/Python/text_classification/classification/models/model_{date.today()}.h5')
    tokenizer_json = tokenizer_to_save.to_json()
    with io.open('tokenizer.json', 'w', encoding='utf-8') as f:
        f.write(json.dumps(tokenizer_json, ensure_ascii=False))


def train_model(model,train, val, epochs):
    history = model.fit(train[0], 
                        train[1], 
                        epochs=epochs, 
                        verbose=False, 
                        validation_data=val
                        )
    return model, history


def test_model(model, test:tuple):
    loss, accuracy = model.evaluate(test[0], 
                                    test[1], 
                                    verbose=False
                                    )
    preds = model.predict(test[0])
    predictions = []
    for pred in preds:
        if pred[0] > pred[1]:
            predictions.append(0)
        else:
            predictions.append(1)   
                    
    return accuracy, loss, predictions


def plot(test_labels, predictions, displayed_labels, model_history):
    plt.figure(0)
    ConfusionMatrixDisplay.from_predictions(test_labels, predictions,
                                            cmap=plt.cm.Blues, 
                                            display_labels=displayed_labels)
    plt.show()
    
    plt.figure(1)
    plt.plot(model_history.history['accuracy'])
    plt.plot(model_history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    
    # summarize history for loss
    plt.figure(2)
    plt.plot(model_history.history['loss'])
    plt.plot(model_history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()



def prepare_data(depressed_together, non_depressed_together):
    depressed_together_length = get_max_sentence_length(depressed_together)
    non_depressed_together_length = get_max_sentence_length(non_depressed_together)

    maxlen = get_max_length(depressed_together_length,non_depressed_together_length)

    tokenizer = Tokenizer(num_words=30000)

    depressed = process(depressed_together, maxlen, tokenizer)
    non_depressed = process(non_depressed_together, maxlen, tokenizer)

    depressed = label(depressed, label = 1)
    non_depressed = label(non_depressed, label = 0)

    depressed = shuffle(depressed)
    non_depressed = shuffle(non_depressed)

    depressed_train, depressed_test = split(depressed, 0.8)
    non_depressed_train, non_depressed_test = split(non_depressed, 0.8)

    train = join(depressed_train,non_depressed_train)
    test = join(depressed_test,non_depressed_test)

    train = shuffle(train)
    test = shuffle(test)

    x_train, y_train = get_data_and_labels(train)
    x_test, y_test = get_data_and_labels(test)
    
    test_labels = y_test.copy()

    y_train, y_test = expand_labels(y_train, y_test)

    size_of_val_set = 1700

    x_val =   x_train[0:size_of_val_set]
    y_val = y_train[0:size_of_val_set]

    x_train =  x_train[size_of_val_set:]
    y_train = y_train[size_of_val_set:]

    x_test = np.array(x_test)
    y_test = np.array(y_test)

    x_train = np.array(x_train)
    y_train = np.array(y_train)

    x_val = np.array(x_val)
    y_val = np.array(y_val)
    
    print('TRAIN SET:',len(x_train))
    print('TEST SET:',len(x_test))
    print('VAL SET:',len(x_val))
    
    return x_train, y_train, x_test, y_test, x_val,y_val, len(tokenizer.word_docs), maxlen, tokenizer, test_labels