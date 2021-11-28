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


def get_grater(x,y):
    if x > y:
        return x
    else:
        return y 

def random(min,max):
    return r.randint(min,max)



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
