import numpy as np
from tensorflow.python.keras.engine.sequential import Sequential
import processing.processing as processing
from classification.cnn_utils import *
from data.reader import read
from tensorflow.keras.models import load_model
import tensorflow as tf

model = load_model('G:/Python/text_classification/classification/text_model')
sadness = read(path='G:\Python/text_classification/data/sadness_data.txt')
joy = read(path='G:\Python/text_classification/data/other.txt')

dataset = join(sadness,joy)

sentence = "My life is nightmare"
sentence = processing.process(sentence,dataset)
sentence = tf.reshape(sentence,[1,286])
result = model.predict(sentence)
print("sad",result[0][0])
print("other",result[0][1])



