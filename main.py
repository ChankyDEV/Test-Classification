from keras.engine.saving import load_model
import processing.processing as processing
from classification.cnn_utils import *


model = load_model('G:/Python/text_classification/classification/text_classification_model.h5')
sadness = read(path='G:\Python/text_classification/data/sadness_data.txt')
joy = read(path='G:\Python/text_classification/data/other.txt')

dataset = join(sadness,joy)

sentence = "Im lonely"
sentence = processing.process(sentence,dataset)

result = model.predict([[sentence]])
print("sad",result[0][0])
print("other",result[0][1])



