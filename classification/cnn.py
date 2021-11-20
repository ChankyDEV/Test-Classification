from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.preprocessing.text import Tokenizer
from cnn_utils import *
from data.reader import read, read_emotions_csv
from processing.utils import get_max_sentence_length
from model import create_model


depressed_from_csv, non_depressed_from_csv = read_emotions_csv('G:\Python/text_classification/data/Emotion_final.csv')
depressed_from_txt = read(path='G:\Python/text_classification/data/sadness_data.txt')
non_depressed_from_txt = read(path='G:\Python/text_classification/data/other.txt')

depressed_together = [*depressed_from_txt, *depressed_from_csv]
non_depressed_together = [*non_depressed_from_txt, *non_depressed_from_csv]

x_train, y_train, x_test, y_test, x_val,y_val, vocab_size, maxlen, tokenizer, test_labels = prepare_data(depressed_together, non_depressed_together) 


print('TRAIN SET:',len(x_train))
print('TEST SET:',len(x_test))
print('VAL SET:',len(x_val))
print('WORDS COUNT:', vocab_size)

# 0.7479
actualMax = 0.8683
model = create_model(vocab_size=vocab_size, maxlen = maxlen)
acc, model_to_save = learn_model(model=model,
                                 train=(x_train,y_train),              
                                 test=(x_test,y_test),
                                 val=(x_val,y_val),
                                 epochs = 11, 
                                 test_labels=test_labels
                                 )
if acc > actualMax:
    actualMax = acc
    save(model_to_save, tokenizer)
    print("New best accuracy:  {:.4f}".format(actualMax))
      
    
    
