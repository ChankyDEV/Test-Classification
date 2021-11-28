from cnn_utils import *
from data.reader import read, read_emotions_csv
from model import create_model
import random as r

class Dataset:
    
    def __init__(self, x_train,y_train,x_test,y_test,x_val,y_val,y_test_not_categorical):
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.x_val = x_val
        self.y_val = y_val
        self.y_test_not_categorical = y_test_not_categorical
        

class BinaryClassTextDataPrepration:
    
    def __get_max_length(self, first_arr, second_arr):
        first_arr_length = get_max_sentence_length(first_arr)
        second_arr_length = get_max_sentence_length(second_arr)
        return get_max_length(first_arr_length, second_arr_length)
    
    def __vectorize(self, first_class_data, second_class_data, max_len, tokenizer):
        second_class_data_vectorized = process(second_class_data, max_len, tokenizer)
        first_class_data_vectorized = process(first_class_data, max_len, tokenizer)
        return first_class_data_vectorized, second_class_data_vectorized
    
    def __label(self, arr, label):
        tuples = []
        for sentence in arr:
            tuples.append((sentence,label))
        return tuples
    
    def __shuffle(self, arr):
        r.shuffle(arr)
        return arr
    
    def __split_into_val_set(self, size, x_train, y_train):
        x_val =   x_train[0:size]
        y_val = y_train[0:size]
        x_train =  x_train[size:]
        y_train = y_train[size:]
        return x_val, y_val, x_train, y_train
    
    
    
    def __init__(self, first_class_data, second_class_data, train_set_percent, size_of_validation_dataset):        
        max_len = self.__get_max_length(first_class_data, second_class_data)
        tokenizer = Tokenizer(num_words=len([*first_class_data, *second_class_data]))
        first_class, second_class = self.__vectorize(first_class_data, second_class_data, max_len, tokenizer)
        first_class = self.__label(first_class, label = 0)
        second_class = self.__label(second_class, label = 1)
        
        first_class = self.__shuffle(first_class)
        second_class = self.__shuffle(second_class)
        
        first_class_train, first_class_test = split(first_class, train_set_percent)
        second_class_train, second_class_test = split(second_class, train_set_percent)
        
        train = join(second_class_train, first_class_train)
        test = join(second_class_test, first_class_test)
        
        train = shuffle(train)
        test = shuffle(test)
        
        x_train, y_train = get_data_and_labels(train)
        x_test, y_test = get_data_and_labels(test)
        
        y_test_not_categorical = y_test.copy()
        y_train, y_test = expand_labels(y_train, y_test)
        
        x_val, y_val, x_train, y_train = self.__split_into_val_set(size_of_validation_dataset,x_train,y_train)
        
        x_test = np.array(x_test)
        y_test = np.array(y_test)
        x_train = np.array(x_train)
        y_train = np.array(y_train)
        x_val = np.array(x_val)
        y_val = np.array(y_val)
        
        self.vocab_size = len(tokenizer.word_docs)
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.dataset = Dataset(x_train=x_train,
                               y_train=y_train,
                               x_test=x_test,
                               y_test=y_test,
                               x_val=x_val,
                               y_val=y_val,
                               y_test_not_categorical=y_test_not_categorical)
        
    
depressed_from_csv, non_depressed_from_csv = read_emotions_csv('G:\Python/text_classification/data/Emotion_final.csv')
depressed_from_txt = read(path='G:\Python/text_classification/data/sadness_data.txt')
non_depressed_from_txt = read(path='G:\Python/text_classification/data/other.txt')

depressed_together = [*depressed_from_txt, *depressed_from_csv]
non_depressed_together = [*non_depressed_from_txt, *non_depressed_from_csv]

binaryDataPreparation = BinaryClassTextDataPrepration(non_depressed_together, depressed_together, 0.8, 1700)
dataset = binaryDataPreparation.dataset
vocab_size = binaryDataPreparation.vocab_size
max_len = binaryDataPreparation.max_len
tokenizer = binaryDataPreparation.tokenizer


model = create_model(vocab_size=vocab_size, maxlen = max_len)
model_to_save, history = train_model(model=model,
                                    train=(dataset.x_train,dataset.y_train),              
                                    val=(dataset.x_val,dataset.y_val),
                                    epochs = 7)

accuracy, loss, predictions = test_model(model_to_save,(dataset.x_test,dataset.y_test))

print("Actual accuracy:  {:.4f}".format(accuracy))
print("Actual loss:  {:.4f}".format(loss))

classes = ['Charakter nie depresyjny', 'Charakter depresyjny']

plot(dataset.y_test_not_categorical, predictions, classes, history)

if accuracy > 0.80:
    save(model_to_save, tokenizer)
    print("New best accuracy:  {:.4f}".format(accuracy))
        
        
    
