import random as r
from tensorflow.keras.utils import to_categorical
from tensorflow.python.keras.preprocessing.text import Tokenizer
from cnn_utils import get_grater, random, process, get_max_sentence_length
import numpy as np
from classification.dataset import Dataset


class BinaryClassTextDataPrepration:
    
    def __get_max_length(self, first_arr, second_arr):
        first_arr_length = get_max_sentence_length(first_arr)
        second_arr_length = get_max_sentence_length(second_arr)
        return get_grater(first_arr_length, second_arr_length)
    
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
    
    
    def __split(self, arr, percent):
        arr_length = len(arr)-1
        train_set_length = int(len(arr)*percent)
        train = []
        test = []
        for i in range(train_set_length):
            random_index = random(0,arr_length)
            train.append(arr[random_index])
            del arr[random_index]
            arr_length = len(arr)-1
        test = arr
        return train,test  
    
    def __join(self, first_arr, second_arr):
        return [*first_arr, *second_arr]
    
    def __shuffle(self, arr):
        r.shuffle(arr)
        return arr
    
    def __split_for_set_and_labels(self, tuples):
        data = []
        labels = []
        for tuple in tuples:
            data.append(tuple[0])
            labels.append(tuple[1])
        return data, labels
    
    def __expand_labels(self, first_arr, second_arr):
        first_arr = to_categorical(first_arr, num_classes=2)
        second_arr = to_categorical(second_arr, num_classes=2)
        return first_arr, second_arr
    
    def __init__(self, first_class_data, second_class_data, train_set_percent, size_of_validation_dataset):        
        max_len = self.__get_max_length(first_class_data, second_class_data)
        tokenizer = Tokenizer(num_words=len([*first_class_data, *second_class_data]))
        first_class, second_class = self.__vectorize(first_class_data, second_class_data, max_len, tokenizer)
        first_class = self.__label(first_class, label = 0)
        second_class = self.__label(second_class, label = 1)
        
        first_class = self.__shuffle(first_class)
        second_class = self.__shuffle(second_class)
        
        first_class_train, first_class_test = self.__split(first_class, train_set_percent)
        second_class_train, second_class_test = self.__split(second_class, train_set_percent)
        
        train = self.__join(second_class_train, first_class_train)
        test = self.__join(second_class_test, first_class_test)
        
        train = self.__shuffle(train)
        test = self.__shuffle(test)
        
        x_train, y_train = self.__split_for_set_and_labels(train)
        x_test, y_test = self.__split_for_set_and_labels(test)
        
        y_test_not_categorical = y_test.copy()
        y_train, y_test = self.__expand_labels(y_train, y_test)
        
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