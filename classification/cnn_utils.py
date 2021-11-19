from tensorflow.keras.utils import to_categorical
import processing.processing as processing
import random as r

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
    r.shuffle(arr)
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