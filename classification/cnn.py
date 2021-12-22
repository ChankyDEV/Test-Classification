from cnn_utils import *
from data.reader import read, read_emotions_csv
from model import create_model
from classification.binary_class_text_data_preparation import BinaryClassTextDataPrepration
from tensorflow.keras.layers import Embedding, Conv1D, MaxPool1D, Flatten, Dense, Dropout
import tensorflow as tf
        
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


print(dataset.__str__())

embedding_layer = Embedding(input_dim=vocab_size+1,                      
                                     output_dim=3, 
                                     input_length=max_len)
result = embedding_layer(tf.constant([31.0,43.0,125.0,74.0]))
resu = result.numpy()
print(resu)

model = create_model(layers = [
                            Embedding(input_dim=vocab_size+1,                      
                                    output_dim=44, 
                                    input_length=max_len),
                            Conv1D(1,5,activation='relu'),
                            MaxPool1D(),
                            Conv1D(3,3,activation='relu'),
                            MaxPool1D(),
                            Conv1D(2,3,activation='relu'),
                            MaxPool1D(),
                            Conv1D(1,3,activation='relu'),
                            MaxPool1D(),
                            Flatten(),
                            Dense(2, activation='sigmoid')],
                    optimizer='adam',
                    loss='binary_crossentropy',)

model_to_save, history = train_model(model=model,
                                    train=(dataset.x_train,dataset.y_train),              
                                    val=(dataset.x_val,dataset.y_val),
                                    epochs = 10)


accuracy, loss, predictions = test_model(model_to_save,(dataset.x_test,dataset.y_test))

print("Actual accuracy:  {:.4f}".format(accuracy))
print("Actual loss:  {:.4f}".format(loss))

classes = ['Charakter nie depresyjny', 'Charakter depresyjny']

plot(dataset.y_test_not_categorical, predictions, classes, history)

if accuracy > 0.50:
    save(model_to_save, tokenizer)
    print("New best accuracy:  {:.4f}".format(accuracy))
        
        
    
