from cnn_utils import *
from data.reader import read, read_emotions_csv
from model import create_model
from classification.binary_class_text_data_preparation import BinaryClassTextDataPrepration
        
    
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

if accuracy > 0.90:
    save(model_to_save, tokenizer)
    print("New best accuracy:  {:.4f}".format(accuracy))
        
        
    
