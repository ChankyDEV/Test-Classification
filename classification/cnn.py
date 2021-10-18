from cnn_utils import *

ACTUAL_BEST_ACCURACY = 1.0000

sadness = read(path='G:\Python/text_classification/data/sadness_data.txt')
joy = read(path='G:\Python/text_classification/data/other.txt')

dataset = join(sadness,joy)

sadness_length = get_max_sentence_length(sadness)
joy_length = get_max_sentence_length(joy)

maxlen = get_max_length(sadness_length,joy_length)

sadness = process(sadness, maxlen)
joy = process(joy, maxlen)

sadness = label(sadness,label = 0)
joy = label(joy,label = 1)

sadness_train, sadness_test = split(sadness, 0.85)
joy_train, joy_test = split(joy, 0.85)

train = join(sadness_train,joy_train)
test = join(sadness_test,joy_test)

train = shuffle(train)
test = shuffle(test)

train_data, train_labels = get_data_and_labels(train)
test_data, test_labels = get_data_and_labels(train)

train_labels, test_labels = expand_labels(train_labels, test_labels)

train_data = np.array(train_data)
train_labels = np.array(train_labels)

test_data = np.array(test_data)
test_labels = np.array(test_labels)

model = create_model(vocab_size=50000, maxlen=maxlen)
history = model.fit(train_data, train_labels, epochs=60, verbose=False)
loss, accuracy = model.evaluate(test_data, test_labels, verbose=False)
print("Testing Accuracy:  {:.4f}".format(accuracy))

if accuracy > ACTUAL_BEST_ACCURACY:
    model.save('G:/Python/text_classification/classification/text_classification_model.h5')
