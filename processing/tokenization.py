from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer

from processing.utils import get_max_sentence_length


def expand(tokenized_sentences, maxLength):
    return pad_sequences(tokenized_sentences, padding='post', maxlen=maxLength)


def tokenize(sentences):
    tokenizer = Tokenizer(num_words=5000)
    tokenizer.fit_on_texts(sentences)
    return tokenizer.texts_to_sequences(sentences)
