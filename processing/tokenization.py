from keras_preprocessing.sequence import pad_sequences
from keras_preprocessing.text import Tokenizer

from processing.utils import get_max_sentence_length


def expand(tokenized_sentences):
    max_sentence_length = get_max_sentence_length(tokenized_sentences)
    return pad_sequences(tokenized_sentences, padding='post', maxlen=max_sentence_length)


def tokenize(sentences):
    tokenizer = Tokenizer(num_words=5000)
    tokenizer.fit_on_texts(sentences)
    return tokenizer.texts_to_sequences(sentences)
