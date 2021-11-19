from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer


def expand(tokenized_sentences, maxLength):
    return pad_sequences(tokenized_sentences, padding='post', maxlen=maxLength, dtype='float32')


def tokenize(sentences, tokenizer):
    tokenizer.fit_on_texts(sentences)
    return tokenizer.texts_to_sequences(sentences)
