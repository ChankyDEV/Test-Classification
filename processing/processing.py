from processing.tokenization import tokenize, expand
from processing.utils import get_max_sentence_length


def process_all(sentences, tokenizer, maxLength = 0):
    tokenized_sentences = tokenize(sentences, tokenizer)
    expanded_sentences = expand(tokenized_sentences, maxLength= maxLength)
    return expanded_sentences


def process(sentence_to_process, all_sentences):
    all_sentences.append(sentence_to_process)
    max_length = get_max_sentence_length(all_sentences)
    tokenized_sentences = tokenize(all_sentences)
    expanded_sentences = expand(tokenized_sentences, max_length)
    proccesed_sentence = expanded_sentences[-1]
    all_sentences.pop()
    return proccesed_sentence
