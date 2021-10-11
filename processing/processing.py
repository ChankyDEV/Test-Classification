from processing.tokenization import tokenize, expand


def process_all(sentences):
    tokenized_sentences = tokenize(sentences)
    expanded_sentences = expand(tokenized_sentences)
    return expanded_sentences


def process(sentence_to_process, all_sentences):
    all_sentences.append(sentence_to_process)
    tokenized_sentences = tokenize(all_sentences)
    expanded_sentences = expand(tokenized_sentences)
    proccesed_sentence = expanded_sentences[-1]
    all_sentences.pop()
    return proccesed_sentence
