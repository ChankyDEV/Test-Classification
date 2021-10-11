def get_max_sentence_length(tokens_array):
    length_of_sentences = [len(i) for i in tokens_array]
    index_of_max_length_sentence = length_of_sentences.index(max(length_of_sentences))
    max_length_sentence = tokens_array[index_of_max_length_sentence]
    length_of_max_length_sentence = len(max_length_sentence)
    return length_of_max_length_sentence
