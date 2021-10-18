
def read(path):
    f=open(path, "rb")
    lines=f.readlines()
    sentences = []
    for line in lines:
        line = line.decode('utf-8')
        parts=line.split("\t")
        if len(parts)==4:
            sentence=parts[1]
            sentences.append(sentence)
    f.close()  
    return sentences


