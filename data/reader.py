import csv

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

def read_csv(path):
    with open(path,"rt",encoding="utf-8") as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=';')
        line_count = 0
        depressed = []
        non_depressed = []
        proccesed = 0
        for text in csv_reader:
            if line_count == 0:
                line_count += 1
            else:
                row = text[0].split(';')
                if len(row) == 3:
                    if row[2] == 0:
                        non_depressed.append(row[1])
                    else:
                        depressed.append(row[1])
                    proccesed +=1
                line_count += 1
        return depressed, non_depressed

def read_emotions_csv(path):
    with open(path,"rt",encoding="utf-8") as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        depressed = []
        non_depressed = []
        proccesed = 0
        for text in csv_reader:
            if line_count == 0:
                line_count += 1
            else:
                if text[1] == 'sadness' or text[1] == 'fear' or text[1] == 'anger':
                    depressed.append(text[0])
                else:
                    non_depressed.append(text[0])
                line_count += 1
        return depressed, non_depressed
    