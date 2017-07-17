import numpy as np
import os
from string import ascii_lowercase as al
from collections import Counter
import pickle

def text_from_seed(model, name, ind_to_char, char=False):
    seed = np.random.normal(size=(1, 100))
    output = model.predict(seed)
    tweet = ''
    for i in range(output.shape[1]):
        if char is True:
            tweet += ind_to_char[np.argmax(output[0][i])]
        else:
            tweet += ind_to_char[np.argmax(output[0][i])] + ' '
    print(tweet)
    with open(name, mode='w') as f:
        f.write(tweet)


# This does two passes through the data, the first one to figure out what are the characters
# or words that interest us, getting rid of those
def convert_text_to_nptensor(directory='../Data/1-billion-word/', cutoff=5, min_frequency_words=10000, max_lines=2000000, name='Google', chars=True):
    words = Counter()
    n_lines = 0
    files = []
    for file in os.listdir(directory):
            files.append(directory + file)
    for file_idx in range(len(files)):
        with open(files[file_idx], encoding='utf-8') as f:
            text = f.readlines()
        for line in text:
            line = line.replace('\n', '').replace('.', '').replace('!', '').replace('?', '')
            line = line.lower()
            if chars is True:
                words.update(list(line))
                n_lines += len(list(line))//cutoff
            else:
                words.update(line.split(' '))
                n_lines += len(line.split(' '))//cutoff

    number_of_words = 0
    removed = 0
    words_init = len(words)
    for k in list(words):
        number_of_words += words[k]
        if words[k] < min_frequency_words:
            removed += words[k]
            del words[k]
    print('% of raw words remaining :', (number_of_words - removed)/number_of_words*100.0)
    print('Initial amount of tokens :', words_init)
    print('Current amount of tokens :', len(words))
    print('% of remaining tokens :', len(words)/words_init)
    print('Max amount of lines :', n_lines)

    word_to_ind = dict((c, i) for i, c in enumerate(list(set(words))))
    ind_to_word = dict((i, c) for i, c in enumerate(list(set(words))))
    X = np.zeros((min(max_lines, n_lines), cutoff, len(word_to_ind)), dtype=np.bool)

    n_lines = 0
    files = []
    for file in os.listdir(directory):
        files.append(directory + file)
    for file_idx in range(len(files)):
        with open(files[file_idx], encoding='utf-8') as f:
            text = f.readlines()
        for line in text:
            line = line.replace('\n', '')
            line = line.lower()
            if chars is True:
                line = list(line)
            else:
                line = line.split(' ')
            offset = 0
            while(len(line) < offset + cutoff) & (n_lines < max_lines):
                check_word = True
                for t, word in enumerate(line[offset: offset+cutoff]):
                    try:
                        word_to_ind[word]
                    except KeyError:
                        check_word = False
                if check_word == False:
                    offset += cutoff
                    pass
                else:
                    for t, word in enumerate(line[offset: offset + cutoff]):

                        X[n_lines, t, word_to_ind[word]] = 1
                    offset += cutoff
                    n_lines += 1

    print(n_lines)
    np.save('../text_' + name, X)
    with open('../ind_to_word_' +  name + '.pickle', 'wb') as pck:
        pickle.dump(ind_to_word, pck)

if __name__ == '__main__':
    convert_text_to_nptensor(chars=False)