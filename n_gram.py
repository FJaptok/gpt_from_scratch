from byte_pair_encoding import create_corpus, merge_corpus, load




def load_pickle(file_path):
    """ Loads a vocabulary from a pickle file """
    import pickle
    with open(file_path, 'rb') as f:
        return pickle.load(f)


def uni_gram(corpus, vocab):
    """ Merges unigrams in a given corpus according to a given vocabulary """
    counter = {}
    for token in corpus:
        if token in vocab:
            counter[token] = counter.get(token, 0) + 1
    return counter

path_2_pickle = 'vocab.pickle'
path_2_train = "Shakespeare_clean_train.txt"
# load pickle vocab
created_vocab = load_pickle(path_2_pickle)

#train = load(path_2_train)

#corpus2 = create_corpus(train)

#corpus3 = merge_corpus(created_vocab,corpus2)
#import pickle
#with open('corpus3.pickle', 'wb') as f:
#    pickle.dump(corpus3, f)

corpus3 = load_pickle('corpus3.pickle')

#print(corpus3[:1000]) 

counter = uni_gram(corpus3, created_vocab)

print(counter)