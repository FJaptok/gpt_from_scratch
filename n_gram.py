from byte_pair_encoding import create_corpus, merge_corpus, load




def load_pickle(file_path):
    """ Loads a vocabulary from a pickle file """
    import pickle
    with open(file_path, 'rb') as f:
        return pickle.load(f)


def uni_gram(corpus, vocab, norm="laplace"):
    """ Merges unigrams in a given corpus according to a given vocabulary """
    counter = {}
    if norm=="laplace":
        increment = 1
    else:
        increment = 0
    
    for token in corpus:
        if token in vocab:
            counter[token] = counter.get(token, increment) + 1
    return counter

def bi_gram(corpus, vocab, norm="laplace"):
    """ Merges bigrams in a given corpus according to a given vocabulary """
    counter = {}
    if norm == "laplace":
        increment = 1
    else:
        increment = 0
    
    for token_1 in vocab:
        for token_2 in vocab:
            token = (token_1, token_2)
            counter[token] = increment
    
    
    for i in range(len(corpus) - 1):
        token = (corpus[i] , corpus[i + 1])
        if token in counter:
            counter[token] += 1
    return counter

def n_gram(corpus, vocab, n, norm="laplace"):
    
    counter_uni = {}
    counter_bi = {}
    counter_tri = {}
    counter_four = {}

    if norm == "laplace":
        increment = 1
    else:
        increment = 0

    """
    for token_1 in vocab:
        counter_uni[token_1] = increment
        if n == 1: continue
        for token_2 in vocab:
            token = (token_1, token_2)
            counter_bi[token] = increment
            if n == 2: continue
            for token_3 in vocab:
                token = (token_1, token_2, token_3)
                counter_tri[token] = increment
                if n == 3: continue
                for token_4 in vocab:
                    token = (token_1, token_2, token_3, token_4)
                    counter_four[token] = increment
    
    """
    
    for i in range(len(corpus) - 1):
        token = (corpus[i]) 
        if token in counter_uni:
            counter_uni[token] += 1
        else:
            counter_uni[token] = 1 + increment
        if n == 1: continue
        
        try:
            token = (corpus[i] , corpus[i + 1])
            if token in counter_bi:
                counter_bi[token] += 1
            else:
                counter_bi[token] = 1 + increment
            if n == 2: continue
        except:
            continue
        
        try:
            token = (corpus[i] , corpus[i + 1], corpus[i + 2])
            if token in counter_tri:
                counter_tri[token] += 1
            else:
                counter_tri[token] = 1 + increment
            if n == 3: continue
        except:
            continue
        
        try:
            token = (corpus[i] , corpus[i + 1], corpus[i + 2], corpus[i + 3])
            if token in counter_four:
                counter_four[token] += 1
            else:
                counter_four[token] = 1 + increment
        except:
            continue

    counters = [counter_uni, counter_bi, counter_tri, counter_four]
    return counters[:n]
        
            
path_2_pickle = 'vocab.pickle'
path_2_train = "Shakespeare_clean_train.txt"
# load pickle vocab
created_vocab = load_pickle(path_2_pickle)



corpus3 = load_pickle('corpus3.pickle')

counter_uni = uni_gram(corpus3, created_vocab)

counter_bi = bi_gram(corpus3, created_vocab, "laplace")


counters = n_gram(corpus3, created_vocab, 4)
counter_uni_test = counters[0]
counter_bi_test = counters[1]
counter_tri_test = counters[2]
counter_four_test = counters[3]


joined_probability = []


i = 0
for pair_2 in counter_four_test:
    

    token_1_1, token_2_1, token_3_1, token_4_1 = pair_2
    
    uni_prob = counter_uni[token_1_1]
    
    bi_prob = counter_bi[(token_1_1, token_2_1)]


    uni_1 = counter_uni_test[token_1_1]
    bi_1 = counter_bi_test[(token_1_1, token_2_1)]
    tri_1 = counter_tri_test[(token_1_1, token_2_1, token_3_1)]
    four_1 = counter_four_test[(token_1_1, token_2_1, token_3_1, token_4_1)]



    if i > 1_000:

        print(f"joined prob original bi {(token_1_1, token_2_1)} :", bi_prob / uni_prob)

        print(f"joined prob n bi {(token_1_1, token_2_1)} :", bi_1 / uni_1)

        print(f"joined prob n tri {(token_1_1, token_2_1, token_3_1)} :", tri_1 / bi_1)

        print(f"joined prob n four {pair_2} :", four_1 / tri_1, "\n")



    if i == 1_100:
        break
    i += 1

