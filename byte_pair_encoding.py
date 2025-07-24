import re
from collections import Counter


def load(file_path):
    """A short function to load a file at a given path"""
    with open(file_path, "r") as file:
        return file.read()


def unix_copy(text):
    """The copy of the sorting unix console commands as python code"""
    text = re.sub(r"[^a-zA-Z0-9]+", " ", text)
    words = text.lower().split(" ")

    counts = Counter(words)
    counts = counts.most_common()
    return counts


def create_corpus(text):
    """Creates a corpus (list) from a given string"""
    corpus = []

    # reduce the corpus to lower case because the vocabulary is also only lower case
    text = text.lower()
    # replace all special characters and spaces with underscores
    text = re.sub(r"\s+", "_", text)

    for char in text:
        corpus.append(char)
    return corpus


def get_vocab(corpus):
    """Returns all unique characters inside a given input as a list"""
    vocab = []
    for character in corpus:
        vocab.append(character)
    return list(set(vocab))


def count_pairs(vocab, corpus):
    """Counts how often all possible combinations of tokens inside a given vocabulary occur inside a given corpus"""
    pairs = {}

    # dictionary of possible pairs
    for a in vocab:
        for b in vocab:
            if a == "_":
                pass
            pairs[a + b] = 0

    # iterate through the corpus and count the occurences of each pair
    for i in range(len(corpus)):
        try:
            if corpus[i] != "_":
                pairs[corpus[i] + corpus[i + 1]] += 1
        except:
            pass

    # sort the counts for all pairs
    pairs = sorted(pairs.items(), key=lambda item: item[1], reverse=True)
    # get string part of the tuple of the most common pairing
    new_token = pairs[0][0]
    # add the most common pair to the vocabulary
    vocab.append(new_token)

    new_corpus = []

    i = 0
    # merge all instances of the token pair in the corpus
    while i < len(corpus):
        try:
            if corpus[i] + corpus[i + 1] == new_token:
                new_corpus.append(new_token)
                i += 2
            else:
                new_corpus.append(corpus[i])
                i += 1
        except:
            i += 1
            pass
    print(
        "new token: "
        + new_token
        + " was added to the vocabulary and merged in the corpus"
    )
    return vocab, new_corpus


def merge_corpus(vocab, text):
    """Merges tokens in a given corpus according to a given vocabulary"""

    corpus = create_corpus(text)
    for token in vocab:
        new_corpus = []
        i = 0
        while i < len(corpus):
            try:
                if corpus[i] + corpus[i + 1] == token:
                    new_corpus.append(token)
                    i += 2
                else:
                    new_corpus.append(corpus[i])
                    i += 1
            except:
                i += 1
                corpus = new_corpus
                pass

    return corpus


def byte_pair_encoding(text: str, merges: int):
    """Executes byte pair encoding for a given text with a given number ob merges"""
    corpus = create_corpus(text)
    vocab = get_vocab(corpus)

    for _ in range(merges):
        vocab, corpus = count_pairs(vocab, corpus)

    return vocab, corpus


train = load("Shakespeare_clean_train.txt")

test = load("Shakespeare_clean_test.txt")

vocab, corpus = byte_pair_encoding(train, 200)

corpus2 = merge_corpus(vocab, test)
