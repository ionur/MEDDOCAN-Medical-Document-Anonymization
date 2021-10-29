import pickle
import pprint
import argparse
import numpy as np


pp = pprint.PrettyPrinter()
parser = argparse.ArgumentParser()

parser.add_argument('--trainpickle', type=str, required=True)
parser.add_argument('--devpickle', type=str, required=True)
parser.add_argument('--testpickle', type=str, required=True)
parser.add_argument('--embfile', type=str, required=True)
parser.add_argument('--vocabEmbFile', type=str, required=True)

UNK = "$UNK$"
NUM = "$NUM$"

def get_vocabs(all_sentences):

    vocab_words = set()
    vocab_tags = set()
    vocab_chars = set()

    for sent in all_sentences:
        for i, (word, label, startIndex, postag) in enumerate(sent):
            # print("word == ", word.lower())
            # exit(1)
            vocab_words.add(word.lower())
            vocab_tags.add(label)
            for char in word:
                vocab_chars.add(char)
    print("- done. {} tokens".format(len(vocab_words)))
    return vocab_words, vocab_tags, vocab_chars


def get_fasttext_vocab(filename):
    print("Building vocab...")
    vocab = set()
    with open(filename) as f:
        next(f)
        for line in f:
            word = line.strip().split(' ')[0]
            vocab.add(word)
    # print("- done. {} tokens".format(len(vocab)))
    return vocab

def export_fasttext_vectors(vocab, glove_filename, trimmed_filename, dim):
    embeddings = np.zeros([len(vocab), dim])
    with open(glove_filename) as f:
        next(f)
        for line in f:
            line = line.strip().split(' ')
            word = line[0]
            embedding = [float(x) for x in line[1:]]
            if word in vocab:
                word_idx = vocab[word]
                embeddings[word_idx] = np.asarray(embedding)

    np.savez_compressed(trimmed_filename, embeddings=embeddings)


def load_vocab(vocab):
    d = dict()
    for idx, word in enumerate(vocab):
        word = word.strip()
        d[word] = idx
    return d


if __name__ == '__main__':
    args = parser.parse_args()
    pp.pprint(args)

    all_sentences = []

    file = open(args.trainpickle,'rb')
    train_sentences = pickle.load(file)
    file.close()

    file = open(args.devpickle,'rb')
    dev_sentences = pickle.load(file)
    file.close()

    file = open(args.testpickle,'rb')
    test_sentences = pickle.load(file)
    file.close()

    all_sentences += [i for x in train_sentences.values() for i in x]
    all_sentences += [i for x in dev_sentences.values() for i in x]
    all_sentences += [i for x in test_sentences.values() for i in x]

    vocab_words, vocab_tags, vocab_chars = get_vocabs(all_sentences)

    # Build Word and Tag vocab
    vocab_glove = get_fasttext_vocab(args.embfile)

    vocab = vocab_words & vocab_glove
    vocab.add(UNK)
    vocab.add(NUM)

    print("Words Vocab Size: ", len(vocab_words))
    print("Tags Vocab Size: ", len(vocab_tags))
    print("Chars Vocab Size", len(vocab_chars))

    print("Loading vocab embeddings...")
    vocab = load_vocab(vocab)
    export_fasttext_vectors(vocab, args.embfile, args.vocabEmbFile, 300)


    print("Generating pickles...")
    with open('./words.pickle', 'wb') as handle:
        pickle.dump(vocab, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open('./tags.pickle', 'wb') as handle:
        pickle.dump(vocab_tags, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open('./chars.pickle', 'wb') as handle:
        pickle.dump(vocab_chars, handle, protocol=pickle.HIGHEST_PROTOCOL)




