import os
from model import NeuralModel
import numpy as np
import pickle
from collections import OrderedDict

UNK = "$UNK$"
NUM = "$NUM$"

def load_vocab(vocab):
    d = dict()
    for idx, word in enumerate(vocab):
        word = word.strip()
        d[word] = idx
    return d


def get_fasttext_vectors(filename):
    try:
        with np.load(filename) as data:
            return data["embeddings"]
    except IOError:
        print("File not found: ", filename)
        exit(1)

def get_processing_word(word, vocab_words=None, vocab_chars=None,
                    lowercase=False, chars=False, allow_unk=True):
    if vocab_chars is not None and chars == True:
        char_ids = []
        for char in word:
            # ignore chars out of vocabulary
            if char in vocab_chars:
                char_ids += [vocab_chars[char]]

    if lowercase:
        word = word.lower()
    if word.isdigit():
        word = NUM

    if vocab_words is not None:
        if word in vocab_words:
            word = vocab_words[word]
        else:
            if allow_unk:
                word = vocab_words[UNK]
            else:
                raise Exception("Unknow key is not allowed.")

    if vocab_chars is not None and chars == True:
        return char_ids, word
    else:
        return word

def main():
    config = Config()
    model = NeuralModel(config)
    model.build()
    model.train()

    print("Evaluating the best model...")

    print("Training data evaluation:")
    sentences = config.all_train_sentences_preprocessed
    scores = model.evaluate(sentences, "train_predictions.txt")
    print(scores)

    print("Dev data evaluation:")
    sentences = config.all_dev_sentences_preprocessed
    scores = model.evaluate(sentences, "dev_predictions.txt")
    print(scores)

    print("Test data evaluation:")
    sentences = config.all_test_sentences_preprocessed
    scores = model.evaluate(sentences, "test_predictions.txt")
    print(scores)

    #Evaluate using MEDDOCAN

    # Train
    print("MEDDOCAN ----- Train data")
    with open('../train_predictions.txt') as f:
        y_pred = f.read().split()

    pathRead = "../../../train/system/"
    pathOut = "../../../train/system/"
    # create an .ann format for predictions
    createAnnFormat(config.train_sentences, y_pred, pathRead, pathOut)

    # Dev
    print("MEDDOCAN ----- Dev data")
    with open('../dev_predictions.txt') as f:
        y_pred = f.read().split()

    pathRead = "../../../dev/system/"
    pathOut = "../../../dev/system/"
    # create an .ann format for predictions
    createAnnFormat(config.dev_sentences, y_pred, pathRead, pathOut)

    # Test
    print("MEDDOCAN ----- Test data")
    with open('../test_predictions.txt') as f:
        y_pred = f.read().split()

    pathRead = "../../../output/test/system/"
    pathOut = "../../../output/test/system/"
    # create an .ann format for predictions
    createAnnFormat(config.test_sentences, y_pred, pathRead, pathOut)



# create an .ann format for predictions
def createAnnFormat(preprocess_dict, y_pred, pathRead, pathOut):
    j = 0
    for docId, v in preprocess_dict.items():
        print('Generating .ann for ', docId)
        txt_start_end = ""
        tmp_tags = []
        curr_tag = ""

        for sent in v:
            for item in sent:
                word = item[0]
                word_start = item[2]
                word_end = word_start + len(word) - 1
                pred = y_pred[j]

                # if prediction is a beginning, add the tag to the tag list
                if pred[0:2] == "B-":
                    if txt_start_end != "":
                        tmp_tags.append((txt_start_end, curr_tag))
                        txt_start_end = ""
                    curr_tag = pred[2:]
                    txt_start_end += "" + str(word_start) + "-" + str(word_end) + ","
                # if it is a contuniation keep adding
                elif pred[0:2] == "I-":
                    txt_start_end += "" + str(word_start) + "-" + str(word_end) + ","
                j += 1
        if txt_start_end != "":
            tmp_tags.append((txt_start_end, curr_tag))
            txt_start_end = ""
        # get sentence matching that
        with open(pathRead + docId + ".txt", "r") as f:
            complete_doc = f.read()

            # now for all the text, get their exact match and create tags
        tags = []
        t_count = 1
        for i, i_tag in tmp_tags:
            splits = i.split(",")[:-1]
            beginning_split = splits[0].split("-")
            real_start = int(beginning_split[0])
            if len(splits) == 1:
                real_end = int(beginning_split[1]) + 1
            else:
                end_split = splits[len(splits) - 1].split("-")
                real_end = int(end_split[1]) + 1
            text_appearing = complete_doc[real_start:real_end].split("\n")[0]
            tags.append(
                "T" + str(t_count) + "\t" + i_tag + " " + str(real_start) + " " + str(real_end) + "\t" + text_appearing)
            t_count += 1
        # now output all these
        with open(pathOut + docId + ".ann", "w") as out:
            for i in tags:
                out.write(i + "\n")


class Config():
    def __init__(self):
        file = open('../../train_word_ner_startidx_dict.pickle', 'rb')
        self.train_sentences = pickle.load(file)
        self.train_sentences = OrderedDict(self.train_sentences)
        file.close()

        self.train_docsIds_list = []

        for k, v in self.train_sentences.items():
            self.train_docsIds_list.append(k)

        with open('../trainDocIds.pickle', 'wb') as handle:
            pickle.dump(self.train_docsIds_list, handle, protocol=pickle.HIGHEST_PROTOCOL)

        file = open('../../dev_word_ner_startidx_dict.pickle', 'rb')
        self.dev_sentences = pickle.load(file)
        self.dev_sentences = OrderedDict(self.dev_sentences)
        file.close()

        self.dev_docsIds_list = []

        for k, v in self.dev_sentences.items():
            self.dev_docsIds_list.append(k)

        with open('../devDocIds.pickle', 'wb') as handle:
            pickle.dump(self.dev_docsIds_list, handle, protocol=pickle.HIGHEST_PROTOCOL)

        file = open('../../test_word_ner_startidx_dict.pickle', 'rb')
        self.test_sentences = pickle.load(file)
        self.test_sentences = OrderedDict(self.test_sentences)
        file.close()

        self.test_docsIds_list = []

        for k, v in self.test_sentences.items():
            self.test_docsIds_list.append(k)

        with open('../testDocIds.pickle', 'wb') as handle:
            pickle.dump(self.test_docsIds_list, handle, protocol=pickle.HIGHEST_PROTOCOL)


        file = open('../words.pickle', 'rb')
        vocab_words = pickle.load(file)
        file.close()

        file = open('../chars.pickle', 'rb')
        vocab_chars = pickle.load(file)
        file.close()

        file = open('../tags.pickle', 'rb')
        vocab_tags = pickle.load(file)
        file.close()

        self.vocab_words = load_vocab(list(vocab_words))
        self.vocab_chars = load_vocab(list(vocab_chars))
        self.vocab_tags = load_vocab(list(vocab_tags))

        self.nwords = len(vocab_words)
        self.nchars = len(vocab_chars)
        self.ntags = len(vocab_tags)

        all_train_sentences = []
        all_train_sentences += [i for x in self.train_sentences.values() for i in x]
        self.all_train_sentences_preprocessed = []

        # print(all_train_sentences[0])
        for sent in all_train_sentences:
            current_sentence = []
            for i, (word, label, startIndex, postag) in enumerate(sent):
                w = get_processing_word(word, self.vocab_words, self.vocab_chars, lowercase=True, chars=True)
                t = get_processing_word(label, self.vocab_tags, lowercase=False, allow_unk=False)
                current_sentence.append((w, t, startIndex, postag))
            self.all_train_sentences_preprocessed.append(current_sentence)
        # print(self.all_train_sentences_preprocessed[0])

        all_dev_sentences = []
        all_dev_sentences += [i for x in self.dev_sentences.values() for i in x]
        self.all_dev_sentences_preprocessed = []

        for sent in all_dev_sentences:
            current_sentence = []
            for i, (word, label, startIndex, postag) in enumerate(sent):
                w = get_processing_word(word, self.vocab_words, self.vocab_chars, lowercase=True, chars=True)
                t = get_processing_word(label, self.vocab_tags, lowercase=False, allow_unk=False)
                current_sentence.append((w, t, startIndex, postag))
            self.all_dev_sentences_preprocessed.append(current_sentence)

        all_test_sentences = []
        all_test_sentences += [i for x in self.test_sentences.values() for i in x]
        self.all_test_sentences_preprocessed = []

        for sent in all_test_sentences:
            current_sentence = []
            for i, (word, label, startIndex, postag) in enumerate(sent):
                w = get_processing_word(word, self.vocab_words, self.vocab_chars, lowercase=True, chars=True)
                t = get_processing_word(label, self.vocab_tags, lowercase=False, allow_unk=False)
                current_sentence.append((w, t, startIndex, postag))
            self.all_test_sentences_preprocessed.append(current_sentence)

        self.embeddings = get_fasttext_vectors('../vocab_embeddings.npz')

    cwd = os.getcwd()
    dir_output = os.path.join(cwd, "../Model/")
    dir_model  = os.path.join(dir_output, "weights/")
    dim_word = 300
    dim_char = 100
    fig_confusionplot = os.path.join(cwd, "../plots/confusionplot")
    train_embeddings = False
    nepochs          = 1
    dropout          = 0.5
    batch_size       = 1
    lr_method        = "adam"
    lr               = 0.002
    lr_decay         = 0.5
    hidden_size_char = 100
    hidden_size_lstm = 200
    use_crf = True

if __name__ == "__main__":
    main()

