import numpy as np
import os
import tensorflow as tf

import sys
sys.path.append('../')

from sklearn.metrics import precision_recall_fscore_support

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker



class NeuralModel(object):
    def __init__(self, config):
        # super(NeuralModel, self).__init__(config)
        self.config = config
        self.sess = None
        self.saver = None
        self.idx_to_tag = {idx: tag for tag, idx in
                           list(self.config.vocab_tags.items())}

        self.all_losses = []


    def add_train_optimizer(self, lr_method, lr, loss):
        _lr_m = lr_method.lower()

        with tf.variable_scope("train_step"):
            optimizer = tf.train.AdamOptimizer(lr)
            self.train_op = optimizer.minimize(loss)


    def initialize_session(self):
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()


    def restore_session(self, dir_model):
        self.saver.restore(self.sess, dir_model)


    def save_session(self):
        if not os.path.exists(self.config.dir_model):
            os.makedirs(self.config.dir_model)
        self.saver.save(self.sess, self.config.dir_model)


    def close_session(self):
        self.sess.close()


    def train(self):
        best = 0

        for epoch in range(self.config.nepochs):
            print("Epoch {:} out of {:}".format(epoch + 1, self.config.nepochs))
            score = self.run_epoch(epoch)
            print("score for {} epoch: {}".format(epoch, score))
            self.config.lr *= self.config.lr_decay

            if score >= best:
                self.save_session()
                best = score
                print("new best score: ", best)
            else:
                print("early stopping. Current score: {}, best score: {}".format(score, best))
                break

    def evaluate(self, test, filename):
        metrics = self.run_evaluate_with_pred_output(test, filename)
        return metrics


    def add_placeholders(self):
        self.word_ids = tf.placeholder(tf.int32, shape=[None, None], name="word_ids")

        self.sequence_lengths = tf.placeholder(tf.int32, shape=[None], name="sequence_lengths")

        self.char_ids = tf.placeholder(tf.int32, shape=[None, None, None], name="char_ids")

        self.word_lengths = tf.placeholder(tf.int32, shape=[None, None], name="word_lengths")

        self.labels = tf.placeholder(tf.int32, shape=[None, None], name="labels")

        self.dropout = tf.placeholder(dtype=tf.float32, shape=[], name="dropout")

        self.lr = tf.placeholder(dtype=tf.float32, shape=[], name="lr")


    def get_feed_dict(self, words, labels=None, lr=None, dropout=None):
        char_ids, word_ids = list(zip(*words))
        word_ids, sequence_lengths = pad_sequences(word_ids, 0)
        char_ids, word_lengths = pad_sequences(char_ids, pad_tok=0, nlevels=2)

        feed = {}
        feed[self.word_ids] = word_ids
        feed[self.sequence_lengths] = sequence_lengths
        feed[self.char_ids] = char_ids
        feed[self.word_lengths] = word_lengths

        if labels is not None:
            labels, _ = pad_sequences(labels, 0)
            feed[self.labels] = labels

        if lr is not None:
            feed[self.lr] = lr

        if dropout is not None:
            feed[self.dropout] = dropout

        return feed, sequence_lengths


    def add_word_embeddings_op(self):
        with tf.variable_scope("words"):
            _word_embeddings = tf.Variable(
                        self.config.embeddings,
                        name="_word_embeddings",
                        dtype=tf.float32,
                        trainable=self.config.train_embeddings)

            word_embeddings = tf.nn.embedding_lookup(_word_embeddings, self.word_ids, name="word_embeddings")

        with tf.variable_scope("chars"):
            _char_embeddings = tf.get_variable(
                    name="_char_embeddings",
                    dtype=tf.float32,
                    shape=[self.config.nchars, self.config.dim_char])
            char_embeddings = tf.nn.embedding_lookup(_char_embeddings,
                    self.char_ids, name="char_embeddings")

            s = tf.shape(char_embeddings)
            char_embeddings = tf.reshape(char_embeddings, shape=[s[0]*s[1], s[-2], self.config.dim_char])
            word_lengths = tf.reshape(self.word_lengths, shape=[s[0]*s[1]])

            cell_fw = tf.contrib.rnn.LSTMCell(self.config.hidden_size_char, state_is_tuple=True)
            cell_bw = tf.contrib.rnn.LSTMCell(self.config.hidden_size_char, state_is_tuple=True)
            _output = tf.nn.bidirectional_dynamic_rnn(
                    cell_fw, cell_bw, char_embeddings,
                    sequence_length=word_lengths, dtype=tf.float32)

            _, ((_, output_fw), (_, output_bw)) = _output
            output = tf.concat([output_fw, output_bw], axis=-1)

            output = tf.reshape(output,
                    shape=[s[0], s[1], 2*self.config.hidden_size_char])
            word_embeddings = tf.concat([word_embeddings, output], axis=-1)

        self.word_embeddings =  tf.nn.dropout(word_embeddings, self.dropout)


    def add_logits_op(self):
        with tf.variable_scope("bi-lstm"):
            cell_fw = tf.contrib.rnn.LSTMCell(self.config.hidden_size_lstm)
            cell_bw = tf.contrib.rnn.LSTMCell(self.config.hidden_size_lstm)
            (output_fw, output_bw), _ = tf.nn.bidirectional_dynamic_rnn(
                    cell_fw, cell_bw, self.word_embeddings,
                    sequence_length=self.sequence_lengths, dtype=tf.float32)
            output = tf.concat([output_fw, output_bw], axis=-1)
            output = tf.nn.dropout(output, self.dropout)

        with tf.variable_scope("proj"):
            W = tf.get_variable("W", dtype=tf.float32,
                    shape=[2*self.config.hidden_size_lstm, self.config.ntags])

            b = tf.get_variable("b", shape=[self.config.ntags],
                    dtype=tf.float32, initializer=tf.zeros_initializer())

            nsteps = tf.shape(output)[1]
            output = tf.reshape(output, [-1, 2*self.config.hidden_size_lstm])
            pred = tf.matmul(output, W) + b
            self.logits = tf.reshape(pred, [-1, nsteps, self.config.ntags])


    def add_loss_op(self):
        log_likelihood, trans_params = tf.contrib.crf.crf_log_likelihood(
                self.logits, self.labels, self.sequence_lengths)
        self.trans_params = trans_params
        self.loss = tf.reduce_mean(-log_likelihood)


    def build(self):
        self.add_placeholders()
        self.add_word_embeddings_op()
        self.add_logits_op()
        self.add_loss_op()

        self.add_train_optimizer(self.config.lr_method, self.lr, self.loss)
        self.initialize_session()


    def predict_batch(self, words):
        fd, sequence_lengths = self.get_feed_dict(words, dropout=1.0)

        if self.config.use_crf:
            viterbi_sequences = []
            logits, trans_params = self.sess.run(
                    [self.logits, self.trans_params], feed_dict=fd)

            for logit, sequence_length in zip(logits, sequence_lengths):
                logit = logit[:sequence_length] # keep only the valid steps
                viterbi_seq, viterbi_score = tf.contrib.crf.viterbi_decode(
                        logit, trans_params)
                viterbi_sequences += [viterbi_seq]

            return viterbi_sequences, sequence_lengths

        else:
            labels_pred = self.sess.run(self.labels_pred, feed_dict=fd)

            return labels_pred, sequence_lengths


    def run_epoch(self, epoch):
        for i, sent in enumerate(self.config.all_train_sentences_preprocessed):
            words = []
            labels = []
            for j, (words_, labels_, startIdx, postag) in enumerate(sent):
                words.append(words_)
                labels.append(labels_)

            if len(words) == 0:
                continue

            words = [zip(*words)]
            labels = [labels]
            fd, _ = self.get_feed_dict(words, labels, self.config.lr, self.config.dropout)

            _, train_loss = self.sess.run(
                    [self.train_op, self.loss], feed_dict=fd)

        metrics = self.run_evaluate(self.config.all_dev_sentences_preprocessed, epoch)
        print(metrics)

        return metrics[2]


    def run_evaluate(self, test, epoch):
        confusion = np.zeros((self.config.ntags, self.config.ntags))
        all_tags = []
        all_tags_indices = []
        for idx, tag in self.idx_to_tag.items():
            all_tags.append(tag)
            all_tags_indices.append(idx)

        l_true = []
        l_pred = []

        accs = []
        
        for i, sent in enumerate(test):
            words = []
            labels = []
            for j, (words_, labels_, startIdx, postag) in enumerate(sent):
                words.append(words_)
                labels.append(labels_)

            if len(words) == 0:
                continue

            words = [zip(*words)]
            labels = [labels]

            labels_pred, sequence_lengths = self.predict_batch(words)

            for lab, lab_pred, length in zip(labels, labels_pred,
                                             sequence_lengths):
                lab = lab[:length]
                lab_pred = lab_pred[:length]
                accs += [a==b for (a, b) in zip(lab, lab_pred)]

                for (a, b) in zip(lab, lab_pred):
                    confusion[all_tags_indices.index(a)][all_tags_indices.index(b)] += 1

                l_true += lab
                l_pred += lab_pred


        # Normalize by dividing every row by its sum
        for i in range(self.config.ntags):
            confusion[i] = confusion[i] / confusion[i].sum()

        self.show_confusion_plot(confusion, all_tags, epoch)

        tags = [idx for idx, tag in self.idx_to_tag.items()]
        return precision_recall_fscore_support(y_true = l_true, y_pred = l_pred, labels = tags, average='weighted')

    def run_evaluate_with_pred_output(self, test, filename):
        l_true = []
        l_pred = []

        accs = []

        pred_for_test_words = []

        for i, sent in enumerate(test):
            words = []
            labels = []
            for j, (words_, labels_, startIdx, postag) in enumerate(sent):
                words.append(words_)
                labels.append(labels_)

            if len(words) == 0:
                continue

            words = [zip(*words)]
            labels = [labels]

            labels_pred, sequence_lengths = self.predict_batch(words)

            for lab, lab_pred, length in zip(labels, labels_pred, sequence_lengths):
                for (a, b) in zip(lab, lab_pred):
                    pred_for_test_words.append(b)

                lab = lab[:length]
                lab_pred = lab_pred[:length]
                accs += [a == b for (a, b) in zip(lab, lab_pred)]

                l_true += lab
                l_pred += lab_pred


        print("Writing final predictions file...")
        with open('../' + filename, "w") as f:
            for i, pred_label in enumerate(pred_for_test_words):
                if i != len(pred_for_test_words) - 1:
                    f.write("{}\n".format(self.idx_to_tag[pred_label]))
                else:
                    f.write("{}".format(self.idx_to_tag[pred_label]))

        tags = [idx for idx, tag in self.idx_to_tag.items()]
        return precision_recall_fscore_support(y_true=l_true, y_pred=l_pred, labels=tags, average='weighted')

    def show_confusion_plot(self, confusion, all_tags, epoch):
        # Set up plot
        fig = plt.figure()
        ax = fig.add_subplot(111)
        cax = ax.matshow(confusion)
        fig.colorbar(cax)


        # Set up axes
        ax.set_xticklabels([''] + all_tags, rotation=90)
        ax.set_yticklabels([''] + all_tags)

        # Force label at every tick
        ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
        ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

        # plt.show()
        f = self.config.fig_confusionplot+'_' + str(epoch) +'.png'
        print("saving confusion plot: ", f)
        plt.savefig(f)


#padding
def _pad_sequences(sequences, pad_tok, max_length):
    sequence_padded, sequence_length = [], []

    for seq in sequences:
        seq = list(seq)
        seq_ = seq[:max_length] + [pad_tok]*max(max_length - len(seq), 0)
        sequence_padded +=  [seq_]
        sequence_length += [min(len(seq), max_length)]

    return sequence_padded, sequence_length

#padding
def pad_sequences(sequences, pad_tok, nlevels=1):
    if nlevels == 1:
        max_length = max(map(lambda x : len(x), sequences))
        sequence_padded, sequence_length = _pad_sequences(sequences,
                                            pad_tok, max_length)

    elif nlevels == 2:
        max_length_word = max([max(map(lambda x: len(x), seq))
                               for seq in sequences])
        sequence_padded, sequence_length = [], []
        for seq in sequences:
            sp, sl = _pad_sequences(seq, pad_tok, max_length_word)
            sequence_padded += [sp]
            sequence_length += [sl]

        max_length_sentence = max(map(lambda x : len(x), sequences))
        sequence_padded, _ = _pad_sequences(sequence_padded,
                [pad_tok]*max_length_word, max_length_sentence)
        sequence_length, _ = _pad_sequences(sequence_length, 0,
                max_length_sentence)

    return sequence_padded, sequence_length

