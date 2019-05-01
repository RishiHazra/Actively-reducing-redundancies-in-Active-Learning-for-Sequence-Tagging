import numpy as np
import os
import os.path
import sys
import operator
from collections import Counter

import tensorflow as tf

from .data_utils import minibatches, pad_sequences, get_chunks, UNK
from .general_utils import Progbar
from .base_model import BaseModel
from .modules import CNN_BILSTM_CRF, LSTM_LSTM_CRF, BILSTM_BILSTM_CRF_attention
from .config import Config

sys.path.append("..")
from active_learning import Active_Learning

#---------------------------------------------------------------------
# adapted from : https://github.com/guillaumegenthial/sequence_tagging
#---------------------------------------------------------------------

#-----------------------------------------------------------
# instantiate the classes
#-----------------------------------------------------------
config = Config()
AL = Active_Learning(config.active_algo)

if config.model == "CNN BILSTM CRF":
    model = CNN_BILSTM_CRF
elif config.model == "LSTM LSTM CRF":
    model = LSTM_LSTM_CRF
elif config.model == "BILSTM BILSTM CRF":
    model = BILSTM_BILSTM_CRF_attention


class NERModel(BaseModel):
    """Specialized class of Model for NER"""

    def __init__(self, config):
        super(NERModel, self).__init__(config)
        self.idx_to_tag = {idx: tag for tag, idx in
                           config.vocab_tags.items()}
        self.tag_to_idx = {tag: idx for tag, idx in
                           config.vocab_tags.items()}

    def add_placeholders(self):
        """Define placeholders = entries to computational graph"""
        # shape = (batch size, max length of sentence in batch)
        self.word_ids = tf.placeholder(tf.int32, shape=[None, None],
                                       name="word_ids")

        self.sequence_lengths = tf.placeholder(tf.int32, shape=[None],
                                               name="sequence_lengths")

        # shape = (batch size, max length of sentence, max length of word)
        self.char_ids = tf.placeholder(tf.int32, shape=[None, None, None],
                                       name="char_ids")

        # shape = (batch_size, max_length of sentence)
        self.word_lengths = tf.placeholder(tf.int32, shape=[None, None],
                                           name="word_lengths")

        # shape = (batch size, max length of sentence in batch)
        self.labels = tf.placeholder(tf.int32, shape=[None, None],
                                     name="labels")

        # hyper parameters
        self.dropout = tf.placeholder(dtype=tf.float32, shape=[],
                                      name="dropout")
        self.lr = tf.placeholder(dtype=tf.float32, shape=[],
                                 name="lr")
        self.flair_input = tf.placeholder(tf.float32, shape=[None, None, 2348],
                                          name='flair_input')


    def get_feed_dict(self, words, labels=None, lr=None, dropout=None):
        """Given some data, pad it and build a feed dictionary
        Args:
            words: list of sentences. A sentence is a list of ids of a list of
                words. A word is a list of ids
            labels: list of ids
            lr: (float) learning rate
            dropout: (float) keep prob
        Returns:
            dict {placeholder: value}
        """
        # perform padding of the given data
        if config.use_chars:
            char_ids, word_ids = zip(*words)
            word_ids, sequence_lengths = pad_sequences(word_ids, 0)
            char_ids, word_lengths = pad_sequences(char_ids, pad_tok=0,
                                                   nlevels=2)
        else:
            word_ids, sequence_lengths = pad_sequences(words, 0)

        # build feed dictionary
        feed = {
            self.word_ids: word_ids,
            self.sequence_lengths: sequence_lengths
        }

        if config.use_chars:
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

    def add_pred_op(self):
        """Defines self.labels_pred
        This op is defined only in the case where we don't use a CRF since in
        that case we can make the prediction "in the graph" (thanks to tf
        functions in other words). With the CRF, as the inference is coded
        in python and not in pure tensorflow, we have to make the prediciton
        outside the graph.
        """
        if not config.use_crf:
            self.labels_pred = tf.cast(tf.argmax(self.logits, axis=-1),
                                       tf.int32)


    def add_loss_op(self):
        """Defines the loss"""
        if config.use_crf:
            log_likelihood, trans_params = tf.contrib.crf.crf_log_likelihood(
                self.logits, self.labels, self.sequence_lengths)
            self.trans_params = trans_params  # need to evaluate it for decoding
            self.loss = tf.reduce_mean(-log_likelihood)
        else:
            losses = tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=self.logits, labels=self.labels)
            mask = tf.sequence_mask(self.sequence_lengths)
            losses = tf.boolean_mask(losses, mask)
            self.loss = tf.reduce_mean(losses)

        # for tensorboard
        tf.summary.scalar("loss", self.loss)


    def build(self):
        # NER specific functions
        self.add_placeholders()
        obj = model(config, self.word_ids, self.sequence_lengths, self.char_ids, self.word_lengths,
              self.labels, self.dropout, self.lr)
        self.add_word_embeddings_op = obj.add_word_embeddings_op()
        self.add_logits_op = obj.add_logits_op()
        self.shape = obj.shape
        self.encoded = obj.encoded
        self.model_unaware_embedding = obj.model_unaware_embedding
        self.logits = obj.logits
        self.add_pred_op()
        self.add_loss_op()

        # Generic functions that add training op and initialize session
        self.add_train_op(config.lr_method, self.lr, self.loss,
                          config.clip)
        self.initialize_session()  # now self.sess is defined and vars are init


    def encode_sents(self, sick):
        """
        Args:
            preprocessed sentences, word BiLSTM weights
        return:
            encoded sentences from the SICK 2014 dataset
            which is then used to train the the sentence
            similarity model

        The pair of sentences are stored in a matrix and
        once all the sentences are encoded, we start training
        the other model
        """        
        if not os.path.exists(config.dir_model_encoded_SICK ):
            os.makedirs(config.dir_model_encoded_SICK)
        path = config.dir_model_encoded_SICK
        i = 0    
        for words, labels in minibatches(sick, 2):
            fd, _ = self.get_feed_dict(words, dropout=1.0)
            if config.model_aware:
                #print('\nModel Aware\n')
                enc = self.sess.run([self.encoded], feed_dict=fd)
                enc = np.reshape(enc, (np.array(enc).shape[1], np.array(enc).shape[2]))
                sent1 = np.array_split(enc, 2)[0]
                sent2 = np.array_split(enc, 2)[1]
            else:
                #print('\nModel Unaware\n')
                enc = self.sess.run([self.model_unaware_embedding], feed_dict=fd)
                sent1 = enc[0][0]
                sent2 = enc[0][1]
            i += 1
            np.savez(path + str(i),
                    sent_1 = sent1,
                    sent_2 = sent2,
                    lab = labels[0])
        print('Saved SICK encodings to {}'.format(config.dir_model_encoded_SICK))


    def BALD_MC(self, words):
        viterbi_sequences, scores = {}, {}
        count, samples = [], []

        for _ in range(config.sample_times):
            fd, sequence_lengths = self.get_feed_dict(words, dropout=0.5)
            logits, trans_params = self.sess.run(
                [self.logits, self.trans_params], feed_dict=fd)
            # ----------------------------------------------------------------
            # iterate over the sentences because no batching in viterbi_decode
            # ----------------------------------------------------------------
            for sent_id, (logit, sequence_length) in enumerate(zip(logits, sequence_lengths)):
                logit = logit[:sequence_length]  # keep only the valid steps
                viterbi_seq, viterbi_score = tf.contrib.crf.viterbi_decode(
                    logit, trans_params)

                #viterbi_score = AL.active_strategy(logit, trans_params, self.tag_to_idx)

                if sent_id in viterbi_sequences.keys():
                    viterbi_sequences[sent_id] += [viterbi_seq]
                    scores[sent_id] += [viterbi_score]
                else:
                    viterbi_sequences[sent_id] = [viterbi_seq]
                    scores[sent_id] = [viterbi_score]
        # -----------------------------------------------------------------------------------
        # arg_max ( 1 - ( count( mode( y(1),...,y(T) ) ) ) / T )
        # -----------------------------------------------------------------------------------
        count += [Counter(tuple(x) for x in value) for value in viterbi_sequences.values()]
        mc_samples = [max(dict.items(), key=operator.itemgetter(1))[0] for dict in count]
        scores = [1 - (max(dict.items(), key=operator.itemgetter(1))[1]) / config.sample_times for dict in count]

        return mc_samples, sequence_lengths, list(map(lambda x : -x, scores))


    def predict_batch(self, words):
        """
        Args:
            words: list of sentences
        Returns:
            labels_pred: list of labels for each sentence
            sequence_length
        """
        fd, sequence_lengths = self.get_feed_dict(words, dropout=1.0)

        if config.use_crf:
            # ----------------------------------------------------------------
            # get tag scores and transition params of CRF
            # ----------------------------------------------------------------
            viterbi_sequences = []
            scores = []
            logits, trans_params = self.sess.run(
                [self.logits, self.trans_params], feed_dict=fd)

            # ----------------------------------------------------------------
            # iterate over the sentences because no batching in viterbi_decode
            # ----------------------------------------------------------------
            for logit, sequence_length in zip(logits, sequence_lengths):
                logit = logit[:sequence_length]  # keep only the valid steps
                viterbi_seq, viterbi_score = tf.contrib.crf.viterbi_decode(
                    logit, trans_params)

                #---------------------------------------------------------------
                # first layer of active learning
                #---------------------------------------------------------------

                viterbi_sequences += [viterbi_seq]
                if config.active_algo == "nus":
                    viterbi_score = float(viterbi_score / sequence_length)
                else:
                    viterbi_score = AL.active_strategy(logit, trans_params, self.tag_to_idx)
                scores.append(viterbi_score)

            return viterbi_sequences, sequence_lengths, scores

        else:
            labels_pred = self.sess.run(self.labels_pred, feed_dict=fd)

            return labels_pred, sequence_lengths, None



    def run_epoch(self, train, dev, epoch):
        """Performs one complete pass over the train set and evaluate on dev
        Args:
            train: dataset that yields tuple of sentences, tags
            dev: dataset
            epoch: (int) index of the current epoch
        Returns:
            f1: (python float), score to select model on, higher is better
        """
        #-----------------------------------------------------
        # progbar stuff for logging
        #-----------------------------------------------------
        batch_size = config.batch_size
        nbatches = (len(train) + batch_size - 1) // batch_size
        prog = Progbar(target=nbatches)

        # -----------------------------------------------------
        # iterate over dataset
        # -----------------------------------------------------
        for i, (words, labels) in enumerate(minibatches(train, batch_size)):
            fd, _ = self.get_feed_dict(words, labels, config.lr,
                                       config.dropout)
            #shape = self.sess.run([self.shape], feed_dict=fd)
            _, train_loss, summary = self.sess.run(
                [self.train_op, self.loss, self.merged], feed_dict=fd)
            prog.update(i + 1, [("train loss", train_loss)])

            # --------------------------------------------------
            # tensorboard
            # --------------------------------------------------
            if i % 10 == 0:
                self.file_writer.add_summary(summary, epoch * nbatches + i)

        metrics = self.run_evaluate(dev, dev)
        msg = " - ".join(["{} {:04.2f}".format(k, v)
                          for k, v in metrics.items()])
        self.logger.info(msg)

        return metrics["f1"]



    def align_data(self, data):
        """Given dict with lists, creates aligned strings
        Adapted from Assignment 3 of CS224N
        Args:
            data: (dict) data["x"] = ["I", "love", "you"]
                  (dict) data["y"] = ["O", "O", "O"]
        Returns:
            data_aligned: (dict) data_align["x"] = "I love you"
                               data_align["y"] = "O O    O  "
        """
        spacings = [max([len(seq[i]) for seq in data.values()])
                    for i in range(len(data[list(data.keys())[0]]))]
        data_aligned = dict()

        # for each entry, create aligned string
        for key, seq in data.items():
            str_aligned = ""
            for token, spacing in zip(seq, spacings):
                str_aligned += token + " " * (spacing - len(token) + 1)

            data_aligned[key] = str_aligned

        return data_aligned



    def determine_threshold(self, test, threshold):
        """
         function to determine a proper threshold
         here, threshold is empirically determined
        """
        accs = []
        correct_preds, total_correct, total_preds = 0., 0., 0.

        for words, labels in minibatches(test, config.batch_size):
            labels_pred, sequence_lengths, prob = self.predict_batch(words)

            for lab, lab_pred, length, score in zip(labels, labels_pred,
                                                    sequence_lengths, prob):
                if score <= threshold:
                    lab = lab[:length]
                    lab_pred = lab_pred[:length]
                    accs += [a == b for (a, b) in zip(lab, lab_pred)]
                    lab_chunks = set(get_chunks(lab, config.vocab_tags))
                    lab_pred_chunks = set(get_chunks(lab_pred,
                                                     config.vocab_tags))
                    correct_preds += len(lab_chunks & lab_pred_chunks)
                    total_preds += len(lab_pred_chunks)
                    total_correct += len(lab_chunks)

        p = correct_preds / total_preds if correct_preds > 0 else 0
        r = correct_preds / total_correct if correct_preds > 0 else 0
        f1 = 2 * p * r / (p + r) if correct_preds > 0 else 0
        acc = np.mean(accs)
        return {"acc": 100 * acc, "f1": 100 * f1, "correct_preds": correct_preds, "total_correct": total_correct,
                "p": p, "r": r, "total_preds": total_preds}


    def run_evaluate(self, test, dev, mode="train"):
        """Evaluates performance on test set
        Args:
            dev : AL to be performed on this dataset
            test: dataset that yields tuple of (sentences, tags)
        Returns:
            metrics: (dict) metrics["acc"] = 98.4, ...
        """
        #---------------------------------------------------------------------------
        # retrain_file : split to be retrained on (changes in every round of AL)
        # train_file   : train split for the current AL round
        # newSamples   : contains all low confidence samples from each round of AL
        # AL_file      : file to be sampled from for current round of AL (same as dev file)
        #---------------------------------------------------------------------------

        if config.mode == 'feedback':
            num_fedback = 0.0
            train_file = config.train_split[config.split]
            newSamples = config.filename_newSamples
            retrain_file = config.filename_retrain
            #AL_file = config.train_split[config.sample_split]
            words, tags, prob, enc, seq_len = self.feedback_helper(dev)

            if os.path.isfile('num_fedback.npy'):
                num_fedback = np.load('num_fedback.npy')
            num_fedback += AL.feedback(newSamples, config.dummy_train,
                                       words, tags, prob, enc, seq_len)
            print('\nnum_fedback : {}'.format(num_fedback))
            np.save('num_fedback.npy', num_fedback)
            AL.random_sampling(train_file, newSamples, retrain_file, num_fedback)
            print('Feedback Done...')

        else:
            e, d, f = {}, {}, {}
            accs = []

            correct_preds, total_correct, total_preds= 0., 0., 0.

            for words, labels in minibatches(test, config.batch_size):
                labels_pred, sequence_lengths, prob = self.predict_batch(words)

                for lab, lab_pred, length in zip(labels, labels_pred,
                                                 sequence_lengths):
                    lab = lab[:length]
                    lab_pred = lab_pred[:length]
                    accs += [a == b for (a, b) in zip(lab, lab_pred)]
                    lab_chunks = set(get_chunks(lab, config.vocab_tags))
                    lab_pred_chunks = set(get_chunks(lab_pred,
                                                     config.vocab_tags))

                    c = lab_chunks & lab_pred_chunks
                    correct_preds += len(lab_chunks & lab_pred_chunks)
                    total_preds += len(lab_pred_chunks)
                    total_correct += len(lab_chunks)

                    # ------------------------------------------------------
                    # d : correct predictions
                    # e : total correct
                    # f : total predictions
                    # ------------------------------------------------------

                    def calculate(chunks, preds):
                        for i in range(len(chunks)):
                            l = chunks.pop()[0]
                            if l not in preds.keys():
                                preds[l] = 1
                            else:
                                preds[l] += 1
                        return preds

                    d = calculate(c, d)
                    e = calculate(lab_chunks, e)
                    f = calculate(lab_pred_chunks, f)

            p = correct_preds / total_preds if correct_preds > 0 else 0
            r = correct_preds / total_correct if correct_preds > 0 else 0
            f1 = 2 * p * r / (p + r) if correct_preds > 0 else 0
            acc = np.mean(accs)
            #print('total_correct: ', e, ' correct_preds: ', d, 'total_preds:', f)
            return {"acc": 100 * acc, "f1": 100 * f1, "correct_preds": correct_preds, "total_correct": total_correct,
                    "p": p, "r": r, "total_preds": total_preds}


    def feedback_helper(self, dev):
        """
            helper function for collecting the most confused examples
            and feed it to the subsequent layers of the Active Learning

            dev : file on which AL is performed
        """
        print('\n Feedback mode... \n')
        enc, prob_confused, seq_len_confused, words_confused, tags_confused = \
            [], [], [], [], []

        for words, labels in minibatches(dev, 10):
            if config.active_algo == 'bald':
                labels_pred, sequence_lengths, prob = self.BALD_MC(words)
            else:
                labels_pred, sequence_lengths, prob = self.predict_batch(words)

            for X in range(len(prob)):
                if prob[X] <= config.threshold:
                    fd, _ = self.get_feed_dict([words[X]], dropout=1.0)

                    shape = self.sess.run([self.shape], feed_dict=fd)[0][-2]
                    if shape < 3:
                       continue

                    prob_confused += [prob[X]]
                    seq_len_confused += [sequence_lengths[X]]

                    word_list = []
                    for i in range(sequence_lengths[X]):
                        word_list += [''.join([config.vocab_chars_inv[char] for char in words[X][0][i]])]
                    words_confused += [word_list]
                    tags_confused += [[config.vocab_tags_inv[label] for label in labels[X]]]

                    if config.model_aware:
                        # print('\nModel Aware\n')
                        Enc = self.sess.run([self.encoded], feed_dict=fd)
                        enc += [np.reshape(Enc, (np.array(Enc).shape[1], np.array(Enc).shape[2])).tolist()]
                        #sent1 = np.array_split(enc, 2)[0]
                        #sent2 = np.array_split(enc, 2)[1]
                    else:
                        # print('\nModel Unaware\n')
                        enc += [self.sess.run([self.model_unaware_embedding], feed_dict=fd)]

        print('\n {} low confidence sentences extracted ... \n '.format(len(seq_len_confused)))
        return words_confused, tags_confused, prob_confused, enc, seq_len_confused


    def predict(self, words_raw):
        """Returns list of tags
        Args:
            words_raw: list of words (string), just one sentence (no batch)
        Returns:
            preds: list of tags (string), one for each word in the sentence
        """
        words = [config.processing_word(w) for w in words_raw]
        if type(words[0]) == tuple:
            words = zip(*words)
        pred_ids, _, scores = self.predict_batch([words])
        preds = [self.idx_to_tag[idx] for idx in list(pred_ids[0])]
        to_print = self.align_data({"input": words_raw, "output": preds})

        # to print
        '''
        for key, seq in to_print.items():
            print(seq)
        print(scores,'\n')
        '''
        # return preds