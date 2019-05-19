from model.config import Config
import numpy as np
import math
import random
import collections

import sys
from sklearn.metrics.pairwise import cosine_similarity as cos_sim
from sklearn.cluster import SpectralClustering
import tensorflow as tf

config = Config()

#--------------------------------------------------------------------------
# follow instructions from : https://github.com/ryankiros/skip-thoughts
#--------------------------------------------------------------------------
#sys.path.append('../Skip Thoughts')
#import skipthoughts


class Siamese_Model():
    def __init__(self, session):
        self.sess = session

        self.saver1 = tf.train.import_meta_graph(config.dir_model_similarity
                                           + '/siamese_model.meta')
        self.siamese_graph = tf.get_default_graph()
        self.saver1.restore(self.sess, config.dir_model_similarity
                                           + '/siamese_model')

        self.output1 = self.siamese_graph.get_tensor_by_name('output/out1:0')
        self.output2 = self.siamese_graph.get_tensor_by_name('output/out2:0')

    def run(self, x1_batch, x2_batch, seq_len1, seq_len2, max_len, batch_size, len2):
        feed_dict = {
            'input_x1:0': x1_batch,
            'input_x2:0': x2_batch,
            'seq_len1:0': seq_len1,
            'seq_len2:0': seq_len2,
            'max_seq_len:0': max_len,
            'dropout_keep_prob:0': 1,
            'batch_size_dynamic:0': batch_size,
        }
        out1, out2 = self.sess.run([self.output1, self.output2],
                             feed_dict=feed_dict)
        return out1, out2[:len2]

#----------------------------------------------------------------------
# Active Learning : Layer 1 (retains the most confused examples)
#                 : Layer 2 (retains the most representative examples)
#                 : Layer 3 (outlier detection)
#----------------------------------------------------------------------

class Active_Learning():

    def __init__(self, strategy):
        self.active_algo = strategy

    def active_strategy(self, score, transition_params, tag_to_idx):
        """
        Args: output of CRF
        score: A [seq_len, num_tags] matrix of unary potentials.
        transition_params: A [num_tags, num_tags] matrix of binary potentials.
        """

        if self.active_algo == "cluster":
            # print('score: ',score)
            return score
        trellis = np.zeros_like(score)
        backpointers = np.zeros_like(score, dtype=np.int32)
        trellis[0] = score[0]

        for t in range(1, score.shape[0]):
            v = np.expand_dims(trellis[t - 1], 1) + transition_params
            trellis[t] = score[t] + np.max(v, 0)
            backpointers[t] = np.argmax(v, 0)

        viterbi = [np.argmax(trellis[-1])]
        for bp in reversed(backpointers[1:]):
            viterbi.append(bp[viterbi[-1]])
        viterbi.reverse()

        score_final = np.max(trellis[-1])  # Score of sequences (higher = better)

        if (self.active_algo == 'margin'):
            top_scores = trellis[-1][np.argsort(trellis[-1])[-2:]]
            margin = abs(top_scores[0] - top_scores[1])
            score_final = margin
            # print('score_final: ', score_final)

        elif (self.active_algo == 'ne'):
            ne = ['NE.AMBIG', 'NE.DE', 'NE.LANG3', 'NE.MIXED', 'NE.OTHER', 'NE.TR']
            ne_idx = []
            for i in tag_to_idx:
                if i in ne:
                    ne_idx.append(tag_to_idx[i])
            # Get the highest score of NE
            max_ne = []
            # for i in ne_idx:
            #     max_ne.append(np.max(score[:,i]))
            score_final = 0
            for i in viterbi:
                if i in ne_idx:
                    score_final += 1  # give higher score to sequences that have more named entities
            # score_final = np.max(max_ne)

        elif (self.active_algo == 'nemg'):  # ne margin
            ne_idx = tag_to_idx['NE.DE']
            ne_de = tag_to_idx['DE']
            margin = np.add(score[:, ne_idx], score[:, ne_de])
            margin2 = abs(np.multiply(score[:, ne_idx], score[:, ne_de]))
            margin = np.divide(margin, margin2)
            sum_margin = np.sum(margin)
            score_final = sum_margin

        if (self.active_algo == 'entropy'):
            # Find the highest prob for each token
            ntoken = len(score)
            ntags = len(score[0])
            l = []  # max prob of each token
            for i in range(0, ntoken):
                l.append(np.max(score[i]))
            ne_idx = tag_to_idx
            # Compute entropy
            score_final = 0.0
            for i in range(0, ntoken):
                score_final += l[i] * np.log(l[i])
            score_final = score_final / ntoken #length normalized entropy

        return score_final


    def feedback(self, newSamples, dummy_train, words_conf, tags_conf, prob, enc, seq_len):
        '''
        feeds back data from validation set to retrain
        set in a self-training (active learning) setup

        function performs :
        1) selecting most confused samples
        2) retaining most representative samples from Step 1
        2) outlier removal

        newSamples : writing the most representative samples
                    to a different file for re-trainings.
        '''
        if config.similarity == 'None':
            index = np.arange(len(prob))
        else:
            print('\nClustering to find similarity \n')
            clusters= self.cluster_sentences(enc, seq_len, words_conf)

            most_representative_index = []
            for cluster in range(config.nclusters):
                clust = clusters[cluster]

                # ----------------------------------------------------------------
                # Select the top two lowest confidence sentences from each cluster.

                # The clusters which have lesser than a fixed number of samples are
                # considered as outliers and dropped.
                #-----------------------------------------------------------------
                most_representative_index += \
                    [x for _,x in sorted(zip(list(map(lambda x:prob[x],clust)),clust))][:2]
            index = sorted(most_representative_index)

        with open(newSamples, 'a') as handle1, \
            open(dummy_train, 'a') as handle2:
            for words, tags in zip(np.array(words_conf)[index], np.array(tags_conf)[index]):
                [handle1.write(word + ' ' + tag + '\n')
                        for word, tag in zip(words, tags) if word not in ['$UNK$', '$NUM$']]
                handle1.write('\n\n')
                [handle2.write(word + ' ' + tag + '\n')
                        for word, tag in zip(words, tags) if word not in ['$UNK$', '$NUM$']]
                handle2.write('\n\n')

        print('sentences fedback : ', len(index))
        return len(index)


    def random_sampling(self, train_file, newSamples, retrain_file, num_fedback):
        '''
        mixed samples from train data and the feedback samples
        function performs : mixed sampling for incremental training

        retrain_file : all mixed samples are written to the retrain file
        '''
        split_size = math.ceil(14986 / config.num_splits) # train_size = 14986

        #--------------------------------------------------------------------------------------
        # keeping a fixed proportion of samples from both train file and low confidence samples
        # num_fedback : total samples fedback from all active learning rounds
        #--------------------------------------------------------------------------------------
        train_samples = random.sample(range(split_size), config.sample_train)
        confused_samples = list(random.sample(range(int(num_fedback)),
                                              min(int(num_fedback * 0.5), config.sample_train)))

        def write(samples, filename, text, type):
            j = 0
            curr_line = None
            file = open(filename, 'r')
            with open(retrain_file, type) as sp:
                print('\n Writing {} {} to retrain file...\n'.format(len(samples), text))
                for line in file.readlines():
                    line = line.strip()
                    if line == '' and curr_line == '':
                        j += 1
                    curr_line = line
                    [sp.write(line + '\n') for i in samples if j == i]

        #--------------------------------------------------
        #  writing feedback samples to corresponding files
        #--------------------------------------------------
        write(confused_samples, newSamples, 'low confidence sentences', 'w')
        write(train_samples, train_file, 'new samples from train', 'a')


    def spectral_clustering(self, X, nclusters):
        #--------------------------------------------------------------
        # performs spectral clustering on predefined distance matrix X
        #-------------------------------------------------------------
        print('======Clustering======')
        clustering = SpectralClustering(n_clusters=nclusters, random_state=0, affinity='precomputed').fit(X)
        clusters = collections.defaultdict(list)

        for idx, label in enumerate(clustering.labels_):
            clusters[label].append(idx)
        return clusters


    def cluster_sentences(self, enc, seq_len, words_conf):
        # ------------------------------------------------------------
        # dynamic clustering depending on num low confidence samples
        # ------------------------------------------------------------
        n_clusters = int(len(seq_len) / config.num_clusters)

        print('\nSimilarity metric is {}\n'.format(config.similarity))
        if config.similarity == 'siamese':

            print("\nReloading the sentence similarity model...\n")
            graph = tf.Graph()
            with graph.as_default():
                sess = tf.Session()
                siamese = Siamese_Model(sess)

            #---------------------------------------------------
            # take all possible pairwise combinations of confused
            # samples to obtain the similarity scores pairwise.

            # The similarity matrix is symmetric.
            #---------------------------------------------------

            split1, split2 = np.array_split(np.arange(len(seq_len)),2)
            max_len = max(seq_len)
            seq_len1, seq_len2 = \
                [seq_len[i] for i in split1], [seq_len[i] for i in split2]

            if not config.model_aware:
                sent1, sent2 = [np.array(enc[i][0][0]).tolist() for i in split1], [np.array(enc[i][0][0]).tolist() for i in split2]
            else:
                sent1, sent2 = [enc[i] for i in split1], [enc[i] for i in split2]

            if config.model.split()[1] == 'LSTM' or not config.model_aware:
                dim = config.hidden_size_lstm
            else:
                dim = 2 * config.hidden_size_lstm

            for i, row in enumerate(sent1):
                if len(row) <= max_len:
                    sent1[i] += [np.zeros(dim).tolist()] * (max_len - len(row))
                    try:
                        sent2[i] += [np.zeros(dim).tolist()] * (max_len - len(sent2[i]))
                    except IndexError:
                        sent2 += [[np.zeros(dim).tolist()] * len(sent1[i])]
                        seq_len2 += [1]

            siamese_enc = np.concatenate(siamese.run(sent1, sent2, seq_len1, seq_len2,
                                                     max_len, len(split1), len(split2)))

            def similarity_scores(enc):
                shape = np.array(enc).shape
                out = np.reshape(np.repeat(enc, [shape[0]], axis=0), (-1,shape[0],shape[1]))
                X = np.exp(-1 * np.sqrt(np.sum(
                            np.square(out - np.transpose(out, (1,0,2))), 2, keepdims=False)))
                return X

            X = similarity_scores(siamese_enc)
            clustering = self.spectral_clustering(X, n_clusters)

        elif config.similarity == 'cosine':
            enc1 = [emb[-1] for emb in enc]
            X = np.exp(cos_sim(enc1, enc1))
            clustering = self.spectral_clustering(X, n_clusters)

        elif config.similarity == 'skipthoughts':
            model = skipthoughts.load_model()
            encoder = skipthoughts.Encoder(model)
            vectors = encoder.encode([' '.join(list) for list in words_conf])
            X = np.exp(cos_sim(vectors, vectors))
            #vectors = vectors / np.linalg.norm(vectors)
            #X = np.cos(np.dot(vectors, np.transpose(vectors)))
            clustering = self.spectral_clustering(X, n_clusters)

        return clustering




