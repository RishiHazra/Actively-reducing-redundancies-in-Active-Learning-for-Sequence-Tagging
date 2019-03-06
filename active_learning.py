from model.config import Config
import numpy as np
import math
import random
import collections
from itertools import combinations
from sklearn.cluster import KMeans
from sklearn.cluster import SpectralClustering
import tensorflow as tf

config = Config()

class Siamese_Model():
    def __init__(self, session):
        self.sess = session

        self.saver = tf.train.import_meta_graph(config.dir_model_similarity
                                           + '/siamese_model.meta')
        self.saver.restore(self.sess, config.dir_model_similarity
                                           + '/siamese_model')
        self.siamese_graph = tf.get_default_graph()
        self.distance = self.siamese_graph.get_tensor_by_name('output/distance:0')

    def run(self, x1_batch, x2_batch, seq_len, max_len):
        feed_dict = {
            'input_x1:0': x1_batch,
            'input_x2:0': x2_batch,
            'seq_len:0': seq_len,
            'max_seq_len:0': max_len,
            'dropout_keep_prob:0': 1
        }
        return self.sess.run(self.distance, feed_dict=feed_dict)

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

        if (self.active_algo == 'mg'):
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
            score_final = score_final / ntoken

        return score_final


    def feedback(self, AL_file, newSamples, ind_confused, PROB, enc, seq_len, session):
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
        print('Clustering to find similarity')
        clusters= self.cluster_sentences(np.array(enc), seq_len, config.nclusters, session)

        most_representative_index = []
        for cluster in range(config.nclusters):
            #ind = i[random.choice(clusters[cluster])]
            clust = list(map(lambda x: ind_confused[x], clusters[cluster]))

            # ----------------------------------------------------------------
            # Select the top two lowest confidence sentences from each cluster.

            # The clusters which have lesser than a fixed number of samples are
            # considered as outliers and dropped.
            #-----------------------------------------------------------------
            most_representative_index += \
                [x for _,x in sorted(zip(list(map(lambda x:PROB[x],clust)),clust))][:2]

        index = sorted(most_representative_index )
        curr_line = None


        with open(newSamples, 'a') as feedback:
            for ind in index:
                j = 0
                fin = open(AL_file, 'r')
                for line in fin.readlines()[:-1]:
                    line = line.strip()

                    if line == '' or curr_line == '':
                        if curr_line != line: j += 1
                    curr_line = line
                    if j == 2 * (ind - 1):
                        if len(line) > 1:
                            feedback.write(line)
                            feedback.write('\n')
                        else:
                            continue
                    if j > 2 * (ind - 1) and j < 2 * (ind): feedback.write('\n')

        print('sentences fedback : ', len(index))


    def random_sampling(self, x, train_file, newSamples, retrain_file):
        '''
        mixed samples from train data and the feedback samples (40 batches out of 250)
        function performs : mixed sampling for incremental training

        retrain_file : all mixed samples are written to the retrain file
        '''
        split_size = math.ceil(14986 / config.num_splits) # train_size = 14986
        a = random.sample(range(split_size), 1600)
        b = list(range(40*x,40*(x+1)))
        if x!=0:
            b += list(random.sample(range(40*(x)), 280))

        def write(samples, filename, text):
            j = 0
            curr_line = None
            file = open(filename, 'r')
            with open(retrain_file, 'w') as sp:
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
        write(b, newSamples, 'low confidence sentences')
        write(a, train_file, 'new samples from train')



    def cluster_sentences(self, enc, seq_len, n_clusters, session):
        kmeans = KMeans(n_clusters=n_clusters)

        if config.similarity == 'siamese':
            max_len = max(seq_len)
            X = np.zeros((len(seq_len), len(seq_len)))

            print("\n Reloading the sentence similarity model...")
            siamese = Siamese_Model(session)

            #---------------------------------------------------
            # take all possible pairwisecombinations of confused
            # samples to obtain the similarity scores pairwise.

            # The similarity matrix is symmetric.
            #---------------------------------------------------
            ind_comb = list(combinations(range(len(seq_len)), 2))

            for ind1, ind2 in ind_comb:
                sent1, sent2 = enc[ind1], enc[ind2]
                X[ind1, ind2] = X[ind2, ind1] = \
                     siamese.run(sent1, sent2, seq_len, max_len)

            np.fill_diagonal(X, 1)
            kmeans.fit(X)

        elif config.similarity == 'cosine':
            sents = enc / np.linalg.norm(enc)
            X = np.cos(np.matmul(sents, np.transpose(sents)))
            _, eigen_vectors = np.linalg.eigh(X)
            kmeans.fit(eigen_vectors)

        clustering = collections.defaultdict(list)
        for idx, label in enumerate(kmeans.labels_):
            clustering[label].append(idx)

        return clustering




