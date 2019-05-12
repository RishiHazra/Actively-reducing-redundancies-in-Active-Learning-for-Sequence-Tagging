import tensorflow as tf
from tensorflow.python.ops import gen_nn_ops

#----------------------------------------------------------------------------------------
# Four Modules (Character-Word Encoder Models):
# 1) CNN-BiLSTM
# 2) CNN-CNN
# 3) BiLSTM-BiLSTM
# 4) LSTM-LSTM
#----------------------------------------------------------------------------------------

class CNN_BILSTM_CRF():
    def __init__(self, config, word_ids, sequence_lengths, char_ids, word_lengths, labels,
                 dropout, lr):
        self.config = config
        self.word_ids = word_ids
        self.sequence_lengths = sequence_lengths
        self.char_ids = char_ids
        self.word_lengths = word_lengths
        self.labels = labels
        self.dropout = dropout
        self.lr = lr

    def add_word_embeddings_op(self):
        """Defines self.word_embeddings
        If self.config.embeddings is not None and is a np array initialized
        with pre-trained word vectors, the word embeddings is just a look-up
        and we don't train the vectors. Otherwise, a random matrix with
        the correct shape is initialized.
        """
        with tf.variable_scope("words"):
            if self.config.embeddings is None:
                print("WARNING: randomly initializing word vectors")
                _word_embeddings = tf.get_variable(
                    name="_word_embeddings",
                    dtype=tf.float32,
                    shape=[self.config.nwords, self.config.dim_word])
            else:
                _word_embeddings = tf.Variable(
                    self.config.embeddings,
                    name="_word_embeddings",
                    dtype=tf.float32,
                    trainable=self.config.train_embeddings)

            word_embeddings = tf.nn.embedding_lookup(_word_embeddings,
                                                     self.word_ids, name="word_embeddings")
        self.model_unaware_embedding = tf.identity(word_embeddings)

        pooled_outputs = []
        for i, filter_size in enumerate(self.config.filter_sizes):
            with tf.variable_scope("char_CNN" + str(i)):
                if self.config.use_chars:
                    _char_embeddings = tf.get_variable(
                        name="_char_embeddings",
                        dtype=tf.float32,
                        shape=[self.config.nchars, self.config.dim_char])

                    # [shape = (batch, sentence, word, dim of char emb)]
                    char_embeddings = tf.nn.embedding_lookup(_char_embeddings,
                                                             self.char_ids, name="char_embeddings")

                    # put the time dimension on axis=1
                    s = tf.shape(char_embeddings)
                    self.shape = s
                    char_embeddings = tf.reshape(char_embeddings,
                                                 shape=[s[0] * s[1], s[-2], self.config.dim_char, 1])

                    conv_weight = tf.get_variable(
                        shape=[filter_size, self.config.dim_char, 1, self.config.hidden_size_char],
                        initializer=tf.truncated_normal_initializer(stddev=0.01),
                        name='conv_weights'
                    )
                    conv_bias = tf.get_variable(
                        shape=[self.config.hidden_size_char],
                        initializer=tf.zeros_initializer(),
                        name='conv_bias'
                    )

                    # shape = [batch*sent_len, out_height, out_width, 2*self.config.dim_char]
                    conv = tf.nn.conv2d(char_embeddings, conv_weight, strides=[1, 1, 1, 1], padding='VALID')
                    conv = tf.nn.relu(tf.nn.bias_add(conv, conv_bias))
                    pooled = gen_nn_ops.max_pool_v2(conv,
                                                    ksize=[1, s[-2] - filter_size + 1, 1, 1],
                                                    strides=[1, 1, 1, 1],
                                                    padding='VALID')

                    conv = tf.reshape(pooled, shape=[s[0], s[1], self.config.hidden_size_char])
                    conv = tf.nn.dropout(conv, self.dropout)
                    pooled_outputs.append(conv)
            conv = tf.concat([op for op in pooled_outputs], axis=-1)

        self.word_embeddings = tf.concat([word_embeddings, conv], axis=-1)
        # self.word_embeddings = tf.nn.dropout(word_embeddings, self.dropout)


    def add_logits_op(self):
        """Defines self.logits
        For each word in each sentence of the batch, it corresponds to a vector
        of scores, of dimension equal to the number of tags.
        """
        with tf.variable_scope("bi-lstm"):
            cell_fw = tf.nn.rnn_cell.LSTMCell(self.config.hidden_size_lstm,
                                              state_is_tuple=True)
            cell_bw = tf.nn.rnn_cell.LSTMCell(self.config.hidden_size_lstm,
                                              state_is_tuple=True)

            #-------------------------------------------------------------------------------------
            # Variational Dropout : the same dropout mask at each time step for both inputs,
            # outputs, and recurrent layers (drop the same network units at each time step)
            # Gal and Ghahramani (2015)
            #-------------------------------------------------------------------------------------
            if self.config.variational_dropout:
                print('\nusing Variational Dropout for Word Encoder\n')
                cell_fw = tf.nn.rnn_cell.DropoutWrapper(cell_fw, input_keep_prob=self.config.dropout,
                                                        output_keep_prob=self.config.dropout,
                                                        state_keep_prob=self.config.dropout,
                                                        dtype=tf.float32,
                                                        variational_recurrent=True,
                                                        input_size=2648)
                cell_bw = tf.nn.rnn_cell.DropoutWrapper(cell_bw, input_keep_prob=self.config.dropout,
                                                        output_keep_prob=self.config.dropout,
                                                        state_keep_prob=self.config.dropout,
                                                        dtype=tf.float32,
                                                        variational_recurrent=True,
                                                        input_size=2648)

            (output_fw, output_bw), _ = tf.nn.bidirectional_dynamic_rnn(
                cell_fw, cell_bw, self.word_embeddings,
                sequence_length=self.sequence_lengths, dtype=tf.float32)
            output = tf.concat([output_fw, output_bw], axis=-1)
            output = tf.nn.dropout(output, self.dropout)

        with tf.variable_scope("proj"):
            W = tf.get_variable("W", dtype=tf.float32,
                                shape=[2 * self.config.hidden_size_lstm, self.config.ntags])

            b = tf.get_variable("b", shape=[self.config.ntags],
                                dtype=tf.float32, initializer=tf.zeros_initializer())

            nsteps = tf.shape(output)[1]
            output = tf.reshape(output, [-1, 2 * self.config.hidden_size_lstm])
            self.encoded = output
            pred = tf.matmul(output, W) + b
            self.logits = tf.reshape(pred, [-1, nsteps, self.config.ntags])






class BILSTM_BILSTM_CRF_attention():
    def __init__(self, config, word_ids, sequence_lengths, char_ids, word_lengths, labels,
                 dropout, lr):
        self.config = config
        self.word_ids = word_ids
        self.sequence_lengths = sequence_lengths
        self.char_ids = char_ids
        self.word_lengths = word_lengths
        self.labels = labels
        self.dropout = dropout
        self.lr = lr

    def add_word_embeddings_op(self):
        """Defines self.word_embeddings
        If self.config.embeddings is not None and is a np array initialized
        with pre-trained word vectors, the word embeddings is just a look-up
        and we don't train the vectors. Otherwise, a random matrix with
        the correct shape is initialized.
        """
        loss = 0
        with tf.variable_scope("words"):
            if self.config.embeddings is None:
                print("WARNING: randomly initializing word vectors")
                _word_embeddings = tf.get_variable(
                    name="_word_embeddings",
                    dtype=tf.float32,
                    shape=[self.config.nwords, self.config.dim_word])
            else:
                _word_embeddings = tf.Variable(
                    self.config.embeddings,
                    name="_word_embeddings",
                    dtype=tf.float32,
                    trainable=self.config.train_embeddings)

            word_embeddings = tf.nn.embedding_lookup(_word_embeddings,
                                                     self.word_ids, name="word_embeddings")
        self.model_unaware_embedding = tf.identity(word_embeddings)

        with tf.variable_scope("chars"):
            if self.config.use_chars:
                # get char embeddings matrix
                _char_embeddings = tf.get_variable(
                    name="_char_embeddings",
                    dtype=tf.float32,
                    shape=[self.config.nchars, self.config.dim_char])

                # [shape = (batch, sentence, word, dim of char emb)]
                char_embeddings = tf.nn.embedding_lookup(_char_embeddings,
                                                         self.char_ids, name="char_embeddings")

                # put the time dimension on axis=1
                s = tf.shape(char_embeddings)
                self.shape = s
                char_embeddings = tf.reshape(char_embeddings,
                                             shape=[s[0] * s[1], s[-2], self.config.dim_char])
                word_lengths = tf.reshape(self.word_lengths, shape=[s[0] * s[1]])

                # bi lstm on chars
                cell_fw = tf.contrib.rnn.LSTMCell(self.config.hidden_size_char,
                                                  state_is_tuple=True)
                cell_bw = tf.contrib.rnn.LSTMCell(self.config.hidden_size_char,
                                                  state_is_tuple=True)
                _output = tf.nn.bidirectional_dynamic_rnn(
                    cell_fw, cell_bw, char_embeddings,
                    sequence_length=word_lengths, dtype=tf.float32)

                # read and concat output
                _, ((_, output_fw), (_, output_bw)) = _output
                # shape = [batch * sentence, 2 * char_hidden_size]
                output = tf.concat([output_fw, output_bw], axis=-1)

                # shape = (batch size, max sentence length, 2*char hidden size)
                self.output = tf.reshape(output,
                                         shape=[s[0], s[1], 2 * self.config.hidden_size_char])
                word_embeddings = tf.concat([word_embeddings, self.output], axis=-1)

        self.word_embeddings = tf.nn.dropout(word_embeddings, self.dropout)

    def add_logits_op(self):
        """Defines self.logits
        For each word in each sentence of the batch, it corresponds to a vector
        of scores, of dimension equal to the number of tags.
        """
        with tf.variable_scope("bi-lstm"):
            cell_fw = tf.contrib.rnn.LSTMCell(self.config.hidden_size_lstm)
            cell_bw = tf.contrib.rnn.LSTMCell(self.config.hidden_size_lstm)
            # -------------------------------------------------------------------------------------
            # Variational Dropout : the same dropout mask at each time step for both inputs,
            # outputs, and recurrent layers (drop the same network units at each time step)
            # Gal and Ghahramani (2015)
            # -------------------------------------------------------------------------------------
            if self.config.variational_dropout:
                cell_fw = tf.nn.rnn_cell.DropoutWrapper(cell_fw, input_keep_prob=self.config.dropout,
                                                        output_keep_prob=self.config.dropout,
                                                        state_keep_prob=self.config.dropout,
                                                        dtype=tf.float32,
                                                        variational_recurrent=True,
                                                        input_size=600)
                cell_bw = tf.nn.rnn_cell.DropoutWrapper(cell_bw, input_keep_prob=self.config.dropout,
                                                        output_keep_prob=self.config.dropout,
                                                        state_keep_prob=self.config.dropout,
                                                        dtype=tf.float32,
                                                        variational_recurrent=True,
                                                        input_size=600)
            (output_fw, output_bw), _ = tf.nn.bidirectional_dynamic_rnn(
                cell_fw, cell_bw, self.word_embeddings,
                sequence_length=self.sequence_lengths, dtype=tf.float32)
            output = tf.concat([output_fw, output_bw], axis=-1)

            word_output_tensor = tf.layers.dense(output, 2 * self.config.dim_word, activation=tf.tanh)
            char_output_tensor = tf.layers.dense(self.output, 2 * self.config.dim_word, activation=tf.tanh)
            attention_evidence_tensor = tf.concat([word_output_tensor, char_output_tensor], axis=2)

            # ---------------------------------------------------------
            # passing through 2-dense layers
            # ---------------------------------------------------------
            attention_output = tf.layers.dense(attention_evidence_tensor, 2 * self.config.dim_word,
                                               activation=tf.tanh)
            attention_output = tf.layers.dense(attention_output, 2 * self.config.dim_word,
                                               activation=tf.sigmoid)

            # ----------------------------------------------------------
            # convex combination of char biLSTM o/p and word biLSTM o/p
            # ----------------------------------------------------------
            output = tf.multiply(word_output_tensor, attention_output) + tf.multiply(char_output_tensor,
                                                                                     (1.0 - attention_output))
            output = tf.nn.dropout(output, self.dropout)

        with tf.variable_scope("proj"):
            W = tf.get_variable("W", dtype=tf.float32,
                                shape=[2 * self.config.hidden_size_lstm, self.config.ntags])

            b = tf.get_variable("b", shape=[self.config.ntags],
                                dtype=tf.float32, initializer=tf.zeros_initializer())

            nsteps = tf.shape(output)[1]
            output = tf.reshape(output, [-1, 2 * self.config.hidden_size_lstm])
            self.encoded = output
            pred = tf.matmul(output, W) + b
            self.logits = tf.reshape(pred, [-1, nsteps, self.config.ntags])







class LSTM_LSTM_CRF():
    def __init__(self, config, word_ids, sequence_lengths, char_ids, word_lengths, labels,
                 dropout, lr):
        self.config = config
        self.word_ids = word_ids
        self.sequence_lengths = sequence_lengths
        self.char_ids = char_ids
        self.word_lengths = word_lengths
        self.labels = labels
        self.dropout = dropout
        self.lr = lr

    def add_word_embeddings_op(self):
        """Defines self.word_embeddings
        If self.config.embeddings is not None and is a np array initialized
        with pre-trained word vectors, the word embeddings is just a look-up
        and we don't train the vectors. Otherwise, a random matrix with
        the correct shape is initialized.
        """
        with tf.variable_scope("words"):
            if self.config.embeddings is None:
                print("WARNING: randomly initializing word vectors")
                _word_embeddings = tf.get_variable(
                    name="_word_embeddings",
                    dtype=tf.float32,
                    shape=[self.config.nwords, self.config.dim_word])
            else:
                _word_embeddings = tf.Variable(
                    self.config.embeddings,
                    name="_word_embeddings",
                    dtype=tf.float32,
                    trainable=self.config.train_embeddings)

            word_embeddings = tf.nn.embedding_lookup(_word_embeddings,
                                                     self.word_ids, name="word_embeddings")

        with tf.variable_scope("chars"):
            if self.config.use_chars:
                # get char embeddings matrix
                _char_embeddings = tf.get_variable(
                    name="_char_embeddings",
                    dtype=tf.float32,
                    shape=[self.config.nchars, self.config.dim_char])

                # [shape = (batch, sentence, word, dim of char emb)]
                char_embeddings = tf.nn.embedding_lookup(_char_embeddings,
                                                         self.char_ids, name="char_embeddings")

                # put the time dimension on axis=1
                s = tf.shape(char_embeddings)
                self.shape = s
                char_embeddings = tf.reshape(char_embeddings,
                                             shape=[s[0] * s[1], s[-2], self.config.dim_char])
                word_lengths = tf.reshape(self.word_lengths, shape=[s[0] * s[1]])

                # lstm on chars
                cell_fw = tf.nn.rnn_cell.LSTMCell(self.config.hidden_size_char,
                                                  state_is_tuple=True)
                output, _ = tf.nn.dynamic_rnn(
                    cell_fw, char_embeddings,
                    sequence_length=word_lengths, dtype=tf.float32)

                # shape = (batch size, max sentence length, char hidden size)
                output = tf.reshape(output[:, -1, :],
                                    shape=[s[0], s[1], self.config.hidden_size_char])
                word_embeddings = tf.concat([word_embeddings, output], axis=-1)
        self.word_embeddings = tf.nn.dropout(word_embeddings, self.dropout)


    def add_logits_op(self):
        """Defines self.logits
        For each word in each sentence of the batch, it corresponds to a vector
        of scores, of dimension equal to the number of tags.
        """
        with tf.variable_scope("bi-lstm"):
            cell_fw = tf.nn.rnn_cell.LSTMCell(self.config.hidden_size_lstm,
                                              state_is_tuple=True)
            # -------------------------------------------------------------------------------------
            # Variational Dropout : the same dropout mask at each time step for both inputs,
            # outputs, and recurrent layers (drop the same network units at each time step)
            # Gal and Ghahramani (2015)
            # -------------------------------------------------------------------------------------
            if self.config.variational_dropout:
                cell_fw = tf.nn.rnn_cell.DropoutWrapper(cell_fw, input_keep_prob=self.config.dropout,
                                                        output_keep_prob=self.config.dropout,
                                                        state_keep_prob=self.config.dropout,
                                                        dtype=tf.float32,
                                                        variational_recurrent=True,
                                                        input_size=400)

            output, _ = tf.nn.dynamic_rnn(
                cell_fw, self.word_embeddings,
                sequence_length=self.sequence_lengths, dtype=tf.float32)
            self.output = tf.nn.dropout(output, self.dropout)

        with tf.variable_scope("proj"):
            W = tf.get_variable("W", dtype=tf.float32,
                                shape=[self.config.hidden_size_lstm, self.config.ntags])

            b = tf.get_variable("b", shape=[self.config.ntags],
                                dtype=tf.float32, initializer=tf.zeros_initializer())

            nsteps = tf.shape(output)[1]
            output = tf.reshape(output, [-1, self.config.hidden_size_lstm])
            self.encoded = output
            pred = tf.matmul(output, W) + b
            self.logits = tf.reshape(pred, [-1, nsteps, self.config.ntags])
