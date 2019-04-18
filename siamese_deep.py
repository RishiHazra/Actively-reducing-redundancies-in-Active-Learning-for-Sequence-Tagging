import tensorflow as tf
from model.config import Config
config = Config()
#---------------------------------------------------------------------
# reference : https://github.com/dhwajraj/deep-siamese-text-similarity
#---------------------------------------------------------------------

class Siamese():

    def __init__(
            self, embedding_size, hidden_units, l2_reg_lambda, n_layers, batch_size):
        # -----------------------------------------------------------
        # Placeholders for input, output and dropout
        # -----------------------------------------------------------
        self.input_x1 = tf.placeholder(tf.float32, [None, None, 2 * config.hidden_size_lstm], name="input_x1")
        self.input_x2 = tf.placeholder(tf.float32, [None, None, 2 * config.hidden_size_lstm], name="input_x2")
        self.seq_len1 = tf.placeholder(tf.int32, [None], name='seq_len1')
        self.seq_len2 = tf.placeholder(tf.int32, [None], name='seq_len2')
        self.input_y = tf.placeholder(tf.float32, [None], name="input_y")
        self.max_len = tf.placeholder(tf.int32, name='max_seq_len')
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
        self.batch_size = tf.placeholder_with_default(batch_size, shape=[], name='batch_size_dynamic')


        # Keeping track of l2 regularization loss (optional)
        l2_loss = tf.constant(0.0, name="l2_loss")

        #----------------------------------------------------------
        # output scores
        #----------------------------------------------------------
        self.out1 = self.RNN(self.input_x1, self.dropout_keep_prob, "side1", embedding_size,
                               self.seq_len1, hidden_units, n_layers, self.batch_size, self.max_len)
        self.out2 = self.RNN(self.input_x2, self.dropout_keep_prob, "side2", embedding_size,
                               self.seq_len2, hidden_units, n_layers, self.batch_size, self.max_len)
        self.distance = tf.exp(-1*tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(self.out1, self.out2)),
                                                        1, keepdims=True)))
        #self.distance = tf.div(self.distance, tf.add(tf.sqrt(tf.reduce_sum(tf.square(self.out1), 1, keepdims=True)),
                                    #tf.sqrt(tf.reduce_sum(tf.square(self.out2), 1, keepdims=True))))
        with tf.name_scope('output'):
            self.out1 = tf.identity(self.out1, name='out1')
            self.out2 = tf.identity(self.out2, name='out2')
            self.distance = tf.reshape(self.distance, [-1], name="distance")

        # -----------------------------------------------------------
        # using contrastive loss
        # -----------------------------------------------------------
        with tf.name_scope('loss'):
            self.loss = self.contrastive_loss(self.input_y, self.distance, self.batch_size)


    def RNN(self, x, dropout, scope, embedding_size, sequence_length,
            hidden_units, n_layers, batch_size, max_length):

        with tf.variable_scope('fw'+scope):
            stacked_rnn_fw = []
            for _ in range(n_layers):
                fw_cell = tf.nn.rnn_cell.LSTMCell(num_units=hidden_units)
                drop_fw = tf.nn.rnn_cell.DropoutWrapper(fw_cell)
                stacked_rnn_fw.append(drop_fw)
            if n_layers > 1:
                drop_fw= tf.nn.rnn_cell.MultiRNNCell(stacked_rnn_fw)

        with tf.variable_scope('bw' + scope):
            stacked_rnn_bw = []
            for _ in range(n_layers):
                lstm_cell = tf.nn.rnn_cell.LSTMCell(num_units=hidden_units)
                drop_bw = tf.nn.rnn_cell.DropoutWrapper(lstm_cell)
                stacked_rnn_bw.append(drop_bw)
            if n_layers > 1:
                drop_bw = tf.nn.rnn_cell.MultiRNNCell(stacked_rnn_bw)

        with tf.variable_scope("bilstm" + scope):
            (output_fw, output_bw), _ = tf.nn.bidirectional_dynamic_rnn(drop_fw, drop_bw,
                                      inputs=x, sequence_length=sequence_length, dtype= tf.float32)
            outputs = tf.concat([output_fw, output_bw], axis=-1)

            out_size = int(outputs.get_shape()[2])
            index = tf.range(0, batch_size) * max_length + (sequence_length - 1)
            flat = tf.reshape(outputs, [-1, out_size])
            relevant = tf.gather(flat, index)
            return relevant


    def contrastive_loss(self, y, d, batch_size):
        tmp = tf.abs(y-d)
        #tmp = y * tf.square(d) + (1 - y) * tf.square(tf.maximum((1 - d), 0))
        return tf.reduce_mean(tmp) / 2
