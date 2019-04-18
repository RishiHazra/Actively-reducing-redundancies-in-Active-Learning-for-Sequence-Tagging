import tensorflow as tf

import os
import time
import warnings
import numpy as np
import argparse as ap
from random import random
import matplotlib.pyplot as plt

from model.config import Config
from siamese_deep import Siamese
from siamese_dataloader import Dataset

#-------------------------------------------------------------------
# arguments (batch size of the train and validation
#               are fixed keeping the dynamic rnn indexing
# Please don't alter !!!)
#-------------------------------------------------------------------
parser = ap.ArgumentParser()
parser.add_argument('--embedding_dim', type=int, default=600,
                    help='Dimension of NER model encoded embedding')
parser.add_argument('--dropout_keep_prob', type=int, default=1.0,
                    help='Dropout keep probability')
parser.add_argument('--l2_reg_lambda', type=int, default=0.0,
                    help='L2 regularization lambda')
parser.add_argument('--hidden_units', type=int, default=80,
                    help='Number of hidden units')
parser.add_argument('--batch_size', type=int, default=48,
                    help='Batch Size')
parser.add_argument('--val_batch', type=int, default=48,
                    help='Validation batch size')
parser.add_argument('--num_epochs', type=int, default=31,
                    help='Number of training epochs')
parser.add_argument('--evaluate_every', type=int, default=2,
                    help='Evaluate model on dev set after this many steps')
parser.add_argument('--checkpoint_every', type=int, default=10,
                    help='Save model after this many steps')
parser.add_argument('--n_layers', type=int, default=1,
                    help='Number of layers of Siamese network')
parser.add_argument('--split', type=float, default=0.8,
                    help='Train / dev split')
args = parser.parse_args()

#--------------------------------------------------------------------
# define class instances
#--------------------------------------------------------------------
config = Config()
dataloader = Dataset(config.dir_model_encoded_SICK, args.batch_size)
siameseModel = Siamese(
    embedding_size=args.embedding_dim,
    hidden_units=args.hidden_units,
    l2_reg_lambda=args.l2_reg_lambda,
    n_layers=args.n_layers,
    batch_size=args.batch_size
)

#--------------------------------------------------------------------
# checkpoint directory
#--------------------------------------------------------------------
checkpoint_prefix = os.path.join(config.dir_model_similarity,
                                 "siamese_model")
if not os.path.exists(config.dir_model_similarity):
    os.makedirs(config.dir_model_similarity)

#---------------------------------------------------------------------
# splitting into train and dev (total train samples = 9840)
#---------------------------------------------------------------------
train_samples = np.arange(1,9841)[:int(args.split * 9840)]
val_samples = np.arange(1,9841)[int(args.split * 9840):]

if not len(train_samples) % args.batch_size == 0:
    warnings.warn('Train batch size not suited for dynamic RNN indexing \n'
                  'Please change the train split size or batch_size.')

if not len(val_samples) % args.val_batch == 0:
    warnings.warn('Validation batch size not suited for dynamic RNN indexing \n'
                  'Please change the val split size or val_batch.')

#---------------------------------------------------------------------
# Plot histogram of labels
# to compare against actual labels
#---------------------------------------------------------------------
def plot_hist(predicted_lab, epoch):
    print('Plotting the histogram for validation labels')
    lab = np.load('similarity_labels.pkl')
    lab = list(map(float, lab))
    plt.hist(lab, label='act')
    plt.hist(predicted_lab, label='pred'+str(epoch))
    plt.show()

#---------------------------------------------------------------------
# Training step
#---------------------------------------------------------------------
def train_step(x1_batch, x2_batch, y_batch, seq_len, max_len, epoch):

    if random() > 0.5 :
        feed_dict = {
            siameseModel.input_x1 : x1_batch,
            siameseModel.input_x2 : x2_batch,
            siameseModel.seq_len1 : seq_len,
            siameseModel.seq_len2 : seq_len,
            siameseModel.input_y  : y_batch,
            siameseModel.max_len  : max_len,
            siameseModel.dropout_keep_prob : args.dropout_keep_prob
        }

    else :
        feed_dict = {
            siameseModel.input_x1 : x2_batch,
            siameseModel.input_x2 : x1_batch,
            siameseModel.seq_len1 : seq_len,
            siameseModel.seq_len2 : seq_len,
            siameseModel.input_y  : y_batch,
            siameseModel.max_len  : max_len,
            siameseModel.dropout_keep_prob : args.dropout_keep_prob
        }

    _, loss, dist = \
        sess.run([optimizer, siameseModel.loss,
                  siameseModel.distance], feed_dict=feed_dict)

    #print("TRAIN : epoch {}, loss {:g}".format(epoch, loss))
    #print(y_batch, '\n', dist)
    return loss

#----------------------------------------------------------------------
# Validation
#----------------------------------------------------------------------
def dev_step(x1_batch, x2_batch, y_batch, seq_len, max_len, plot=False):
    
     if random() > 0.5 :
         feed_dict = {
             siameseModel.input_x1 : x1_batch,
             siameseModel.input_x2 : x2_batch,
             siameseModel.seq_len1 : seq_len,
             siameseModel.seq_len2 : seq_len,
             siameseModel.input_y  : y_batch,
             siameseModel.max_len: max_len,
             siameseModel.dropout_keep_prob : 1.0
         }

     else :
         feed_dict = {
             siameseModel.input_x1 : x2_batch,
             siameseModel.input_x2 : x1_batch,
             siameseModel.seq_len1 : seq_len,
             siameseModel.seq_len2 : seq_len,
             siameseModel.input_y  : y_batch,
             siameseModel.max_len  : max_len,
             siameseModel.dropout_keep_prob : 1.0
         }
     loss, distance = \
             sess.run([siameseModel.loss,
                       siameseModel.distance], feed_dict=feed_dict)
     # print("DEV : step {}, loss {:g}".format(loss))
     return loss, distance

#------------------------------------------------------------------------
# Trainer 
#------------------------------------------------------------------------
opt = tf.train.AdamOptimizer(1e-5)
optimizer = opt.minimize(siameseModel.loss)

saver = tf.train.Saver(tf.global_variables(), max_to_keep=100)
init = tf.global_variables_initializer()

with tf.Session() as sess:
    tic = time.time()    
    sess.run(init)

    for epoch in range(args.num_epochs):
        total_train_loss, total_dev_loss = 0.0, 0.0
        splits = np.array_split(train_samples,
                                len(train_samples)/args.batch_size)

        for split in splits:
            x1_batch, x2_batch, y_batch, seq_len, max_len = \
                dataloader.get_item(split, args.embedding_dim)
            total_train_loss += train_step(x1_batch, x2_batch, y_batch, seq_len, max_len, epoch)
        print('train Loss {} for epoch {}'.format(total_train_loss, epoch))

        if epoch % args.evaluate_every == 0:
            print('\nEvaluation:')
            splits_val = np.array_split(val_samples, \
                         len(val_samples) / args.val_batch)

            lab_val = []
            for split in splits_val:
                x1_dev, x2_dev, y_dev, seq_len_val, max_len_val = \
                    dataloader.get_item(split, args.embedding_dim)
                try:
                    loss, lab = dev_step(x1_dev, x2_dev, y_dev, seq_len_val, max_len_val)
                    total_dev_loss += loss
                    lab_val += [lab]
                except ValueError:
                    continue
            print('dev Loss {} \n'.format(total_dev_loss))
    

        if epoch % args.checkpoint_every == 0:
           saver.save(sess, checkpoint_prefix)
           print("Saved model {} checkpoint to {}\n".
                      format(epoch, checkpoint_prefix))
           #plot_hist(np.reshape(lab_val,-1), epoch)

    toc = time.time()
    print('tic-toc : {} sec'.format(toc-tic))