import argparse as ap

def Arguments():

    parser = ap.ArgumentParser()
    parser.add_argument('--dim_word', type=int, default=300,
                        help = 'glove embedding dimension')
    parser.add_argument('--dim_char', type=int, default=100,
                        help='character encoding dimension')
    parser.add_argument('--num_splits', type=int, default=10,
                        help='num_splits for active learning')
    parser.add_argument('--nepochs', type=int, default=7,
                        help='number of epochs of training/retraining')
    parser.add_argument('--dropout', type=int, default=0.5,
                        help='dropout')
    parser.add_argument('--batch_size', type=int, default=20,
                        help='batch size')
    parser.add_argument('--optimizer', type=str, default='adam',
                        help='optimization method : adam/adagrad/sgd/rmsprop')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate')
    parser.add_argument('--lr_decay', type=float, default=0.9,
                        help='decay in learning rate')
    parser.add_argument('--use_crf', type=bool, default=True,
                        help='use CRF ?')
    parser.add_argument('--use_chars', type=bool, default=True,
                        help='use char encoder ?')
    parser.add_argument('--num_query', type=int, default=40,
                        help='number of query examples')
    parser.add_argument('--active_strategy', type=str, default='entropy',
                        help='active learning strategy')
    parser.add_argument('--clip', type=int, default=-1,
                        help='gradient clipping')
    parser.add_argument('--early_stop', type=int, default=3,
                        help='stop if no improvement')
    parser.add_argument('--mode', type=str, default='feedback',
                        help='train, retrain, feedback')
    parser.add_argument('--split', type=int, default=0,
                        help='split to train on')
    parser.add_argument('--sample_split', type=int, default=1,
                        help='split to feedback/sample from')
    parser.add_argument('--num_retrain', type=int, default=1,
                        help='number of retrains so far')
    parser.add_argument('--nclusters', type=int, default=20,
                        help='number of clusters')
    parser.add_argument('--similarity', type=str, default='siamese',
                        help='cosine/siamese/skip thoughts')
    parser.add_argument('--encode', type=bool, default=False,
                        help='Encode SICK dataset using NER model')

    return parser.parse_args()