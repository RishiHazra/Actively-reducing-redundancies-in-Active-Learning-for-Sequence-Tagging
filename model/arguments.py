import argparse as ap

def Arguments():
    parser = ap.ArgumentParser()
    # -------------------------------------------------------------
    # dynamic arguments
    # -------------------------------------------------------------
    parser.add_argument('--nepochs', type=int, default=65,
                        help='number of epochs of training/retraining')
    parser.add_argument('--mode', type=str, default='train',
                        help='train, retrain, feedback')
    parser.add_argument('--split', type=int, default=0,
                        help='split to train on')
    parser.add_argument('--sample_split', type=int, default=2,
                        help='split to feedback/sample from')
    parser.add_argument('--num_retrain', type=int, default=1,
                        help='number of retrains so far')
    parser.add_argument('--sample_train', type=int, default=150,
                        help='number of train samples for each retrain step')
    parser.add_argument('--active_strategy', type=str, default='margin',
                        help='active learning strategy')
    parser.add_argument('--nclusters', type=int, default=20,
                        help='number of clusters')
    parser.add_argument('--similarity', type=str, default='siamese',
                        help='cosine/siamese/skipthoughts')
    parser.add_argument('--encode', type=bool, default=False,
                        help='Encode SICK dataset using NER model')
    parser.add_argument('--num_clusters', type=int, default=10,
                        help='Number of clusters for 2nd layer of AL')
    parser.add_argument('--threshold', type=float, default=15,
                        help='confusion based threshold')
    parser.add_argument('--periodic', type=bool, default=False,
                        help='periodically train the model completely')
    parser.add_argument('--sample_times', type=int, default=51,
                        help='Number of forward passes for BALD ALS')
    parser.add_argument('--model', type=str, default="CNN BILSTM CRF",
                        help='model used for active learning')
    parser.add_argument('--excel_id', type=int, default=1,
                        help='excel train id')
    parser.add_argument('--variational_dropout', type=bool,
                        default=False, help='variational_dropout')
    parser.add_argument('--model_aware', type=bool, default=False,
                        help='model aware or not')
    # -------------------------------------------------------------
    # fixed arguments
    # -------------------------------------------------------------
    parser.add_argument('--dim_word', type=int, default=300,
                        help = 'glove embedding dimension')
    parser.add_argument('--dim_char', type=int, default=100,
                        help='character encoding dimension')
    parser.add_argument('--num_splits', type=int, default=50,
                        help='num_splits for active learning')
    parser.add_argument('--dropout', type=int, default=0.5,
                        help='dropout')
    parser.add_argument('--batch_size', type=int, default=12,
                        help='batch size')
    parser.add_argument('--optimizer', type=str, default='adam',
                        help='optimization method : adam/adagrad/sgd/rmsprop')
    parser.add_argument('--lr', type=float, default=0.005,
                        help='learning rate')
    parser.add_argument('--lr_decay', type=float, default=0.9,
                        help='decay in learning rate')
    parser.add_argument('--use_crf', type=bool, default=True,
                        help='use CRF ?')
    parser.add_argument('--use_chars', type=bool, default=True,
                        help='use char encoder ?')
    parser.add_argument('--clip', type=int, default=-1,
                        help='gradient clipping')
    parser.add_argument('--early_stop', type=int, default=14,
                        help='stop if no improvement')
    parser.add_argument('--filter_sizes', type=list, default=[1,2,3],
                        help='char CNN filter sizes')

    return parser.parse_args()
