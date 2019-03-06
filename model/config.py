import os
from .general_utils import get_logger
from .arguments import Arguments
from .data_utils import get_trimmed_glove_vectors, load_vocab, \
        get_processing_word, get_sent_similarity_score


class Config():
    def __init__(self, load=True):
        """Initialize hyperparameters and load vocabs
        Args:
            load_embeddings: (bool) if True, load embeddings into
                np array, else None
        """
        # directory for training outputs
        if not os.path.exists(self.dir_output):
            os.makedirs(self.dir_output)

        # create instance of logger
        self.logger = get_logger(self.path_log)
        # load if requested (default)
        if load:
            self.load()


    def load(self):
        """Loads vocabulary, processing functions and embeddings
        Supposes that build_data.py has been run successfully and that
        the corresponding files have been created (vocab and trimmed GloVe
        vectors)
        """
        # 1. vocabulary
        self.vocab_words = load_vocab(self.filename_words)
        self.vocab_tags  = load_vocab(self.filename_tags)
        self.vocab_chars = load_vocab(self.filename_chars)

        self.nwords     = len(self.vocab_words)
        self.nchars     = len(self.vocab_chars)
        self.ntags      = len(self.vocab_tags)

        # 2. get processing functions that map str -> id
        self.processing_word = get_processing_word(self.vocab_words,
                self.vocab_chars, lowercase=True, chars=self.use_chars)
        self.processing_tag  = get_processing_word(self.vocab_tags,
                lowercase=False, allow_unk=False)

        # 3. get pre-trained embeddings
        self.embeddings = (get_trimmed_glove_vectors(self.filename_trimmed)
                if self.use_pretrained else None)

        # 4. get processed labels for sentence similarity
        self.processing_scores = get_sent_similarity_score()

    # general config   
    #------------------------------------------------------------------
    # directories 
    #------------------------------------------------------------------
    dir_output = "test/"
    dir_model  = dir_output + "model.weights/"
    dir_model_similarity = dir_output + "similarity_ckpt"
    dir_model_encoded_SICK = dir_output + 'sick_encoded/'
    path_log   = dir_output + "log.txt"
    args = Arguments()
    
    #--------------------------------------------------------------------
    # embeddings
    #------------------------------------------------------------------
    dim_word = args.dim_word
    dim_char = args.dim_char
    num_splits = args.num_splits

    #------------------------------------------------------------------
    # glove files
    #------------------------------------------------------------------
    filename_glove = "ner/data/glove.6B/glove.6B.{}d.txt".format(dim_word)
    filename_trimmed = "ner/data/glove.6B.{}d.trimmed.npz".format(dim_word)
    use_pretrained = True

    #------------------------------------------------------------------
    # dataset
    #------------------------------------------------------------------
    filename_train = "ner/data/ner_train.txt"
    filename_dev = "ner/data/ner_dev.txt"
    filename_test = "ner/data/ner_test.txt"

    train_split = [[]]*num_splits
    for i in range(1,num_splits+1):
        train_split[i-1] = "ner/train_split_{}.txt".format(i)

    filename_retrain = "ner/train_samples.txt"
    filename_sick = "ner/data/sick_processed.txt"
    filename_newSamples = "ner/newSamples.txt"

    max_iter = None # if not None, max number of examples in Dataset

    #------------------------------------------------------------------
    # vocab (created from dataset with build_data.py)
    #------------------------------------------------------------------
    filename_words = "ner/words.txt"
    filename_tags = "ner/tags.txt"
    filename_chars = "ner/chars.txt"
    
    #------------------------------------------------------------------
    # training parameters
    #------------------------------------------------------------------
    train_embeddings = False
    nepochs          = args.nepochs
    dropout          = args.dropout
    batch_size       = args.batch_size
    lr_method        = args.optimizer
    lr               = args.lr
    lr_decay         = args.lr_decay
    clip             = args.clip # if negative, no clipping
    nepoch_no_imprv  = args.early_stop
    
    #------------------------------------------------------------------
    # model hyperparameters
    #------------------------------------------------------------------
    hidden_size_char = 100 # lstm on chars
    hidden_size_lstm = 300 # lstm on word embeddings

    # NOTE: if both chars and crf, only 1.6x slower on GPU
    use_crf = args.use_crf # if crf, training is 1.7x slower on CPU
    use_chars = args.use_chars # if char embedding, training is 3.5x slower on CPU
    
    #------------------------------------------------------------------
    # other parameters
    #------------------------------------------------------------------
    num_query = args.num_query  # number of queries each round of active learning
    active_algo = args.active_strategy
    split = args.split
    sample_split = args.sample_split
    num_retrain = args.num_retrain
    nclusters = args.nclusters
    similarity = args.similarity
    encode = args.encode


    file_out = "results/files/out_" + str(num_query) + "query_" + active_algo
    mode = args.mode