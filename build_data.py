import math
from model.config import Config
from model.data_utils import CoNLLDataset, get_vocabs, UNK, NUM, \
    get_glove_vocab, write_vocab, load_vocab, get_char_vocab, \
    export_trimmed_glove_vectors, get_processing_word


def split_train(config):
    """   
     function to split the train set into the required number of equal splits.
      
    """
    curr_line = None
    lower_lim, upper_lim = 0, math.ceil(14986 / config.num_splits)
    for i in range(1, config.num_splits+1):
        j = 0
        f = open(config.filename_train)  # train_samples = 14986
        with open(config.train_split[i-1], 'w') as handle:
            for line in f.readlines():
                line = line.strip()
                if line == '-DOCSTART- O':
                    continue
                if line == '' and curr_line == '': j+=1
                curr_line = line
                if j >= lower_lim and j < upper_lim:
                    handle.write(line + '\n')
                elif j >= upper_lim:
                    lower_lim = upper_lim
                    upper_lim += math.ceil(14986 / config.num_splits)
                    break
    f.close()
    open(config.dummy_train, "w").writelines([l for l in open(config.train_split[config.split]).readlines()])


def main():
    """Procedure to build data
    You MUST RUN this procedure. It iterates over the whole dataset (train,
    dev and test) and extract the vocabularies in terms of words, tags, and
    characters. Having built the vocabularies it writes them in a file. The
    writing of vocabulary in a file assigns an id (the line #) to each word.
    It then extract the relevant GloVe vectors and stores them in a np array
    such that the i-th entry corresponds to the i-th word in the vocabulary.
    Args:
        config: (instance of Config) has attributes like hyper-params...
    """
    # get config and processing of words
    config = Config(load=False)
    processing_word = get_processing_word(lowercase=True)
    logger = config.logger
    
    #------------------------------------------------------------------
    # Generators
    # ------------------------------------------------------------------
    dev   = CoNLLDataset(config.filename_dev, processing_word)
    test  = CoNLLDataset(config.filename_test, processing_word)
    train = CoNLLDataset(config.filename_train, processing_word)
    sick  = CoNLLDataset(config.filename_sick, processing_word)

    # ------------------------------------------------------------------
    # Build Word and Tag vocab
    # ------------------------------------------------------------------
    vocab_words, vocab_tags = get_vocabs([train, dev, test, sick])
    vocab_glove = get_glove_vocab(config.filename_glove)

    vocab = vocab_words & vocab_glove
    vocab.add(UNK)
    vocab.add(NUM)

    # ------------------------------------------------------------------
    # Save vocab
    # ------------------------------------------------------------------
    write_vocab(vocab, config.filename_words)
    write_vocab(vocab_tags, config.filename_tags)

    # ------------------------------------------------------------------
    # Trim GloVe Vectors
    # ------------------------------------------------------------------
    vocab, _ = load_vocab(config.filename_words)
    export_trimmed_glove_vectors(vocab, config.filename_glove,
                                config.filename_trimmed, config.dim_word)

    # ------------------------------------------------------------------
    # Build and save char vocab
    # ------------------------------------------------------------------
    train = CoNLLDataset(config.filename_train)
    vocab_chars = get_char_vocab(train)
    write_vocab(vocab_chars, config.filename_chars)

    # ------------------------------------------------------------------
    #split train files
    # ------------------------------------------------------------------
    logger.info('\n Splitting the train file into {} splits ...'.format(config.num_splits))
    split_train(config)
    logger.info('Saved the train splits in {}'.format('ner/data/'))


if __name__ == "__main__":
    main()