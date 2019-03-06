from model.data_utils import CoNLLDataset
from model.ner_model import NERModel
from model.config import Config


def align_data(data):
    """Given dict with lists, creates aligned strings
    Adapted from Assignment 3 of CS224N
    Args:
        data: (dict) data["x"] = ["I", "love", "you"]
              (dict) data["y"] = ["O", "O", "O"]
    Returns:
        data_aligned: (dict) data_align["x"] = "I love you"
                           data_align["y"] = "O O    O  "
    """
    print('data',data)
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



def interactive_shell(model):
    """Creates interactive shell to play with model
    Args:
        model: instance of NERModel
    """
    model.logger.info("""
This is an interactive mode.
To exit, enter 'exit'.
You can enter a sentence like
input> I love Paris""")
    '''
    while True:
        try:
            # for python 2
            sentence = raw_input("input> ")
        except NameError:
            # for python 3
            sentence = input("input> ")
    
        if words_raw == ["exit"]:
            break
    '''


    words_raw = 'A woman is surfing'.strip().split(" ")
    preds = model.predict(words_raw)
    print(preds)

    to_print = align_data({"input": words_raw, "output": preds})

    for key, seq in to_print.items():
        model.logger.info(seq)


def main():
    config = Config()

    # -----------------------------------------------------
    # restore model
    # -----------------------------------------------------
    model = NERModel(config)
    model.build()
    model.restore_session(config.dir_model)

    # -----------------------------------------------------
    # create dataset
    # -----------------------------------------------------
    test  = CoNLLDataset(config.filename_test, config.processing_word,
                         config.processing_tag, config.max_iter)
    dev = CoNLLDataset(config.train_split[config.sample_split], config.processing_word,
                        config.processing_tag, config.max_iter)

    sick = CoNLLDataset(config.filename_sick, config.processing_word,
                        config.processing_scores, config.max_iter)

    # -----------------------------------------------------
    # encode SICK dataset using pretrained NER model
    ##-----------------------------------------------------
    if config.encode:
        model.get_encoded(sick)

    # -----------------------------------------------------
    # determine threshold
    #-----------------------------------------------------
    #determine threshold for active learning
    #threshold = 20
    #model.get_threshold(test, threshold)
    # -----------------------------------------------------

    # -----------------------------------------------------
    # evaluate and interact
    # -----------------------------------------------------
    model.evaluate(test, dev, "test")
    #interactive_shell(model)


if __name__ == "__main__":
    main()