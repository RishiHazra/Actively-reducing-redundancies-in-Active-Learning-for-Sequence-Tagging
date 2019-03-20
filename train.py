from model.data_utils import CoNLLDataset
from model.ner_model import NERModel
from model.config import Config

def main():
    config = Config()

    #-------------------------------------------------------------------
    # build model
    # ------------------------------------------------------------------
    model = NERModel(config)
    model.build()

    # ------------------------------------------------------------------
    # train mode
    # ------------------------------------------------------------------
    if config.mode == 'train':
        print('\n ... training model ... \n')
        test = CoNLLDataset(config.filename_test, config.processing_word,
                         config.processing_tag, config.max_iter)
        if config.periodic:
            split = CoNLLDataset(config.dummy_train, config.processing_word,
                         config.processing_tag, config.max_iter)
        else:
            split = CoNLLDataset(config.train_split[config.split], config.processing_word,
                         config.processing_tag, config.max_iter)
        model.train(split, test)

    # ------------------------------------------------------------------
    # retrain mode
    # ------------------------------------------------------------------
    if config.mode == 'retrain':
        print('\n ... retraining model ... \n')
        model.restore_session(config.dir_model)
        retrain = CoNLLDataset(config.filename_retrain, config.processing_word,
                           config.processing_tag, config.max_iter)
        test = CoNLLDataset(config.filename_test, config.processing_word,
                       config.processing_tag, config.max_iter)
        model.train(retrain, test)


if __name__ == "__main__":
    main()