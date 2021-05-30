# Active^2 Learning: Actively-reducing-redundancies-in-Active-Learning-for-Sequence-Tagging

## Installation
Simply clone this repository via
```
$ git clone https://github.com/RishiHazra/Actively-reducing-redundancies-in-Active-Learning-for-Sequence-Tagging.git
```

## Dependencies
* Python 3 with NumPy
* tensorflow
* Scikit-Learn
* matplotlib
* nltk
* gensim (for GENIA dataset)

## To run the code
```
$ vi run.sh  (set the desired model , data split, npeochs)
$ ./run.sh
```

## Publication
* Rishi Hazra, Parag Dutta, Shubham Gupta, Mohammed Abdul Qaathir, Ambedkar Dukkipati, In **NAACL 2021**. [Active^2 Learning: Actively reducing redundancies in Active Learning methods for Sequence Tagging and Machine Translation](https://www.aclweb.org/anthology/2021.naacl-main.159/)

```
@inproceedings{hazra-etal-2021-active2,
    title = "Active$^2$ Learning: Actively reducing redundancies in Active Learning methods for Sequence Tagging and Machine Translation",
    author = "Hazra, Rishi  and
      Dutta, Parag  and
      Gupta, Shubham  and
      Qaathir, Mohammed Abdul  and
      Dukkipati, Ambedkar",
    booktitle = "Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies",
    month = jun,
    year = "2021",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/2021.naacl-main.159",
    pages = "1982--1995",
    abstract = "While deep learning is a powerful tool for natural language processing (NLP) problems, successful solutions to these problems rely heavily on large amounts of annotated samples. However, manually annotating data is expensive and time-consuming. Active Learning (AL) strategies reduce the need for huge volumes of labeled data by iteratively selecting a small number of examples for manual annotation based on their estimated utility in training the given model. In this paper, we argue that since AL strategies choose examples independently, they may potentially select similar examples, all of which may not contribute significantly to the learning process. Our proposed approach, Active$\mathbf{^2}$ Learning (A$\mathbf{^2}$L), actively adapts to the deep learning model being trained to eliminate such redundant examples chosen by an AL strategy. We show that A$\mathbf{^2}$L is widely applicable by using it in conjunction with several different AL strategies and NLP tasks. We empirically demonstrate that the proposed approach is further able to reduce the data requirements of state-of-the-art AL strategies by $\approx \mathbf{3-25\%}$ on an absolute scale on multiple NLP tasks while achieving the same performance with virtually no additional computation overhead.",
}
```
