* Actively-reducing-redundancies-in-Active-Learning-for-Sequence-Tagging
Implementation of [[https://papers.nips.cc/paper/7213-poincare-embeddings-for-learning-hierarchical-representations][Poincaré Embeddings for Learning Hierarchical Representations]] using distributed frameworks such as TensorFlow, PyTorch and HOROVOD.

[[file:pytorch/plots/mammal_closure.tsv_poincare_dim2_e1000.png]]

** Installation
Simply clone this repository via
#+BEGIN_SRC sh
git clone https://github.com/sobalgi/DS222-Poincare-Embedding.git
cd DS222-Poincare-Embedding
#+END_SRC


** Dependencies
- Python 3 with NumPy
- PyTorch
- TensorFlow
- Horovod
- Scikit-Learn
- NLTK (to generate the WordNet data)

** Folders
For TensorFLow implementations
#+BEGIN_SRC sh
cd tensorflow
#+END_SRC

For PyTorch implementations
#+BEGIN_SRC sh
cd pytorch
#+END_SRC

For Numpy implementation
#+BEGIN_SRC sh
cd numpy
#+END_SRC

The steps to reproduce the results of the experiments are given in the respective readme files of the folders.

