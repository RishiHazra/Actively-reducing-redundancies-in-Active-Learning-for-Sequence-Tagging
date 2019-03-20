#!/usr/bin/env bash

#python3 build_data.py
#python3 train.py --nepochs 15 --mode 'train' --split -0

#python3 evaluate.py --mode 'train' --split 0 --encode True
#python3 sentence_similarity.py


for i in {1..10}
do
    python3 evaluate.py --mode 'feedback' --split 0 --sample_split $i --num_retrain $i --sample_train $(( 20 * $i ))
    python3 train.py --nepochs 2 --mode 'retrain' --num_retrain $i
done

python3 train.py --nepochs 20 --mode 'train' --periodic True
python3 evaluate.py --mode 'train' --split 0 --encode True
python3 sentence_similarity.py

for i in {11..20}
do
    python3 evaluate.py --mode 'feedback' --split 0 --sample_split $i --num_retrain $i --sample_train $(( 20 * $(($i - 10)) ))
    python3 train.py --nepochs 2 --mode 'retrain' --num_retrain $i
done

python3 train.py --nepochs 20 --mode 'train' --periodic True
python3 evaluate.py --mode 'train' --split 0 --encode True
python3 sentence_similarity.py

for i in {21..30}
do
    python3 evaluate.py --mode 'feedback' --split 0 --sample_split $i --num_retrain $i --sample_train $(( 20 * $(($i - 20)) ))
    python3 train.py --nepochs 2 --mode 'retrain' --num_retrain $i
done

python3 train.py --nepochs 20 --mode 'train' --periodic True
python3 evaluate.py --mode 'train' --split 0 --encode True
python3 sentence_similarity.py


for i in {31..40}
do
    python3 evaluate.py --mode 'feedback' --split 0 --sample_split $i --num_retrain $i --sample_train $(( 20 * $(($i - 30)) ))
    python3 train.py --nepochs 2 --mode 'retrain' --num_retrain $i
done

python3 train.py --nepochs 20 --mode 'train' --periodic True
python3 evaluate.py --mode 'train' --split 0 --encode True
python3 sentence_similarity.py

for i in {41..49}
do
    python3 evaluate.py --mode 'feedback' --split 0 --sample_split $i --num_retrain $i --sample_train $(( 20 * $(($i - 40)) ))
    python3 train.py --nepochs 2 --mode 'retrain' --num_retrain $i
done

python3 train.py --nepochs 20 --mode 'train' --periodic True
#python3 evaluate.py --mode 'train' --split 0 --encode True
#python3 sentence_similarity.py