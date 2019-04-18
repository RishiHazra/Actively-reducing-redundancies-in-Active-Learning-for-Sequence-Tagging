#!/usr/bin/env bash

declare -a arr=("siamese" "cosine" "skipthoughts")

for sim in "${arr[@]}"
do
    python build_data.py --split 45
    python train.py --nepochs 16 --mode 'train' --split 45 --excel_id 1 --model 'CNN BILSTM CRF'
    for k in 0 10 20 30 40 0 10 20 30 40 0 10 20 30 40
    do
        for i in 0 1 2 3 4 5 6 7 8 9
        do
            if [ "$(( $k + $i ))" -ne "50" ] && [ "$(( $k + $i ))" -ne "45" ]
            then
                python evaluate.py --mode 'feedback' --split 45 --sample_split $(( $k + $i )) --num_retrain $(( $k + $i )) --sample_train $(( 20 * $i )) --active_strategy 'entropy' --similarity $sim --threshold 40 --model 'CNN BILSTM CRF'
                python train.py --nepochs 2 --mode 'retrain' --num_retrain $(( $k + $i )) --model 'CNN BILSTM CRF'
            fi
        done
    python train.py --nepochs $(($(( $k / 10 )) + 16 )) --mode 'train' --split 45 --periodic True --excel_id $(($(($k / 10))+2)) --model 'CNN BILSTM CRF'
    #python evaluate.py --mode 'train' --split 45 --encode True
    #python sentence_similarity.py
    done
    rm num_fedback.npy
    rm ner/newSamples.txt
    rm ner/dummy_train.txt
    rm ner/retrain.txt
done