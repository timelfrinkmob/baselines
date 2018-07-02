#!/bin/bash
for seed in 1 2 3
do
    for i in 10 20 30 40 50 60 70 80 90 100;
    do
	    for exp in "" "--noisy=1" "--greedy=1" "--bootstrap=1";
	    do
            python -m baselines.deepq.experiments.train_chain --n=$i $exp --seed=$seed
        done
    done
done

aws s3 cp ./models/ s3://thesis-tim-files/baseline/chain/ --recursive
sudo shutdown -P now