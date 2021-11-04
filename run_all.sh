task_name=$1
for seed in 13 21 42 87 100
do
    for bs in 2 4 8
    do
        for lr in 1e-5 2e-5 5e-5
        do
            for al in 0.1 0.3 0.5 0.7 1
            do
                ALPHA=$al \
                TAG=exp \
                TYPE=prompt \
                TASK=$task_name \
                BS=$bs \
                LR=$lr \
                SEED=$seed \
                MODEL=roberta-large \
                bash run_experiment.sh
                rm -rf result/
            done
        done
    done
done
