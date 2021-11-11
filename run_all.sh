task_name=$1
# gpu=$2
for seed in 13 21 42 87 100
do
    for bs in 2 4 8 16
    do
        for lr in 1e-5 2e-5 5e-5
        do
            for al in 0.1 0.3 0.5 0.7 0.9
            do
                ALPHA=$al \
                TAG=exp \
                TYPE=finetune \
                TASK=$task_name \
                BS=$bs \
                LR=$lr \
                SEED=$seed \
                MODEL=./roberta-large \
                bash run_experiment.sh
                rm -rf result/
            done
        done
    done
done
# task_name=$1
# for seed in 13
# do
#     for bs in 16
#     do
#         for lr in 1e-5
#         do
#             for al in 0.3
#             do
#                 ALPHA=$al \
#                 TAG=exp \
#                 TYPE=finetune \
#                 TASK=$task_name \
#                 BS=$bs \
#                 LR=$lr \
#                 SEED=$seed \
#                 MODEL=./roberta-large \
#                 bash run_experiment.sh
#                 rm -rf result/
#             done
#         done
#     done
# done
