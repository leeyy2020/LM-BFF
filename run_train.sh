#!/bin/bash


task_name=$1

./run_all.sh $task_name &> train.log

mv log output/${task_name}_pet_SCL_log


# cd output
# for i in `ls`;do
#     if [[ "${i}" == step* ]];then
#         tar -czf ${i}.tar.gz ${i}
#         rm -rf ../${i}
#         mv ${i} ../
#     fi
# done
# cd -

#for i in `ls output`;do
#    if [[ "${i}" == score* ]];then
#        echo ${i} >> output/test.log
#        sh metric/run_metric.sh output/${i} ${DATA_PATH}/qtp.test.id >> output/test.log
#    fi
#done
