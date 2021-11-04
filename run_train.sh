#!/bin/bash


job_name=$1

./run_all.sh $job_name &> train.log


mv log output/${job_name}_log

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