#!/bin/bash
wget 10.9.189.4:9876/roberta-large.tgz
tar zxvf roberta-large.tgz


wget 10.9.189.4:9876/python3.7.0.tgz
tar zxvf python3.7.0.tgz 
export https_proxy=http://172.19.56.199:3128 
export http_proxy=http://172.19.56.199:3128

./python3.7.0/bin/python3.7 -m pip install torch==1.7.0+cu110 -f https://download.pytorch.org/whl/torch_stable.html &> install.log  

wget 10.9.189.4:9876/data.tgz
tar zxvf data.tgz 
rm -f data.tgz


