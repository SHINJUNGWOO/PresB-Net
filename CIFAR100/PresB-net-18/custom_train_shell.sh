#!/bin/bash

SAVEDIR='./propose_ptr/version_1'
TRAINNAME="custom_train_2stage.py"
DATAPATH="../cifar100"
Epoch=400

#: <<"END"
python -u ${TRAINNAME} \
    --weight_decay 1e-5 \
    --epochs ${Epoch} \
    --data ${DATAPATH} \
    --save ${SAVEDIR}/pre

#END

python -u ${TRAINNAME} \
    --weight_decay 0 \
    --epochs ${Epoch} \
    --save ${SAVEDIR}/post\
    --data ${DATAPATH} \
    --binary_w \
    --pretrained ${SAVEDIR}/pre/model_best.pth.tar

