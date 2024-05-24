LR=3e-4
MU=0.1
TEMP=0.1
DATASET=Amazon
CUDA_VISIBLE_DEVICES=3 nohup python pretrain.py --lr $LR --mu $MU --temperature $TEMP --dataset $DATASET > log-$DATASET-LR$LR-MU$MU-TEMP$TEMP.txt &



LR=3e-4
MU=0.1
TEMP=0.5
DATASET=Douban
CUDA_VISIBLE_DEVICES=3 nohup python pretrain.py --lr $LR --mu $MU --temperature $TEMP --dataset Douban --max_token_length 16 > log_douban-LR$LR-MU$MU-TEMP$TEMP.txt &