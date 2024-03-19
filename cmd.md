
conda activate bev_kxk

训练指令
CUDA_VISIBLE_DEVICES=2,3,4,5 torchpack dist-run -np 4 python tools/train.py configs/groupmix-qkv-C.yaml

--load_from  继续训练
--resume_from 恢复训练

消除显存卡死
fuser -v /dev/nvidia* |awk '{for(i=1;i<=NF;i++)print "kill -9 " $i;}' | sh


