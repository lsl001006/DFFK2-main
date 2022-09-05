python kd_mosaic.py --gpu 1 \
--pipeline=multi_teacher --teacher wrn40_2 --student resnet8 \
--dataset cifar10 --unlabeled cifar10 \
--epochs 500 \
--ckpt_path /data/repo/code/1sl/DFFK/checkpoints/resnet8/fixLr0.01_500e_n20_b16/ \
--fp16 \
--logfile fixLr0.01_16 \
--resume /data/repo/code/1sl/DFFK/checkpoints/resnet8/fixLr0.01_400e_n20_b16/latest.pth