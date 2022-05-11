# PoseC3D-with-Attention
Introduced 3D CBAM Attention to the PoseC3D in MMAction2

本Git只提供了code，实际要到https://github.com/open-mmlab/mmaction2 这个git页去clone跑，新py文件替换原本文件

配置教程：https://blog.csdn.net/WhiffeYF/article/details/120556253

针对于一个数据集

先在data下 wget pkl file （annotation file）

再train: 在mmaction2下跑./tools/dist_train.sh ./configs/skeleton/posec3d/slowonly_kinetics400_pretrained_r50_u48_120e_ucf101_split1_keypoint.py 2 --validate

再test :在mmaction2下跑 ./tools/dist_test.sh ./configs/skeleton/posec3d/slowonly_kinetics400_pretrained_r50_u48_120e_ucf101_split1_keypoint.py ./work_dirs/posec3d_iclr/slowonly_kinetics400_pretrained_r50_u48_120e_ucf101_split1_keypoint/best_top1_acc_epoch_11.pth 2 --eval top_k_accuracy

路径都是一样的

ntu60的keypoint和limb各自分开都可以用

topkaccuracy全部公用

benchmark是榜单的意思 (paper with code上找)

heatmap visualization直接在jupyter里跑，下载好pickle就行，放到相应路径

ntu60subtrain是train，val是测试

NTU60:
./tools/dist_train.sh ./configs/skeleton/posec3d/slowonly_r50_u48_240e_ntu60_xsub_keypoint.py 4 --validate

 ./tools/dist_test.sh ./configs/skeleton/posec3d/slowonly_r50_u48_240e_ntu60_xsub_keypoint.py ./work_dirs/posec3d/slowonly_r50_u48_240e_ntu60_xsub_keypoint/best_top1_acc_epoch_11.pth 2 --eval top_k_accuracy

 HMDB:
./tools/dist_train.sh ./configs/skeleton/posec3d/slowonly_kinetics400_pretrained_r50_u48_120e_hmdb51_split1_keypoint.py 2 --validate

 ./tools/dist_test.sh ./configs/skeleton/posec3d/slowonly_kinetics400_pretrained_r50_u48_120e_hmdb51_split1_keypoint.py ./work_dirs/posec3d_iclr/slowonly_kinetics400_pretrained_r50_u48_120e_hmdb51_split1_keypoint/best_top1_acc_epoch_12.pth 2 --eval top_k_accuracy

为了保持结构的一致性，我把configs里的ucf和hmdb里面stage_blocks改成了(4,6,3). 总共只有三个stage
