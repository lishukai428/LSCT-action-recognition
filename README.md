# LSCT-action-recognition
action recognition This repository implements action recognition network LSCT.

Requirements Python3 PyTorch 1.0 or higher scikit-learn tqdm

Data Preparation:

We need to first extract videos into frames for fast reading. Please refer to TSN repo for the detailed guide of data pre-processing.

Basically, the processing of video data can be summarized into 3 steps:

Extract frames from videos (refer to tools/vid2img_sthv2.py for Something-Something-V2 example) Generate annotations needed for dataloader (refer to tools/gen_label_sthv1.py for Something-Something-V1 example, and tools/gen_label_sthv2.py for Something-Something-V2 example) Add the information to ops/dataset_configs.py

Training: python main.py something RGB
--arch resnet50 --num_segments 8
--gd 20 --lr 0.01 --lr_steps 20 40 45 --epochs 60
--batch-size 48 -j 16 --dropout 0.5 --consensus_type=avg --eval-freq=1
--shift --shift_div=8 --shift_place=blockres --npb

testing: python test_models.py something
--weights=checkpoint/TSM_something_RGB_resnet50_shift8_blockres_avg_segment8_e45.pth
--test_segments=8 --batch_size=32 -j 24 --test_crops=1
