# OMNet: Learning Overlapping Mask for Partial-to-Partial Point Cloud Registration
Code for ICCV2021 paper [OMNet](https://openaccess.thecvf.com/content/ICCV2021/papers/Xu_OMNet_Learning_Overlapping_Mask_for_Partial-to-Partial_Point_Cloud_Registration_ICCV_2021_paper.pdf)
The illustration of our pipeline:
![image](https://user-images.githubusercontent.com/46584121/137711441-62672d6b-f5e5-4946-86de-fecdb9c6b42a.png)
## Dependencies
MegEngine == 1.6.0
## Training and evaluation
### Data preparation
You need to download our preprocessed ModelNet40 dataset first, where 8 axisymmetrical categories are removed and all CAD models have 40 randomly sampled point clouds. The download link is [data.zip]().
### Begin training
For ModelNet40 dataset, you can just use:
```
python3 train.py --model_dir=./experiments/train_demo
```
For other dataset, you need to add your own dataset class in `./dataset/data_loader.py`.
### Begin testing
You need to download the pretrained checkpoint and use:
```
python3 evaluate.py --model_dir=./experiments/test_demo --restore_file=./experiments/test_demo/val_best_model.pth --only_weights
```
## Pretrained model
MegEngine checkpoint for ModelNet40 dataset can be download via [Google Drive]() or [Github Release]().
