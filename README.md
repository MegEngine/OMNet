# OMNet: Learning Overlapping Mask for Partial-to-Partial Point Cloud Registration

MegEngine implementation of our ICCV2021 paper [OMNet](https://openaccess.thecvf.com/content/ICCV2021/papers/Xu_OMNet_Learning_Overlapping_Mask_for_Partial-to-Partial_Point_Cloud_Registration_ICCV_2021_paper.pdf).

The illustration of our pipeline:
![image](https://user-images.githubusercontent.com/46584121/137711441-62672d6b-f5e5-4946-86de-fecdb9c6b42a.png)

## Dependencies

Main requirements:

* python==3.6.9
* MegEngine==1.6.0

Other requirements please refer to `requirements.txt`.

## Data preparation

### OS data

We refer the original data from PointNet as OS data, where point clouds are only sampled once from corresponding CAD models. We offer two ways to use OS data, (1) you can download this data from its original link [original_OS_data.zip](http://modelnet.cs.princeton.edu/). (2) you can also download the data that has been preprocessed by us from link [our_OS_data.zip](https://drive.google.com/file/d/1rXnbXwD72tkeu8x6wboMP0X7iL9LiBPq/view?usp=sharing).

### TS data

Since OS data incurs over-fitting issue, we propose our TS data, where point clouds are randomly sampled twice from CAD models. You need to download our preprocessed ModelNet40 dataset first, where 8 axisymmetrical categories are removed and all CAD models have 40 randomly sampled point clouds. The download link is [TS_data.zip](https://drive.google.com/file/d/1-zcp5oR69WM6lMI71uHmOi6OFpzQomgI/view?usp=sharing). All 40 point clouds of a CAD model are stacked to form a (40, 2048, 3) numpy array, you can easily obtain this data by using following code:

```
import numpy as np
points = np.load("path_of_npy_file")
print(points.shape, type(points))  # (40, 2048, 3), <class 'numpy.ndarray'>
```

Then, you need to put the data into `./dataset/data`, and the contents of directories are as follows:

```
./dataset/data/
├── modelnet40_half1_rm_rotate.txt
├── modelnet40_half2_rm_rotate.txt
├── modelnet_os
│   ├── modelnet_os_test.pickle
│   ├── modelnet_os_train.pickle
│   ├── modelnet_os_val.pickle
│   ├── test [1146 entries exceeds filelimit, not opening dir]
│   ├── train [4194 entries exceeds filelimit, not opening dir]
│   └── val [1002 entries exceeds filelimit, not opening dir]
└── modelnet_ts
    ├── modelnet_ts_test.pickle
    ├── modelnet_ts_train.pickle
    ├── modelnet_ts_val.pickle
    ├── shape_names.txt
    ├── test [1146 entries exceeds filelimit, not opening dir]
    ├── train [4196 entries exceeds filelimit, not opening dir]
    └── val [1002 entries exceeds filelimit, not opening dir]
```

## Training and evaluation

### Begin training

For ModelNet40 dataset, you can just run:

```
python3 train.py --model_dir=./experiments/experiment_omnet/
```

For other dataset, you need to add your own dataset class in `./dataset/data_loader.py`. Training with lower batch size, such as 16, may obtain worse performance.

### Begin testing

You need to download the pretrained checkpoint and run:

```
python3 evaluate.py --model_dir=./experiments/experiment_omnet --restore_file=./experiments/experiment_omnet/model_best.pth --only_weights
```

This model weight is for TS data with Gaussian noise. Note that the performance is a little bit worse than our Pytorch imlementation.

## Pretrained model

MegEngine checkpoint for ModelNet40 dataset can be download via [Google Drive]() or [Github Release]().

## Citation

```
@InProceedings{Xu_2021_ICCV,
    author={Xu, Hao and Liu, Shuaicheng and Wang, Guangfu and Liu, Guanghui and Zeng, Bing},
    title={OMNet: Learning Overlapping Mask for Partial-to-Partial Point Cloud Registration},
    booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    month={October},
    year={2021},
    pages={3132-3141}
}
```