# [ICCV 2021] OMNet: Learning Overlapping Mask for Partial-to-Partial Point Cloud Registration

This is the official implementation (MegEngine implementation) of our ICCV2021 paper [OMNet](https://openaccess.thecvf.com/content/ICCV2021/papers/Xu_OMNet_Learning_Overlapping_Mask_for_Partial-to-Partial_Point_Cloud_Registration_ICCV_2021_paper.pdf). For our Pytorch implementation, please refer to [this repo](https://github.com/hxwork/OMNet_Pytorch).

Our presentation video: [[Youtube](https://www.youtube.com/watch?v=u2lTKsom8oU)][[Bilibili](https://www.bilibili.com/video/BV1Ef4y1J7XP/)].

## Our Poster

![image](./images/OMNet_poster.png)

## Dependencies

* MegEngine==1.6.0
* Other requirements please refer to`requirements.txt`.
* Add`frequency_weighted_cross_entropy` to MegEngine source code.

MegEngine==1.6.0 does not support `frequency_weighted_cross_entropy`, so I write this function based on `cross_entropy` in `loss.py` of the original MegEngine source code, whose location should be like this:

(1) conda environment

`[your_conda_env_path]/lib/[python3.x]/site-packages/megengine/functional/loss.py`.

(2) original python environment

`/usr/local/lib/[python3.x]/dist-packages/megengine/functional/loss.py`.

Use your own path to replace the content in `[]`.

```
from .math import sum


@_reduce_output
def frequency_weighted_cross_entropy(
    pred: Tensor,
    label: Tensor,
    weight: Tensor = None,
    axis: int = 1,
    with_logits: bool = True,
    label_smooth: float = 0,
    reduction: str = "mean",
) -> Tensor:

    n0 = pred.ndim
    n1 = label.ndim
    assert n0 == n1 + 1, ("target ndim must be one less than input ndim; input_ndim={} " "target_ndim={}".format(n0, n1))

    if weight is not None:
        weight = weight / sum(weight)
        class_weight = weight[label.flatten().astype(np.int32)].reshape(label.shape)

    ls = label_smooth

    if with_logits:
        logZ = logsumexp(pred, axis)
        primary_term = indexing_one_hot(pred, label, axis)
    else:
        logZ = 0
        primary_term = log(indexing_one_hot(pred, label, axis))
    if ls is None or type(ls) in (int, float) and ls == 0:
        if weight is None:
            return logZ - primary_term
        else:
            return sum((logZ - primary_term) * class_weight, axis=1, keepdims=True) / sum(class_weight, axis=1, keepdims=True)
    if not with_logits:
        pred = log(pred)
    if weight is None:
        return logZ - ls * pred.mean(axis) - (1 - ls) * primary_term
    else:
        return sum((logZ - ls * pred.mean(axis) -
                    (1 - ls) * primary_term) * class_weight, axis=1, keepdims=True) / sum(class_weight, axis=1, keepdims=True)
```

## Data Preparation

### OS data

We refer the original data from PointNet as OS data, where point clouds are only sampled once from corresponding CAD models. We offer two ways to use OS data, (1) you can download this data from its original link [original_OS_data.zip](http://modelnet.cs.princeton.edu/). (2) you can also download the data that has been preprocessed by us from link [our_OS_data.zip](https://drive.google.com/file/d/1rXnbXwD72tkeu8x6wboMP0X7iL9LiBPq/view?usp=sharing).

### TS data

Since OS data incurs over-fitting issue, we propose our TS data, where point clouds are randomly sampled twice from CAD models. You need to download our preprocessed ModelNet40 dataset first, where 8 axisymmetrical categories are removed and all CAD models have 40 randomly sampled point clouds. The download link is [TS_data.zip](https://drive.google.com/file/d/1DPBBI3Ulvp2Mx7SAZaBEyvADJzBvErFF/view?usp=sharing). All 40 point clouds of a CAD model are stacked to form a (40, 2048, 3) numpy array, you can easily obtain this data by using following code:

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

## Training and Evaluation

### Begin training

For ModelNet40 dataset, you can just run:

```
python3 train.py --model_dir=./experiments/experiment_omnet/
```

For other dataset, you need to add your own dataset class in `./dataset/data_loader.py`. Training with a lower batch size, such as 16, may obtain worse performance than training with a larger batch size, e.g., 64.

### Begin testing

You need to download the pretrained checkpoint and run:

```
python3 evaluate.py --model_dir=./experiments/experiment_omnet --restore_file=./experiments/experiment_omnet/val_model_best.pth
```

This model weight is for TS data with Gaussian noise. Note that the performance is a little bit worse than the results reported in our paper (Pytorch implementation).

MegEngine checkpoint for ModelNet40 dataset can be download via [Google Drive](https://drive.google.com/file/d/1xkWQeMabQhO4zqg6X3aj_VQCMHgeBUsD/view?usp=sharing) or [Github Release](https://github.com/megvii-research/OMNet/releases/download/V1.0.0/val_model_best.pth).

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

## Acknowledgments

In this project we use (parts of) the official implementations of the following works:

* [RPMNet](https://github.com/yewzijian/RPMNet) (ModelNet40 preprocessing and evaluation)
* [PRNet](https://github.com/WangYueFt/prnet) (ModelNet40 preprocessing)

We thank the respective authors for open sourcing their methods.
