import logging
import os
import pickle

import nori2 as nori
import numpy as np
import h5py

from megengine.data import DataLoader
from megengine.data.dataset import Dataset
from megengine.data.sampler import RandomSampler, SequentialSampler
import megengine.distributed as dist

from dataset.transformations import fetch_transform
from common import utils

_logger = logging.getLogger(__name__)


class SIGNSDataset(Dataset):
    """
    A standard PyTorch definition of Dataset which defines the functions __len__ and __getitem__.
    """
    def __init__(self, data_dir, split, transform):
        """
        Store the filenames of the jpgs to use. Specifies transforms to apply on images.

        Args:
            data_dir: (string) directory containing the dataset
            transform: (torchvision.transforms) transformation to apply on image
        """
        self.plk_dict = pickle.load(open(os.path.join(data_dir, '{}_signs'.format(split), '{}.pickle'.format(split)), 'rb'))
        self.sources = list(self.plk_dict.keys())
        self.split = split
        self.fetcher = nori.Fetcher()
        self.transform = transform

    def bytes2np(self, data, c=3, h=64, w=64):
        data = np.fromstring(data, np.float32)
        data = data.reshape((h, w, c))
        return data

    def int_from_bytes(self, xbytes: bytes) -> int:
        return int.from_bytes(xbytes, 'big')

    def __len__(self):
        # return size of dataset
        return len(self.sources)

    def __getitem__(self, idx):
        """
        Fetch index idx image and labels from dataset. Perform transforms on image.

        Args:
            idx: (int) index in [0, 1, ..., size_of_dataset-1]

        Returns:
            image: (Tensor) transformed image
            label: (int) corresponding label of image
        """
        image = self.fetcher.get(self.sources[idx])
        label = self.fetcher.get(self.plk_dict[self.sources[idx]])

        image = self.bytes2np(image)
        label = self.int_from_bytes(label)

        image = np.ascontiguousarray(image, dtype=np.uint8).astype(np.float32)
        image = self.transform(image)

        output = {}
        output["image"] = image
        output["label"] = label
        return output


class ModelNetHdf(Dataset):
    def __init__(self, dataset_path: str, subset: str = "train", categories=None, transform=None):
        """ModelNet40 dataset from PointNet.
        """
        self._logger = logging.getLogger(self.__class__.__name__)
        self._root = dataset_path
        self._is_master = dist.get_rank() == 0

        metadata_fpath = os.path.join(self._root, "{}_files.txt".format(subset))
        utils.master_logger(self._logger, "Loading data from {} for {}".format(metadata_fpath, subset), self._is_master)

        if not os.path.exists(os.path.join(dataset_path)):
            self._download_dataset(dataset_path)

        with open(os.path.join(dataset_path, "shape_names.txt")) as fid:
            self._classes = [l.strip() for l in fid]
            self._category2idx = {e[1]: e[0] for e in enumerate(self._classes)}
            self._idx2category = self._classes

        with open(os.path.join(dataset_path, "{}_files.txt".format(subset))) as fid:
            h5_filelist = [line.strip() for line in fid]
            h5_filelist = [x.replace("data/modelnet40_ply_hdf5_2048/", "") for x in h5_filelist]
            h5_filelist = [os.path.join(self._root, f) for f in h5_filelist]

        if categories is not None:
            categories_idx = [self._category2idx[c] for c in categories]
            utils.master_logger(self._logger, "Categories used: {}.".format(categories_idx), self._is_master)
            self._classes = categories
        else:
            categories_idx = None
            utils.master_logger(self._logger, "Using all categories.", self._is_master)

        self._data, self._labels = self._read_h5_files(h5_filelist, categories_idx)
        self._transform = transform
        utils.master_logger(self._logger, "Loaded {} {} instances.".format(self._data.shape[0], subset), self._is_master)

    @property
    def classes(self):
        return self._classes

    @staticmethod
    def _read_h5_files(fnames, categories):

        all_data = []
        all_labels = []

        for fname in fnames:
            f = h5py.File(fname, mode="r")
            data = np.concatenate([f["data"][:], f["normal"][:]], axis=-1)
            labels = f["label"][:].flatten().astype(np.int64)

            if categories is not None:  # Filter out unwanted categories
                mask = np.isin(labels, categories).flatten()
                data = data[mask, ...]
                labels = labels[mask, ...]

            all_data.append(data)
            all_labels.append(labels)

        all_data = np.concatenate(all_data, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)
        return all_data, all_labels

    @staticmethod
    def _download_dataset(dataset_path: str):
        os.makedirs(dataset_path, exist_ok=True)

        www = "https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip"
        zipfile = os.path.basename(www)
        os.system("wget {}".format(www))
        os.system("unzip {} -d .".format(zipfile))
        os.system("mv {} {}".format(zipfile[:-4], os.path.dirname(dataset_path)))
        os.system("rm {}".format(zipfile))

    def to_category(self, i):
        return self._idx2category[i]

    def __getitem__(self, item):

        sample = {"points": self._data[item, :, :], "label": self._labels[item], "idx": np.array(item, dtype=np.int32)}
        if self._transform:
            sample = self._transform(sample)

        return sample

    def __len__(self):
        return self._data.shape[0]


class ModelNetPCD(Dataset):
    def __init__(self, dataset_path: str, subset: str = "train", categories=None, transform=None):
        """ModelNet40 dataset from PointNet.
        Automatically downloads the dataset if not available

        Args:
            dataset_path (str): Folder containing processed dataset
            subset (str): Dataset subset, either "train" or "test"
            categories (list): Categories to use
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self._logger = logging.getLogger(self.__class__.__name__)
        self._root = dataset_path
        self._subset = subset
        self._fetcher = nori.Fetcher()
        self._is_master = dist.get_rank() == 0

        metadata_fpath = os.path.join(self._root, "{}_files.pickle".format(subset))
        utils.master_logger(self._logger, "Loading data from {} for {}".format(metadata_fpath, subset), self._is_master)
        # self._logger.info("Loading data from {} for {}".format(metadata_fpath, subset))

        if not os.path.exists(os.path.join(dataset_path)):
            assert FileNotFoundError("Not found dataset_path: {}".format(dataset_path))

        with open(os.path.join(dataset_path, "shape_names.txt")) as fid:
            self._classes = [l.strip() for l in fid]
            self._category2idx = {e[1]: e[0] for e in enumerate(self._classes)}
            self._idx2category = self._classes

        if categories is not None:
            categories_idx = [self._category2idx[c] for c in categories]
            utils.master_logger(self._logger, "Categories used: {}.".format(categories_idx), self._is_master)
            # self._logger.info("Categories used: {}.".format(categories_idx))
            self._classes = categories
        else:
            categories_idx = None
            utils.master_logger(self._logger, "Using all categories.", self._is_master)
            # self._logger.info("Using all categories.")

        self._data = self._read_pickle_files(os.path.join(dataset_path, "{}_files.pickle".format(subset)), categories_idx)

        # self._data, self._labels = self._data[:32], self._labels[:32, ...]
        self._transform = transform
        utils.master_logger(self._logger, "Loaded {} {} instances.".format(len(self._data), subset), self._is_master)
        # self._logger.info("Loaded {} {} instances.".format(len(self._data), subset))

    @property
    def classes(self):
        return self._classes

    @staticmethod
    def _read_pickle_files(fnames, categories):

        all_data_dict = []
        with open(fnames, "rb") as f:
            data = pickle.load(f)

        for category in categories:
            all_data_dict.extend(data[category])

        return all_data_dict

    def to_category(self, i):
        return self._idx2category[i]

    def __getitem__(self, item):

        data_dict = self._data[item]

        # load and process data
        nid = data_dict["nid"]
        points_bytes = self._fetcher.get(nid)
        points = np.reshape(np.frombuffer(points_bytes, dtype=np.float32), (-1, 2048, 3))
        label = np.array(data_dict["label"]).astype(np.int64)
        sample = {"points": points, "label": label, "idx": np.array(item, dtype=np.int32)}

        if self._transform:
            sample = self._transform(sample)
        return sample

    def __len__(self):
        return len(self._data)


class ModelNetNpy(Dataset):
    def __init__(self, dataset_path: str, dataset_mode: str, subset: str = "train", categories=None, transform=None):
        """ModelNet40 TS data.
        """
        self._logger = logging.getLogger(self.__class__.__name__)
        self._root = dataset_path
        self._subset = subset
        self._fetcher = nori.Fetcher()
        self._is_master = dist.get_rank() == 0

        metadata_fpath = os.path.join(self._root, "modelnet_{}_{}.pickle".format(dataset_mode, subset))
        utils.master_logger(self._logger, "Loading data from {} for {}".format(metadata_fpath, subset), self._is_master)

        if not os.path.exists(os.path.join(dataset_path)):
            assert FileNotFoundError("Not found dataset_path: {}".format(dataset_path))

        with open(os.path.join(dataset_path, "shape_names.txt")) as fid:
            self._classes = [l.strip() for l in fid]
            self._category2idx = {e[1]: e[0] for e in enumerate(self._classes)}
            self._idx2category = self._classes

        if categories is not None:
            categories_idx = [self._category2idx[c] for c in categories]
            utils.master_logger(self._logger, "Categories used: {}.".format(categories_idx), self._is_master)
            self._classes = categories
        else:
            categories_idx = None
            utils.master_logger(self._logger, "Using all categories.", self._is_master)

        self._data = self._read_pickle_files(os.path.join(dataset_path, "modelnet_{}_{}.pickle".format(dataset_mode, subset)),
                                             categories_idx)

        self._transform = transform
        utils.master_logger(self._logger, "Loaded {} {} instances.".format(len(self._data), subset), self._is_master)

    @property
    def classes(self):
        return self._classes

    @staticmethod
    def _read_pickle_files(fnames, categories):

        all_data_dict = []
        with open(fnames, "rb") as f:
            data = pickle.load(f)

        for category in categories:
            all_data_dict.extend(data[category])

        return all_data_dict

    def to_category(self, i):
        return self._idx2category[i]

    def __getitem__(self, item):

        data_path = self._data[item]

        # load and process data
        points = np.load(data_path)
        idx = np.array(int(os.path.splitext(os.path.basename(data_path))[0].split("_")[1]))
        label = np.array(int(os.path.splitext(os.path.basename(data_path))[0].split("_")[3]))
        sample = {"points": points, "label": label, "idx": idx}

        if self._transform:
            sample = self._transform(sample)
        return sample

    def __len__(self):
        return len(self._data)


def fetch_dataloader(params):
    """
    Fetches the DataLoader object for each type in types from data_dir.

    Args:
        types: (list) has one or more of 'train', 'val', 'test' depending on which data is required
        status_manager: (class) status_manager

    Returns:
        data: (dict) contains the DataLoader object for each type in types
    """
    utils.master_logger(_logger, "Dataset type: {}, transform type: {}".format(params.dataset_type, params.transform_type),
                        dist.get_rank() == 0)
    # _logger.info("Dataset type: {}, transform type: {}".format(params.dataset_type, params.transform_type))
    # more transforms can be found at:
    # https://megengine.org.cn/doc/stable/zh/getting-started/beginner/neural-network-traning-tricks.html#%E6%95%B0%E6%8D%AE%E5%A2%9E%E5%B9%BF
    train_transforms, test_transforms = fetch_transform(params)

    if params.dataset_type == "modelnet_os":
        dataset_path = "/data/code/meg_brain_mine/dataset/data/modelnet_os"
        train_categories = [line.rstrip("\n") for line in open("./dataset/data/modelnet40_half1_rm_rotate.txt")]
        val_categories = [line.rstrip("\n") for line in open("./dataset/data/modelnet40_half1_rm_rotate.txt")]
        test_categories = [line.rstrip("\n") for line in open("./dataset/data/modelnet40_half2_rm_rotate.txt")]
        train_categories.sort()
        val_categories.sort()
        test_categories.sort()
        train_ds = ModelNetNpy(dataset_path, dataset_mode="os", subset="train", categories=train_categories, transform=train_transforms)
        val_ds = ModelNetNpy(dataset_path, dataset_mode="os", subset="val", categories=val_categories, transform=test_transforms)
        test_ds = ModelNetNpy(dataset_path, dataset_mode="os", subset="test", categories=test_categories, transform=test_transforms)

    elif params.dataset_type == "modelnet_ts":
        dataset_path = "/data/code/meg_brain_mine/dataset/data/modelnet_ts"
        train_categories = [line.rstrip("\n") for line in open("./dataset/data/modelnet40_half1_rm_rotate.txt")]
        val_categories = [line.rstrip("\n") for line in open("./dataset/data/modelnet40_half1_rm_rotate.txt")]
        test_categories = [line.rstrip("\n") for line in open("./dataset/data/modelnet40_half2_rm_rotate.txt")]
        train_categories.sort()
        val_categories.sort()
        test_categories.sort()
        train_ds = ModelNetNpy(dataset_path, dataset_mode="ts", subset="train", categories=train_categories, transform=train_transforms)
        val_ds = ModelNetNpy(dataset_path, dataset_mode="ts", subset="val", categories=val_categories, transform=test_transforms)
        test_ds = ModelNetNpy(dataset_path, dataset_mode="ts", subset="test", categories=test_categories, transform=test_transforms)

    dataloaders = {}
    # add defalt train data loader
    train_sampler = RandomSampler(train_ds, batch_size=params.train_batch_size, drop_last=True)
    train_dl = DataLoader(train_ds, train_sampler, num_workers=params.num_workers)
    dataloaders["train"] = train_dl

    # chosse val or test data loader for evaluate
    for split in ["val", "test"]:
        if split in params.eval_type:
            if split == "val":
                val_sampler = SequentialSampler(val_ds, batch_size=params.eval_batch_size)
                dl = DataLoader(val_ds, val_sampler, num_workers=params.num_workers)
            elif split == "test":
                test_sampler = SequentialSampler(test_ds, batch_size=params.eval_batch_size)
                dl = DataLoader(test_ds, test_sampler, num_workers=params.num_workers)
            else:
                raise ValueError("Unknown eval_type in params, should in [val, test]")
            dataloaders[split] = dl
        else:
            dataloaders[split] = None

    return dataloaders
