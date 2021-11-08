import logging
import math
import megengine as mge
import megengine.distributed as dist
import numpy as np
from common import se3, so3, utils
from scipy.spatial.transform import Rotation
from megengine.data.transform import Transform

_logger = logging.getLogger(__name__)


def uniform_2_sphere(num: int = None):
    """Uniform sampling on a 2-sphere

    Source: https://gist.github.com/andrewbolster/10274979

    Args:
        num: Number of vectors to sample (or None if single)

    Returns:
        Random Vector (np.ndarray) of size (num, 3) with norm 1.
        If num is None returned value will have size (3,)

    """
    if num is not None:
        phi = np.random.uniform(0.0, 2 * np.pi, num)
        cos_theta = np.random.uniform(-1.0, 1.0, num)
    else:
        phi = np.random.uniform(0.0, 2 * np.pi)
        cos_theta = np.random.uniform(-1.0, 1.0)

    theta = np.arccos(cos_theta)
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)

    return np.stack((x, y, z), axis=-1)


class SplitSourceRef(Transform):
    """Clones the point cloud into separate source and reference point clouds"""
    def __init__(self, mode="os"):
        self.mode = mode

    def apply(self, sample):
        if "deterministic" in sample and sample["deterministic"]:
            np.random.seed(sample["idx"])

        if self.mode == "os":
            sample["points_raw"] = sample.pop("points").astype(np.float32)[:, :3]
            sample["points_src"] = sample["points_raw"].copy()
            sample["points_ref"] = sample["points_raw"].copy()
            sample["points_src_raw"] = sample["points_src"].copy().astype(np.float32)
            sample["points_ref_raw"] = sample["points_ref"].copy().astype(np.float32)
        elif self.mode == "ts":
            points_raw = sample.pop("points").astype(np.float32)
            points_raw = points_raw[np.random.choice(points_raw.shape[0], 2, replace=False), :, :]
            sample["points_src"] = points_raw[0, :, :].astype(np.float32)
            sample["points_ref"] = points_raw[1, :, :].astype(np.float32)
            sample["points_src_raw"] = sample["points_src"].copy()
            sample["points_ref_raw"] = sample["points_ref"].copy()

        else:
            raise NotImplementedError

        return sample


class Resampler(Transform):
    def __init__(self, num: int):
        """Resamples a point cloud containing N points to one containing M

        Guaranteed to have no repeated points if M <= N.
        Otherwise, it is guaranteed that all points appear at least once.

        Args:
            num (int): Number of points to resample to, i.e. M

        """
        self.num = num

    @staticmethod
    def _resample(points, k):
        """Resamples the points such that there is exactly k points.

        If the input point cloud has <= k points, it is guaranteed the
        resampled point cloud contains every point in the input.
        If the input point cloud has > k points, it is guaranteed the
        resampled point cloud does not contain repeated point.
        """
        # print("===", points.shape[0], k)
        if k < points.shape[0]:
            rand_idxs = np.random.choice(points.shape[0], k, replace=False)
            return points[rand_idxs, :]
        elif points.shape[0] == k:
            return points
        else:
            rand_idxs = np.concatenate([
                np.random.choice(points.shape[0], points.shape[0], replace=False),
                np.random.choice(points.shape[0], k - points.shape[0], replace=True)
            ])
            return points[rand_idxs, :]

    def apply(self, sample):

        if "deterministic" in sample and sample["deterministic"]:
            np.random.seed(sample["idx"])

        if "points" in sample:
            sample["points"] = self._resample(sample["points"], self.num)
        else:
            if "crop_proportion" not in sample:
                src_size, ref_size = self.num, self.num
            elif len(sample["crop_proportion"]) == 1:
                src_size = math.ceil(sample["crop_proportion"][0] * self.num)
                ref_size = self.num
            elif len(sample["crop_proportion"]) == 2:
                src_size = math.ceil(sample["crop_proportion"][0] * self.num)
                ref_size = math.ceil(sample["crop_proportion"][1] * self.num)
            else:
                raise ValueError("Crop proportion must have 1 or 2 elements")

            sample["points_src"] = self._resample(sample["points_src"], src_size)
            sample["points_ref"] = self._resample(sample["points_ref"], ref_size)

            # sample for the raw point clouds
            sample["points_src_raw"] = sample["points_src_raw"][:self.num, :]
            sample["points_ref_raw"] = sample["points_ref_raw"][:self.num, :]

        return sample


class RandomJitter(Transform):
    """ generate perturbations """
    def __init__(self, noise_std=0.01, clip=0.05):
        self.noise_std = noise_std
        self.clip = clip

    def jitter(self, pts):

        noise = np.clip(np.random.normal(0.0, scale=self.noise_std, size=(pts.shape[0], 3)), a_min=-self.clip, a_max=self.clip)
        pts[:, :3] += noise  # Add noise to xyz

        return pts

    def apply(self, sample):

        if "points" in sample:
            sample["points"] = self.jitter(sample["points"])
        else:
            sample["points_src"] = self.jitter(sample["points_src"])
            sample["points_ref"] = self.jitter(sample["points_ref"])

        return sample


class RandomCrop(Transform):
    """Randomly crops the *source* point cloud, approximately retaining half the points

    A direction is randomly sampled from S2, and we retain points which lie within the
    half-space oriented in this direction.
    If p_keep != 0.5, we shift the plane until approximately p_keep points are retained
    """
    def __init__(self, p_keep=None):
        if p_keep is None:
            p_keep = [0.7, 0.7]  # Crop both clouds to 70%
        self.p_keep = np.array(p_keep, dtype=np.float32)

    @staticmethod
    def crop(points, p_keep):
        if p_keep == 1.0:
            mask = np.ones(shape=(points.shape[0], )) > 0

        else:
            rand_xyz = uniform_2_sphere()
            centroid = np.mean(points[:, :3], axis=0)
            points_centered = points[:, :3] - centroid
            dist_from_plane = np.dot(points_centered, rand_xyz)

            if p_keep == 0.5:
                mask = dist_from_plane > 0
            else:
                mask = dist_from_plane > np.percentile(dist_from_plane, (1.0 - p_keep) * 100)

        return points[mask, :]

    def apply(self, sample):

        if "deterministic" in sample and sample["deterministic"]:
            np.random.seed(sample["idx"])

        sample["crop_proportion"] = self.p_keep

        if len(sample["crop_proportion"]) == 1:
            sample["points_src"] = self.crop(sample["points_src"], self.p_keep[0])
            sample["points_ref"] = self.crop(sample["points_ref"], 1.0)
        else:
            sample["points_src"] = self.crop(sample["points_src"], self.p_keep[0])
            sample["points_ref"] = self.crop(sample["points_ref"], self.p_keep[1])

        return sample


class RandomTransformSE3(Transform):
    def __init__(self, rot_mag: float = 180.0, trans_mag: float = 1.0, random_mag: bool = False):
        """Applies a random rigid transformation to the source point cloud

        Args:
            rot_mag (float): Maximum rotation in degrees
            trans_mag (float): Maximum translation T. Random translation will
              be in the range [-X,X] in each axis
            random_mag (bool): If true, will randomize the maximum rotation, i.e. will bias towards small
                               perturbations
        """
        self._rot_mag = rot_mag
        self._trans_mag = trans_mag
        self._random_mag = random_mag

    def generate_transform(self):
        """Generate a random SE3 transformation (3, 4) """

        if self._random_mag:
            attentuation = np.random.random()
            rot_mag, trans_mag = attentuation * self._rot_mag, attentuation * self._trans_mag
        else:
            rot_mag, trans_mag = self._rot_mag, self._trans_mag

        # Generate rotation
        rand_rot = special_ortho_group.rvs(3)
        axis_angle = Rotation.as_rotvec(Rotation.from_dcm(rand_rot))
        axis_angle *= rot_mag / 180.0
        rand_rot = Rotation.from_rotvec(axis_angle).as_dcm()

        # Generate translation
        rand_trans = np.random.uniform(-trans_mag, trans_mag, 3)
        rand_SE3 = np.concatenate((rand_rot, rand_trans[:, None]), axis=1).astype(np.float32)

        return rand_SE3

    def apply_transform(self, p0, transform_mat):
        p1 = se3.np_transform(transform_mat, p0[:, :3])
        if p0.shape[1] == 6:  # Need to rotate normals also
            n1 = so3.transform(transform_mat[:3, :3], p0[:, 3:6])
            p1 = np.concatenate((p1, n1), axis=-1)

        igt = transform_mat
        gt = se3.np_inverse(igt)

        return p1, gt, igt

    def transform(self, tensor):
        transform_mat = self.generate_transform()
        return self.apply_transform(tensor, transform_mat)

    def apply(self, sample):

        if "deterministic" in sample and sample["deterministic"]:
            np.random.seed(sample["idx"])

        if "points" in sample:
            sample["points"], _, _ = self.transform(sample["points"])
        else:
            src_transformed, transform_r_s, transform_s_r = self.transform(sample["points_src"])
            # Apply to source to get reference
            sample["transform_gt"] = transform_r_s
            sample["pose_gt"] = se3.np_mat2quat(transform_r_s)
            sample["transform_igt"] = transform_s_r
            sample["points_src"] = src_transformed
            # transnform the raw source point cloud
            sample["points_src_raw"] = se3.np_transform(transform_s_r, sample["points_src_raw"][:, :3])

        return sample


# noinspection PyPep8Naming
class RandomTransformSE3_euler(RandomTransformSE3):
    """Same as RandomTransformSE3, but rotates using euler angle rotations

    This transformation is consistent to Deep Closest Point but does not
    generate uniform rotations

    """
    def generate_transform(self):

        if self._random_mag:
            attentuation = np.random.random()
            rot_mag, trans_mag = attentuation * self._rot_mag, attentuation * self._trans_mag
        else:
            rot_mag, trans_mag = self._rot_mag, self._trans_mag

        # Generate rotation
        anglex = np.random.uniform() * np.pi * rot_mag / 180.0
        angley = np.random.uniform() * np.pi * rot_mag / 180.0
        anglez = np.random.uniform() * np.pi * rot_mag / 180.0

        cosx = np.cos(anglex)
        cosy = np.cos(angley)
        cosz = np.cos(anglez)
        sinx = np.sin(anglex)
        siny = np.sin(angley)
        sinz = np.sin(anglez)
        Rx = np.array([[1, 0, 0], [0, cosx, -sinx], [0, sinx, cosx]])
        Ry = np.array([[cosy, 0, siny], [0, 1, 0], [-siny, 0, cosy]])
        Rz = np.array([[cosz, -sinz, 0], [sinz, cosz, 0], [0, 0, 1]])
        R_ab = Rx @ Ry @ Rz
        t_ab = np.random.uniform(-trans_mag, trans_mag, 3)

        rand_SE3 = np.concatenate((R_ab, t_ab[:, None]), axis=1).astype(np.float32)
        return rand_SE3


class ShufflePoints(Transform):
    """Shuffles the order of the points"""
    def apply(self, sample):
        if "points" in sample:
            sample["points"] = np.random.permutation(sample["points"])
        else:
            sample["points_ref"] = np.random.permutation(sample["points_ref"])
            sample["points_src"] = np.random.permutation(sample["points_src"])
        return sample


class SetDeterministic(Transform):
    """Adds a deterministic flag to the sample such that subsequent transforms
    use a fixed random seed where applicable. Used for test"""
    def apply(self, sample):
        sample["deterministic"] = True
        return sample


class PRNet(Transform):
    def __init__(self, num_points, rot_mag, trans_mag, noise_std=0.01, clip=0.05, add_noise=True, only_z=False, partial=True):
        self.num_points = num_points
        self.rot_mag = rot_mag
        self.trans_mag = trans_mag
        self.noise_std = noise_std
        self.clip = clip
        self.add_noise = add_noise
        self.only_z = only_z
        self.partial = partial

    def apply_transform(self, p0, transform_mat):
        p1 = se3.np_transform(transform_mat, p0[:, :3])
        if p0.shape[1] == 6:  # Need to rotate normals also
            n1 = so3.transform(transform_mat[:3, :3], p0[:, 3:6])
            p1 = np.concatenate((p1, n1), axis=-1)

        gt = transform_mat

        return p1, gt

    def jitter(self, pts):
        noise = np.clip(np.random.normal(0.0, scale=self.noise_std, size=(pts.shape[0], 3)), a_min=-self.clip, a_max=self.clip)
        pts[:, :3] += noise  # Add noise to xyz

        return pts

    def knn(self, pts, random_pt, k):
        distance = np.sum((pts - random_pt)**2, axis=1)
        idx = np.argsort(distance)[:k]  # (k,)
        return idx

    def apply(self, sample):

        if "deterministic" in sample and sample["deterministic"]:
            np.random.seed(sample["idx"])

        src = sample["points_src"]
        ref = sample["points_ref"]
        # Generate rigid transform
        anglex = np.random.uniform() * np.pi * self.rot_mag / 180.0
        angley = np.random.uniform() * np.pi * self.rot_mag / 180.0
        anglez = np.random.uniform() * np.pi * self.rot_mag / 180.0

        cosx = np.cos(anglex)
        cosy = np.cos(angley)
        cosz = np.cos(anglez)
        sinx = np.sin(anglex)
        siny = np.sin(angley)
        sinz = np.sin(anglez)
        Rx = np.array([[1, 0, 0], [0, cosx, -sinx], [0, sinx, cosx]])
        Ry = np.array([[cosy, 0, siny], [0, 1, 0], [-siny, 0, cosy]])
        Rz = np.array([[cosz, -sinz, 0], [sinz, cosz, 0], [0, 0, 1]])

        if not self.only_z:
            R_ab = Rx @ Ry @ Rz
        else:
            R_ab = Rz
        t_ab = np.random.uniform(-self.trans_mag, self.trans_mag, 3)

        rand_SE3 = np.concatenate((R_ab, t_ab[:, None]), axis=1).astype(np.float32)
        ref, transform_s_r = self.apply_transform(ref, rand_SE3)
        # Apply to source to get reference
        sample["transform_gt"] = transform_s_r
        sample["pose_gt"] = se3.np_mat2quat(transform_s_r)

        # Crop and sample
        if self.partial:
            random_p1 = np.random.random(size=(1, 3)) + np.array([[500, 500, 500]]) * np.random.choice([1, -1, 1, -1])
            idx1 = self.knn(src, random_p1, k=768)
            random_p2 = random_p1
            idx2 = self.knn(ref, random_p2, k=768)
        else:
            idx1 = np.random.choice(src.shape[0], 1024, replace=False),
            idx2 = np.random.choice(ref.shape[0], 1024, replace=False),
            src = mge.tensor(src)
            ref = mge.tensor(ref)

        # add noise
        if self.add_noise:
            sample["points_src"] = self.jitter(src[idx1, :])
            sample["points_ref"] = self.jitter(ref[idx2, :])
        else:
            sample["points_src"] = src[idx1, :]
            sample["points_ref"] = ref[idx2, :]

        return sample


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, input):
        for t in self.transforms:
            input = t.apply(input)
        return input


def fetch_transform(params):

    if params.transform_type == "modelnet_os_rpmnet_noise":
        train_transforms = [
            SplitSourceRef(mode="os"),
            RandomCrop(params.partial_ratio),
            RandomTransformSE3_euler(rot_mag=params.rot_mag, trans_mag=params.trans_mag),
            Resampler(params.num_points),
            RandomJitter(),
            ShufflePoints()
        ]

        test_transforms = [
            SetDeterministic(),
            SplitSourceRef(mode="os"),
            RandomCrop(params.partial_ratio),
            RandomTransformSE3_euler(rot_mag=params.rot_mag, trans_mag=params.trans_mag),
            Resampler(params.num_points),
            RandomJitter(),
            ShufflePoints()
        ]

    elif params.transform_type == "modelnet_os_rpmnet_clean":
        train_transforms = [
            SplitSourceRef(mode="os"),
            RandomCrop(params.partial_ratio),
            RandomTransformSE3_euler(rot_mag=params.rot_mag, trans_mag=params.trans_mag),
            Resampler(params.num_points),
            ShufflePoints()
        ]

        test_transforms = [
            SetDeterministic(),
            SplitSourceRef(mode="os"),
            RandomCrop(params.partial_ratio),
            RandomTransformSE3_euler(rot_mag=params.rot_mag, trans_mag=params.trans_mag),
            Resampler(params.num_points),
            ShufflePoints()
        ]

    elif params.transform_type == "modelnet_ts_rpmnet_noise":
        train_transforms = [
            SplitSourceRef(mode="ts"),
            RandomCrop(params.partial_ratio),
            RandomTransformSE3_euler(rot_mag=params.rot_mag, trans_mag=params.trans_mag),
            Resampler(params.num_points),
            RandomJitter(noise_std=params.noise_std),
            ShufflePoints()
        ]

        test_transforms = [
            SetDeterministic(),
            SplitSourceRef(mode="ts"),
            RandomCrop(params.partial_ratio),
            RandomTransformSE3_euler(rot_mag=params.rot_mag, trans_mag=params.trans_mag),
            Resampler(params.num_points),
            RandomJitter(noise_std=params.noise_std),
            ShufflePoints()
        ]

    elif params.transform_type == "modelnet_ts_rpmnet_clean":
        train_transforms = [
            SplitSourceRef(mode="ts"),
            RandomCrop(params.partial_ratio),
            RandomTransformSE3_euler(rot_mag=params.rot_mag, trans_mag=params.trans_mag),
            Resampler(params.num_points),
            ShufflePoints()
        ]

        test_transforms = [
            SetDeterministic(),
            SplitSourceRef(mode="ts"),
            RandomCrop(params.partial_ratio),
            RandomTransformSE3_euler(rot_mag=params.rot_mag, trans_mag=params.trans_mag),
            Resampler(params.num_points),
            ShufflePoints()
        ]

    elif params.transform_type == "modelnet_ts_prnet_noise":
        train_transforms = [
            SplitSourceRef(mode="ts"),
            ShufflePoints(),
            PRNet(num_points=params.num_points,
                  rot_mag=params.rot_mag,
                  trans_mag=params.trans_mag,
                  noise_std=params.noise_std,
                  add_noise=True)
        ]

        test_transforms = [
            SetDeterministic(),
            SplitSourceRef(mode="ts"),
            ShufflePoints(),
            PRNet(num_points=params.num_points,
                  rot_mag=params.rot_mag,
                  trans_mag=params.trans_mag,
                  noise_std=params.noise_std,
                  add_noise=True)
        ]

    elif params.transform_type == "modelnet_ts_prnet_clean":
        train_transforms = [
            SplitSourceRef(mode="ts"),
            ShufflePoints(),
            PRNet(num_points=params.num_points, rot_mag=params.rot_mag, trans_mag=params.trans_mag, add_noise=False)
        ]

        test_transforms = [
            SetDeterministic(),
            SplitSourceRef(mode="ts"),
            ShufflePoints(),
            PRNet(num_points=params.num_points, rot_mag=params.rot_mag, trans_mag=params.trans_mag, add_noise=False)
        ]

    elif params.transform_type == "modelnet_os_prnet_noise":
        train_transforms = [
            SplitSourceRef(mode="os"),
            ShufflePoints(),
            PRNet(num_points=params.num_points, rot_mag=params.rot_mag, trans_mag=params.trans_mag, add_noise=True)
        ]

        test_transforms = [
            SetDeterministic(),
            SplitSourceRef(mode="os"),
            ShufflePoints(),
            PRNet(num_points=params.num_points, rot_mag=params.rot_mag, trans_mag=params.trans_mag, add_noise=True)
        ]

    elif params.transform_type == "modelnet_os_prnet_clean":
        train_transforms = [
            SplitSourceRef(mode="os"),
            ShufflePoints(),
            PRNet(num_points=params.num_points, rot_mag=params.rot_mag, trans_mag=params.trans_mag, add_noise=False)
        ]

        test_transforms = [
            SetDeterministic(),
            SplitSourceRef(mode="os"),
            ShufflePoints(),
            PRNet(num_points=params.num_points, rot_mag=params.rot_mag, trans_mag=params.trans_mag, add_noise=False)
        ]

    utils.master_logger(_logger, "Train transforms: {}".format(", ".join([type(t).__name__ for t in train_transforms])),
                        dist.get_rank() == 0)
    utils.master_logger(_logger, "Val and Test transforms: {}".format(", ".join([type(t).__name__ for t in test_transforms])),
                        dist.get_rank() == 0)
    train_transforms = Compose(train_transforms)
    test_transforms = Compose(test_transforms)
    return train_transforms, test_transforms
