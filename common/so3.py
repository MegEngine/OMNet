import numpy as np
import megengine as mge
from scipy.spatial.transform import Rotation


def np_dcm2euler(mats: np.ndarray, seq: str = "zyx", degrees: bool = True):
    """Converts rotation matrix to euler angles

    Args:
        mats: (B, 3, 3) containing the B rotation matricecs
        seq: Sequence of euler rotations (default: "zyx")
        degrees (bool): If true (default), will return in degrees instead of radians

    Returns:

    """

    eulers = []
    for i in range(mats.shape[0]):
        r = Rotation.from_matrix(mats[i])
        eulers.append(r.as_euler(seq, degrees=degrees))
    return np.stack(eulers)


def np_transform(g: np.ndarray, pts: np.ndarray):
    """ Applies the SO3 transform

    Args:
        g: SO3 transformation matrix of size (B, 3, 3)
        pts: Points to be transformed (B, N, 3)

    Returns:
        transformed points of size (B, N, 3)

    """
    rot = g[..., :3, :3]  # (3, 3)
    transformed = pts[..., :3] @ np.swapaxes(rot, -1, -2)
    return transformed


def np_inverse(g: np.ndarray):
    """Returns the inverse of the SE3 transform

    Args:
        g: ([B,] 3/4, 4) transform

    Returns:
        ([B,] 3/4, 4) matrix containing the inverse

    """
    rot = g[..., :3, :3]  # (3, 3)

    inv_rot = np.swapaxes(rot, -1, -2)

    return inv_rot


def mge_dcm2euler(mats, seq, degrees=True):
    mats = mats.numpy()
    eulers = []
    for i in range(mats.shape[0]):
        r = Rotation.from_matrix(mats[i])
        eulers.append(r.as_euler(seq, degrees=degrees))
    return mge.tensor(np.stack(eulers))
