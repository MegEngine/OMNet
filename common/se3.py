import numpy as np
import megengine as mge
import megengine.functional as F
import torch
# import transforms3d.euler as t3de
import transforms3d.quaternions as t3d
from scipy.spatial.transform import Rotation


def torch_inverse(g):
    """ Returns the inverse of the SE3 transform

    Args:
        g: (B, 3/4, 4) transform

    Returns:
        (B, 3, 4) matrix containing the inverse

    """
    # Compute inverse
    rot = g[..., 0:3, 0:3]
    trans = g[..., 0:3, 3]
    inverse_transform = torch.cat([rot.transpose(-1, -2), rot.transpose(-1, -2) @ -trans[..., None]], dim=-1)

    return inverse_transform


def torch_concatenate(a, b):
    """Concatenate two SE3 transforms,
    i.e. return a@b (but note that our SE3 is represented as a 3x4 matrix)

    Args:
        a: (B, 3/4, 4)
        b: (B, 3/4, 4)

    Returns:
        (B, 3/4, 4)
    """

    rot1 = a[..., :3, :3]
    trans1 = a[..., :3, 3]
    rot2 = b[..., :3, :3]
    trans2 = b[..., :3, 3]

    rot_cat = rot1 @ rot2
    trans_cat = rot1 @ trans2[..., None] + trans1[..., None]
    concatenated = torch.cat([rot_cat, trans_cat], dim=-1)

    return concatenated


def torch_transform(g, a, normals=None):
    """ Applies the SE3 transform

    Args:
        g: SE3 transformation matrix of size ([1,] 3/4, 4) or (B, 3/4, 4)
        a: Points to be transformed (N, 3) or (B, N, 3)
        normals: (Optional). If provided, normals will be transformed

    Returns:
        transformed points of size (N, 3) or (B, N, 3)

    """
    R = g[..., :3, :3]  # (B, 3, 3)
    p = g[..., :3, 3]  # (B, 3)

    if len(g.size()) == len(a.size()):
        b = torch.matmul(a, R.transpose(-1, -2)) + p[..., None, :]
    else:
        raise NotImplementedError
        b = R.matmul(a.unsqueeze(-1)).squeeze(-1) + p  # No batch. Not checked

    if normals is not None:
        rotated_normals = normals @ R.transpose(-1, -2)
        return b, rotated_normals

    else:
        return b


def torch_mat2quat(M):
    all_pose = []
    for i in range(M.size()[0]):
        rotate = M[i, :3, :3]
        translate = M[i, :3, 3]

        # Qyx refers to the contribution of the y input vector component to
        # the x output vector component.  Qyx is therefore the same as
        # M[0,1].  The notation is from the Wikipedia article.
        Qxx, Qyx, Qzx, Qxy, Qyy, Qzy, Qxz, Qyz, Qzz = rotate.flatten()
        #     print(Qxx, Qyx, Qzx, Qxy, Qyy, Qzy, Qxz, Qyz, Qzz)
        # Fill only lower half of symmetric matrix
        K = torch.tensor([[Qxx - Qyy - Qzz, 0, 0, 0], [Qyx + Qxy, Qyy - Qxx - Qzz, 0, 0], [Qzx + Qxz, Qzy + Qyz, Qzz - Qxx - Qyy, 0],
                          [Qyz - Qzy, Qzx - Qxz, Qxy - Qyx, Qxx + Qyy + Qzz]]) / 3.0
        # Use Hermitian eigenvectors, values for speed
        vals, vecs = torch.symeig(K, True, False)
        # Select largest eigenvector, reorder to w,x,y,z quaternion

        q = vecs[[3, 0, 1, 2], torch.argmax(vals)]
        # Prefer quaternion with positive w
        # (q * -1 corresponds to same rotation as q)
        if q[0] < 0:
            q *= -1

        pose = torch.cat((q, translate), dim=0)
        all_pose.append(pose)
    all_pose = torch.stack(all_pose, dim=0)
    return all_pose  # (B, 7)


def mge_inverse(g):
    """ Returns the inverse of the SE3 transform

    Args:
        g: (B, 3/4, 4) transform

    Returns:
        (B, 3, 4) matrix containing the inverse

    """
    # Compute inverse
    rot = g[..., 0:3, 0:3]
    trans = g[..., 0:3, 3]
    inverse_transform = F.concat([rot.transpose(0, 2, 1), F.matmul(rot.transpose(0, 2, 1), F.expand_dims(-trans, axis=-1))], axis=-1)

    return inverse_transform


def mge_concatenate(a, b):
    """Concatenate two SE3 transforms,
    i.e. return a@b (but note that our SE3 is represented as a 3x4 matrix)

    Args:
        a: (B, 3/4, 4)
        b: (B, 3/4, 4)

    Returns:
        (B, 3/4, 4)
    """

    rot1 = a[..., :3, :3]
    trans1 = a[..., :3, 3]
    rot2 = b[..., :3, :3]
    trans2 = b[..., :3, 3]

    rot_cat = F.matmul(rot1, rot2)
    trans_cat = F.matmul(rot1, F.expand_dims(trans2, axis=-1)) + F.expand_dims(trans1, axis=-1)
    concatenated = F.concat([rot_cat, trans_cat], axis=-1)

    return concatenated


def mge_transform(g, a, normals=None):
    """ Applies the SE3 transform

    Args:
        g: SE3 transformation matrix of size ([1,] 3/4, 4) or (B, 3/4, 4)
        a: Points to be transformed (N, 3) or (B, N, 3)
        normals: (Optional). If provided, normals will be transformed

    Returns:
        transformed points of size (N, 3) or (B, N, 3)

    """
    R = g[..., :3, :3]  # (B, 3, 3)
    p = g[..., :3, 3]  # (B, 3)

    if len(g.shape) == len(a.shape):
        b = F.matmul(a, R.transpose(0, 2, 1)) + F.expand_dims(p, axis=1)
    else:
        raise NotImplementedError

    if normals is not None:
        rotated_normals = F.matmul(normals, R.transpose(0, 2, 1))
        return b, rotated_normals

    else:
        return b


def np_identity():
    return np.eye(3, 4)


def np_transform(g: np.ndarray, pts: np.ndarray):
    """ Applies the SE3 transform

    Args:
        g: SE3 transformation matrix of size ([B,] 3/4, 4)
        pts: Points to be transformed ([B,] N, 3)

    Returns:
        transformed points of size (N, 3)
    """
    rot = g[..., :3, :3]  # (3, 3)
    trans = g[..., :3, 3]  # (3)

    transformed = pts[..., :3] @ np.swapaxes(rot, -1, -2) + trans[..., None, :]
    return transformed


def np_inverse(g: np.ndarray):
    """Returns the inverse of the SE3 transform

    Args:
        g: ([B,] 3/4, 4) transform

    Returns:
        ([B,] 3/4, 4) matrix containing the inverse

    """
    rot = g[..., :3, :3]  # (3, 3)
    trans = g[..., :3, 3]  # (3)

    inv_rot = np.swapaxes(rot, -1, -2)
    inverse_transform = np.concatenate([inv_rot, inv_rot @ -trans[..., None]], axis=-1)
    if g.shape[-2] == 4:
        inverse_transform = np.concatenate([inverse_transform, [[0.0, 0.0, 0.0, 1.0]]], axis=-2)

    return inverse_transform


def np_concatenate(a: np.ndarray, b: np.ndarray):
    """ Concatenate two SE3 transforms

    Args:
        a: First transform ([B,] 3/4, 4)
        b: Second transform ([B,] 3/4, 4)

    Returns:
        a*b ([B, ] 3/4, 4)

    """

    r_a, t_a = a[..., :3, :3], a[..., :3, 3]
    r_b, t_b = b[..., :3, :3], b[..., :3, 3]

    r_ab = r_a @ r_b
    t_ab = r_a @ t_b[..., None] + t_a[..., None]

    concatenated = np.concatenate([r_ab, t_ab], axis=-1)

    if a.shape[-2] == 4:
        concatenated = np.concatenate([concatenated, [[0.0, 0.0, 0.0, 1.0]]], axis=-2)

    return concatenated


def np_from_xyzquat(xyzquat):
    """Constructs SE3 matrix from x, y, z, qx, qy, qz, qw

    Args:
        xyzquat: np.array (7,) containing translation and quaterion

    Returns:
        SE3 matrix (4, 4)
    """
    rot = Rotation.from_quat(xyzquat[3:])
    trans = rot.apply(-xyzquat[:3])
    transform = np.concatenate([rot.as_dcm(), trans[:, None]], axis=1)
    transform = np.concatenate([transform, [[0.0, 0.0, 0.0, 1.0]]], axis=0)

    return transform


def np_mat2quat(transform):
    rotate = transform[:3, :3]
    translate = transform[:3, 3]
    quat = t3d.mat2quat(rotate)
    pose = np.concatenate([quat, translate], axis=0)
    return pose  # (7, )


def np_quat2mat(pose):
    # Separate each quaternion value.
    q0, q1, q2, q3 = pose[:, 0], pose[:, 1], pose[:, 2], pose[:, 3]
    # Convert quaternion to rotation matrix.
    # Ref: 	http://www-evasion.inrialpes.fr/people/Franck.Hetroy/Teaching/ProjetsImage/2007/Bib/besl_mckay-pami1992.pdf
    # A method for Registration of 3D shapes paper by Paul J. Besl and Neil D McKay.
    R11 = q0 * q0 + q1 * q1 - q2 * q2 - q3 * q3
    R12 = 2 * (q1 * q2 - q0 * q3)
    R13 = 2 * (q1 * q3 + q0 * q2)
    R21 = 2 * (q1 * q2 + q0 * q3)
    R22 = q0 * q0 + q2 * q2 - q1 * q1 - q3 * q3
    R23 = 2 * (q2 * q3 - q0 * q1)
    R31 = 2 * (q1 * q3 - q0 * q2)
    R32 = 2 * (q2 * q3 + q0 * q1)
    R33 = q0 * q0 + q3 * q3 - q1 * q1 - q2 * q2
    R = np.stack((np.stack((R11, R12, R13), axis=0), np.stack((R21, R22, R23), axis=0), np.stack((R31, R32, R33), axis=0)), axis=0)

    rot_mat = R.transpose((2, 0, 1))  # (B, 3, 3)
    translation = pose[:, 4:][:, :, None]  # (B, 3, 1)
    transform = np.concatenate((rot_mat, translation), axis=2)
    return transform  # (B, 3, 4)


if __name__ == "__main__":
    anglex = np.random.uniform() * np.pi * 2
    angley = np.random.uniform() * np.pi * 2
    anglez = np.random.uniform() * np.pi * 2
    cosx = np.cos(anglex)
    cosy = np.cos(angley)
    cosz = np.cos(anglez)
    sinx = np.sin(anglex)
    siny = np.sin(angley)
    sinz = np.sin(anglez)
    Rx = np.array([[1, 0, 0], [0, cosx, -sinx], [0, sinx, cosx]])
    Ry = np.array([[cosy, 0, siny], [0, 1, 0], [-siny, 0, cosy]])
    Rz = np.array([[cosz, -sinz, 0], [sinz, cosz, 0], [0, 0, 1]])
    R = (Rx @ Ry @ Rz)[None, ...]  # (1, 3, 3)
    t = np.random.uniform(-0.5, 0.5, 3)[None, :, None]  # (1, 3, 1)
    trans = np.concatenate((R, t), axis=2)  # (1, 3, 4)
    pc = np.random.randn(1, 5, 3)
    torch_trans = torch.from_numpy(trans)
    torch_pc = torch.from_numpy(pc)
    mge_trans = mge.tensor(trans)
    mge_pc = mge.tensor(pc)

    ################### [torch_inverse] vs [mge_inverse] ################### pass
    torch_trans_inv = torch_inverse(torch_trans)
    mge_trans_inv = mge_inverse(mge_trans)
    print(torch_trans_inv)
    print(mge_trans_inv)

    ################### [torch_concatenate] vs [mge_concatenate] ################### pass
    torch_trans_concat = torch_concatenate(torch_trans, torch_trans)
    mge_trans_concat = mge_concatenate(mge_trans, mge_trans)
    print(torch_trans_concat)
    print(mge_trans_concat)

    ################### [torch_transform] vs [mge_transform] ################### pass
    torch_pc_trans = torch_transform(torch_trans, torch_pc)
    mge_pc_trans = mge_transform(mge_trans, mge_pc)
    print(torch_pc_trans)
    print(mge_pc_trans)
