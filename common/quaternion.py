import torch
import numpy as np
import megengine as mge
import megengine.functional as F


def torch_qmul(q1, q2):
    """
    Multiply quaternion(s) q2q1, rotate q1 first, rotate q2 second.
    Expects two equally-sized tensors of shape (*, 4), where * denotes any number of dimensions.
    Returns q*r as a tensor of shape (*, 4).
    """
    assert q1.shape[-1] == 4
    assert q2.shape[-1] == 4

    original_shape = q1.shape

    # Compute outer product
    terms = torch.bmm(q1.view(-1, 4, 1), q2.view(-1, 1, 4))

    w = terms[:, 0, 0] - terms[:, 1, 1] - terms[:, 2, 2] - terms[:, 3, 3]
    x = terms[:, 0, 1] + terms[:, 1, 0] - terms[:, 2, 3] + terms[:, 3, 2]
    y = terms[:, 0, 2] + terms[:, 1, 3] + terms[:, 2, 0] - terms[:, 3, 1]
    z = terms[:, 0, 3] - terms[:, 1, 2] + terms[:, 2, 1] + terms[:, 3, 0]
    return torch.stack((w, x, y, z), dim=1).view(original_shape)


def torch_qrot(q, v):
    """
    Rotate vector(s) v about the rotation described by quaternion(s) q.
    Expects a tensor of shape (*, 4) for q and a tensor of shape (*, 3) for v,
    where * denotes any number of dimensions.
    Returns a tensor of shape (*, 3).
    """
    assert q.shape[-1] == 4
    assert v.shape[-1] == 3
    assert q.shape[:-1] == v.shape[:-1]

    original_shape = list(v.shape)
    q = q.view(-1, 4)
    v = v.view(-1, 3)

    qvec = q[:, 1:]
    uv = torch.cross(qvec, v, dim=1)
    uuv = torch.cross(qvec, uv, dim=1)
    return (v + 2 * (q[:, :1] * uv + uuv)).view(original_shape)


def torch_quat2euler(q, order, epsilon=0):
    """
    Convert quaternion(s) q to Euler angles.
    Expects a tensor of shape (*, 4), where * denotes any number of dimensions.
    Returns a tensor of shape (*, 3).
    """
    assert q.shape[-1] == 4

    original_shape = list(q.shape)
    original_shape[-1] = 3
    q = q.view(-1, 4)

    q0 = q[:, 0]
    q1 = q[:, 1]
    q2 = q[:, 2]
    q3 = q[:, 3]

    if order == "xyz":
        x = torch.atan2(2 * (q0 * q1 - q2 * q3), 1 - 2 * (q1 * q1 + q2 * q2))
        y = torch.asin(torch.clamp(2 * (q1 * q3 + q0 * q2), -1 + epsilon, 1 - epsilon))
        z = torch.atan2(2 * (q0 * q3 - q1 * q2), 1 - 2 * (q2 * q2 + q3 * q3))
    elif order == "yzx":
        x = torch.atan2(2 * (q0 * q1 - q2 * q3), 1 - 2 * (q1 * q1 + q3 * q3))
        y = torch.atan2(2 * (q0 * q2 - q1 * q3), 1 - 2 * (q2 * q2 + q3 * q3))
        z = torch.asin(torch.clamp(2 * (q1 * q2 + q0 * q3), -1 + epsilon, 1 - epsilon))
    elif order == "zxy":
        x = torch.asin(torch.clamp(2 * (q0 * q1 + q2 * q3), -1 + epsilon, 1 - epsilon))
        y = torch.atan2(2 * (q0 * q2 - q1 * q3), 1 - 2 * (q1 * q1 + q2 * q2))
        z = torch.atan2(2 * (q0 * q3 - q1 * q2), 1 - 2 * (q1 * q1 + q3 * q3))
    elif order == "xzy":
        x = torch.atan2(2 * (q0 * q1 + q2 * q3), 1 - 2 * (q1 * q1 + q3 * q3))
        y = torch.atan2(2 * (q0 * q2 + q1 * q3), 1 - 2 * (q2 * q2 + q3 * q3))
        z = torch.asin(torch.clamp(2 * (q0 * q3 - q1 * q2), -1 + epsilon, 1 - epsilon))
    elif order == "yxz":
        x = torch.asin(torch.clamp(2 * (q0 * q1 - q2 * q3), -1 + epsilon, 1 - epsilon))
        y = torch.atan2(2 * (q1 * q3 + q0 * q2), 1 - 2 * (q1 * q1 + q2 * q2))
        z = torch.atan2(2 * (q1 * q2 + q0 * q3), 1 - 2 * (q1 * q1 + q3 * q3))
    elif order == "zyx":
        x = torch.atan2(2 * (q0 * q1 + q2 * q3), 1 - 2 * (q1 * q1 + q2 * q2))
        y = torch.asin(torch.clamp(2 * (q0 * q2 - q1 * q3), -1 + epsilon, 1 - epsilon))
        z = torch.atan2(2 * (q0 * q3 + q1 * q2), 1 - 2 * (q2 * q2 + q3 * q3))
    else:
        raise

    return torch.stack((x, y, z), dim=1).view(original_shape)


def torch_euler2quat(e, order):
    """
    Convert Euler angles to quaternions.
    """
    assert e.size()[-1] == 3

    original_shape = [e.size()[0], 4]

    x = e[:, 0]
    y = e[:, 1]
    z = e[:, 2]

    rx = torch.stack((torch.cos(x / 2), torch.sin(x / 2), torch.zeros_like(x).cuda(), torch.zeros_like(x).cuda()), dim=1)
    ry = torch.stack((torch.cos(y / 2), torch.zeros_like(y).cuda(), torch.sin(y / 2), torch.zeros_like(y).cuda()), dim=1)
    rz = torch.stack((torch.cos(z / 2), torch.zeros_like(z).cuda(), torch.zeros_like(z).cuda(), torch.sin(z / 2)), dim=1)

    result = None
    for coord in order:
        if coord == "x":
            r = rx
        elif coord == "y":
            r = ry
        elif coord == "z":
            r = rz
        else:
            raise
        if result is None:
            result = r
        else:
            result = torch_qmul(result, r)

    # Reverse antipodal representation to have a non-negative "w"
    if order in ["xyz", "yzx", "zxy"]:
        result *= -1

    return result.reshape(original_shape)


def torch_quat2mat(pose):
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
    R = torch.stack((torch.stack((R11, R12, R13), dim=0), torch.stack((R21, R22, R23), dim=0), torch.stack((R31, R32, R33), dim=0)), dim=0)

    rot_mat = R.permute((2, 0, 1))  # (B, 3, 3)
    translation = pose[:, 4:].unsqueeze(2)  # (B, 3, 1)
    transform = torch.cat((rot_mat, translation), dim=2)
    return transform  # (B, 3, 4)


def torch_transform_pose(pose_old, pose_new):
    quat_old, translate_old = pose_old[:, :4], pose_old[:, 4:]
    quat_new, translate_new = pose_new[:, :4], pose_new[:, 4:]

    quat = torch_qmul(quat_old, quat_new)
    translate = torch_qrot(quat_new, translate_old) + translate_new
    pose = torch.cat((quat, translate), dim=1)

    return pose


def torch_qinv(q):
    # expectes q in (w,x,y,z) format
    w = q[:, 0:1]
    v = q[:, 1:]
    inv = torch.cat([w, -v], dim=1)
    return inv


def torch_quat_rotate(point_cloud: torch.Tensor, pose_7d: torch.Tensor):
    ndim = point_cloud.dim()
    if ndim == 2:
        N, _ = point_cloud.shape
        assert pose_7d.shape[0] == 1
        # repeat transformation vector for each point in shape
        quat = pose_7d[:, 0:4].expand([N, -1])
        rotated_point_cloud = torch_qrot(quat, point_cloud)

    elif ndim == 3:
        B, N, _ = point_cloud.shape
        quat = pose_7d[:, 0:4].unsqueeze(1).expand([-1, N, -1]).contiguous()
        rotated_point_cloud = torch_qrot(quat, point_cloud)

    else:
        raise RuntimeError("point cloud dim must be 2 or 3 !")

    return rotated_point_cloud


def torch_quat_transform(pose_7d: torch.Tensor, pc: torch.Tensor, normal: torch.Tensor = None):
    pc_t = torch_quat_rotate(pc, pose_7d) + pose_7d[:, 4:].view(-1, 1, 3).repeat(1, pc.shape[1], 1)  # Ps" = R*Ps + t
    if normal is not None:
        normal_t = torch_quat_rotate(normal, pose_7d)
        return pc_t, normal_t
    else:
        return pc_t


def mge_qmul(q1, q2):
    """
    Multiply quaternion(s) q2q1, rotate q1 first, rotate q2 second.
    Expects two equally-sized tensors of shape (*, 4), where * denotes any number of dimensions.
    Returns q*r as a tensor of shape (*, 4).
    """
    assert q1.shape[-1] == 4
    assert q2.shape[-1] == 4

    original_shape = q1.shape

    # Compute outer product
    terms = F.matmul(q1.reshape(-1, 4, 1), q2.reshape(-1, 1, 4))

    w = terms[:, 0, 0] - terms[:, 1, 1] - terms[:, 2, 2] - terms[:, 3, 3]
    x = terms[:, 0, 1] + terms[:, 1, 0] - terms[:, 2, 3] + terms[:, 3, 2]
    y = terms[:, 0, 2] + terms[:, 1, 3] + terms[:, 2, 0] - terms[:, 3, 1]
    z = terms[:, 0, 3] - terms[:, 1, 2] + terms[:, 2, 1] + terms[:, 3, 0]
    return F.stack((w, x, y, z), axis=1).reshape(original_shape)


def mge_qrot(q, v):
    """
    Rotate vector(s) v about the rotation described by quaternion(s) q.
    Expects a tensor of shape (*, 4) for q and a tensor of shape (*, 3) for v,
    where * denotes any number of dimensions.
    Returns a tensor of shape (*, 3).
    """
    assert q.shape[-1] == 4
    assert v.shape[-1] == 3
    assert q.shape[:-1] == v.shape[:-1]

    original_shape = list(v.shape)
    q = q.reshape(-1, 4)
    v = v.reshape(-1, 3)

    qvec = q[:, 1:]
    uv = F.stack((
        qvec[:, 1] * v[:, 2] - qvec[:, 2] * v[:, 1],
        qvec[:, 2] * v[:, 0] - qvec[:, 0] * v[:, 2],
        qvec[:, 0] * v[:, 1] - qvec[:, 1] * v[:, 0],
    ),
                 axis=1)
    uuv = F.stack((
        qvec[:, 1] * uv[:, 2] - qvec[:, 2] * uv[:, 1],
        qvec[:, 2] * uv[:, 0] - qvec[:, 0] * uv[:, 2],
        qvec[:, 0] * uv[:, 1] - qvec[:, 1] * uv[:, 0],
    ),
                  axis=1)
    # uv = F.cross(qvec, v, dim=1)
    # uuv = F.cross(qvec, uv, dim=1)
    return (v + 2 * (q[:, :1] * uv + uuv)).reshape(original_shape)


# TODO: check
def mge_quat2euler(q, order, epsilon=0):
    """
    Convert quaternion(s) q to Euler angles.
    Expects a tensor of shape (*, 4), where * denotes any number of dimensions.
    Returns a tensor of shape (*, 3).
    """
    assert q.shape[-1] == 4

    original_shape = list(q.shape)
    original_shape[-1] = 3
    q = q.reshape(-1, 4)

    q0 = q[:, 0]
    q1 = q[:, 1]
    q2 = q[:, 2]
    q3 = q[:, 3]

    if order == "xyz":
        x = F.atan2(2 * (q0 * q1 - q2 * q3), 1 - 2 * (q1 * q1 + q2 * q2))
        y = F.asin(F.clip(2 * (q1 * q3 + q0 * q2), -1 + epsilon, 1 - epsilon))
        z = F.atan2(2 * (q0 * q3 - q1 * q2), 1 - 2 * (q2 * q2 + q3 * q3))
    elif order == "yzx":
        x = F.atan2(2 * (q0 * q1 - q2 * q3), 1 - 2 * (q1 * q1 + q3 * q3))
        y = F.atan2(2 * (q0 * q2 - q1 * q3), 1 - 2 * (q2 * q2 + q3 * q3))
        z = F.asin(F.clip(2 * (q1 * q2 + q0 * q3), -1 + epsilon, 1 - epsilon))
    elif order == "zxy":
        x = F.asin(F.clip(2 * (q0 * q1 + q2 * q3), -1 + epsilon, 1 - epsilon))
        y = F.atan2(2 * (q0 * q2 - q1 * q3), 1 - 2 * (q1 * q1 + q2 * q2))
        z = F.atan2(2 * (q0 * q3 - q1 * q2), 1 - 2 * (q1 * q1 + q3 * q3))
    elif order == "xzy":
        x = F.atan2(2 * (q0 * q1 + q2 * q3), 1 - 2 * (q1 * q1 + q3 * q3))
        y = F.atan2(2 * (q0 * q2 + q1 * q3), 1 - 2 * (q2 * q2 + q3 * q3))
        z = F.asin(F.clip(2 * (q0 * q3 - q1 * q2), -1 + epsilon, 1 - epsilon))
    elif order == "yxz":
        x = F.asin(F.clip(2 * (q0 * q1 - q2 * q3), -1 + epsilon, 1 - epsilon))
        y = F.atan2(2 * (q1 * q3 + q0 * q2), 1 - 2 * (q1 * q1 + q2 * q2))
        z = F.atan2(2 * (q1 * q2 + q0 * q3), 1 - 2 * (q1 * q1 + q3 * q3))
    elif order == "zyx":
        x = F.atan2(2 * (q0 * q1 + q2 * q3), 1 - 2 * (q1 * q1 + q2 * q2))
        y = F.asin(F.clip(2 * (q0 * q2 - q1 * q3), -1 + epsilon, 1 - epsilon))
        z = F.atan2(2 * (q0 * q3 + q1 * q2), 1 - 2 * (q2 * q2 + q3 * q3))
    else:
        raise

    return F.stack((x, y, z), axis=1).reshape(original_shape)


# TODO: check
def mge_euler2quat(e, order):
    """
    Convert Euler angles to quaternions.
    """
    assert e.shape[-1] == 3

    original_shape = [e.shape[0], 4]

    x = e[:, 0]
    y = e[:, 1]
    z = e[:, 2]

    rx = F.stack((F.cos(x / 2), F.sin(x / 2), F.zeros_like(x).cuda(), F.zeros_like(x).cuda()), axis=1)
    ry = F.stack((F.cos(y / 2), F.zeros_like(y).cuda(), F.sin(y / 2), F.zeros_like(y).cuda()), axis=1)
    rz = F.stack((F.cos(z / 2), F.zeros_like(z).cuda(), F.zeros_like(z).cuda(), F.sin(z / 2)), axis=1)

    result = None
    for coord in order:
        if coord == "x":
            r = rx
        elif coord == "y":
            r = ry
        elif coord == "z":
            r = rz
        else:
            raise
        if result is None:
            result = r
        else:
            result = mge_qmul(result, r)

    # Reverse antipodal representation to have a non-negative "w"
    if order in ["xyz", "yzx", "zxy"]:
        result *= -1

    return result.reshape(original_shape)


def mge_quat2mat(pose):
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
    R = F.stack((F.stack((R11, R12, R13), axis=0), F.stack((R21, R22, R23), axis=0), F.stack((R31, R32, R33), axis=0)), axis=0)

    rot_mat = F.transpose(R, (2, 0, 1))  # (B, 3, 3)
    translation = F.expand_dims(pose[:, 4:], axis=-1)  # (B, 3, 1)
    transform = F.concat((rot_mat, translation), axis=2)
    return transform  # (B, 3, 4)


def mge_transform_pose(pose_old, pose_new):
    quat_old, translate_old = pose_old[:, :4], pose_old[:, 4:]
    quat_new, translate_new = pose_new[:, :4], pose_new[:, 4:]

    quat = mge_qmul(quat_old, quat_new)
    translate = mge_qrot(quat_new, translate_old) + translate_new
    pose = F.concat((quat, translate), axis=1)

    return pose


# TODO: check
def mge_qinv(q):
    # expectes q in (w,x,y,z) format
    w = q[:, 0:1]
    v = q[:, 1:]
    inv = F.concat([w, -v], axis=1)
    return inv


def mge_quat_rotate(point_cloud, pose_7d):
    ndim = point_cloud.ndim
    if ndim == 2:
        N, _ = point_cloud.shape
        assert pose_7d.shape[0] == 1
        # repeat transformation vector for each point in shape
        quat = pose_7d[:, 0:4].expand([N, 1])
        rotated_point_cloud = mge_qrot(quat, point_cloud)

    elif ndim == 3:
        B, N, _ = point_cloud.shape
        quat = F.tile(F.expand_dims(pose_7d[:, 0:4], axis=1), (1, N, 1))
        rotated_point_cloud = mge_qrot(quat, point_cloud)

    else:
        raise RuntimeError("point cloud dim must be 2 or 3 !")

    return rotated_point_cloud


def mge_quat_transform(pose_7d, pc, normal=None):
    pc_t = mge_quat_rotate(pc, pose_7d) + pose_7d[:, 4:].reshape(-1, 1, 3)  # Ps" = R*Ps + t
    if normal is not None:
        normal_t = mge_quat_rotate(normal, pose_7d)
        return pc_t, normal_t
    else:
        return pc_t


def np_qmul(q, r):
    q = torch.from_numpy(q).contiguous()
    r = torch.from_numpy(r).contiguous()
    return torch_qmul(q, r).numpy()


def np_qrot(q, v):
    q = torch.from_numpy(q).contiguous()
    v = torch.from_numpy(v).contiguous()
    return torch_qrot(q, v).numpy()


def np_quat2euler(q, order, epsilon=0, use_gpu=False):
    if use_gpu:
        q = torch.from_numpy(q).cuda()
        return torch_quat2euler(q, order, epsilon).cpu().numpy()
    else:
        q = torch.from_numpy(q).contiguous()
        return torch_quat2euler(q, order, epsilon).numpy()


def np_qfix(q):
    """
    Enforce quaternion continuity across the time dimension by selecting
    the representation (q or -q) with minimal euclidean_distance (or, equivalently, maximal dot product)
    between two consecutive frames.
    Expects a tensor of shape (L, J, 4), where L is the sequence length and J is the number of joints.
    Returns a tensor of the same shape.
    """
    assert len(q.shape) == 3
    assert q.shape[-1] == 4

    result = q.copy()
    dot_products = np.sum(q[1:] * q[:-1], axis=2)
    mask = dot_products < 0
    mask = (np.cumsum(mask, axis=0) % 2).astype(bool)
    result[1:][mask] *= -1
    return result


def np_expmap2quat(e):
    """
    Convert axis-angle rotations (aka exponential maps) to quaternions.
    Stable formula from "Practical Parameterization of Rotations Using the Exponential Map".
    Expects a tensor of shape (*, 3), where * denotes any number of dimensions.
    Returns a tensor of shape (*, 4).
    """
    assert e.shape[-1] == 3

    original_shape = list(e.shape)
    original_shape[-1] = 4
    e = e.reshape(-1, 3)

    theta = np.linalg.norm(e, axis=1).reshape(-1, 1)
    w = np.cos(0.5 * theta).reshape(-1, 1)
    xyz = 0.5 * np.sinc(0.5 * theta / np.pi) * e
    return np.concatenate((w, xyz), axis=1).reshape(original_shape)


def np_euler2quat(e, order):
    """
    Convert Euler angles to quaternions.
    """
    assert e.shape[-1] == 3

    original_shape = list(e.shape)
    original_shape[-1] = 4

    e = e.reshape(-1, 3)

    x = e[:, 0]
    y = e[:, 1]
    z = e[:, 2]

    rx = np.stack((np.cos(x / 2), np.sin(x / 2), np.zeros_like(x), np.zeros_like(x)), axis=1)
    ry = np.stack((np.cos(y / 2), np.zeros_like(y), np.sin(y / 2), np.zeros_like(y)), axis=1)
    rz = np.stack((np.cos(z / 2), np.zeros_like(z), np.zeros_like(z), np.sin(z / 2)), axis=1)

    result = None
    for coord in order:
        if coord == "x":
            r = rx
        elif coord == "y":
            r = ry
        elif coord == "z":
            r = rz
        else:
            raise
        if result is None:
            result = r
        else:
            result = np_qmul(result, r)

    # Reverse antipodal representation to have a non-negative "w"
    if order in ["xyz", "yzx", "zxy"]:
        result *= -1

    return result.reshape(original_shape)


if __name__ == "__main__":
    import se3
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
    R = (Rx @ Ry @ Rz)  # (3, 3)
    t = np.random.uniform(-0.5, 0.5, 3)[:, None]  # (3, 1)
    trans = np.concatenate((R, t), axis=1)[None, :, :].astype(np.float32)  # (1, 3, 4)
    pose = se3.np_mat2quat(np.concatenate((R, t), axis=1))[None, :].astype(np.float32)  # (1, 7)
    pc = np.random.randn(1, 5, 3).astype(np.float32)
    torch_trans = torch.from_numpy(trans)
    torch_pose = torch.from_numpy(pose)
    torch_pc = torch.from_numpy(pc)
    mge_trans = mge.tensor(trans)
    mge_pose = mge.tensor(pose)
    mge_pc = mge.tensor(pc)

    ################### [torch_qmul] vs [mge_qmul] ################### pass
    torch_quat_mul = torch_qmul(torch_pose[:, :4], torch_pose[:, :4])
    mge_quat_mul = mge_qmul(mge_pose[:, :4], mge_pose[:, :4])
    print(torch_quat_mul)
    print(mge_quat_mul)

    ################### [torch_qrot] vs [mge_qrot] ################### pass
    torch_q = torch.cat((torch_pose[:, :4], torch_pose[:, :4]), dim=0)  # (B, 4)
    torch_v = torch.cat((torch_pose[:, 4:], torch_pose[:, 4:]), dim=0)  # (B, 3)
    mge_q = F.concat((mge_pose[:, :4], mge_pose[:, :4]), axis=0)
    mge_v = F.concat((mge_pose[:, 4:], mge_pose[:, 4:]), axis=0)
    torch_quat_rot = torch_qrot(torch_q, torch_v)
    mge_quat_rot = mge_qrot(mge_q, mge_v)
    print(torch_quat_rot)
    print(mge_quat_rot)

    ################### [torch_quat2mat] vs [mge_quat2mat] ################### pass
    torch_mat = torch_quat2mat(torch_pose)
    mge_mat = mge_quat2mat(mge_pose)
    print(torch_mat)
    print(mge_mat)

    ################### [torch_transform_pose] vs [mge_transform_pose] ################### pass
    torch_pose = torch_transform_pose(torch_pose, torch_pose)
    mge_pose = mge_transform_pose(mge_pose, mge_pose)
    print(torch_pose)
    print(mge_pose)

    ################### [torch_quat_rotate] vs [mge_quat_rotate] ################### pass
    torch_pc_trans = torch_quat_rotate(torch_pc, torch_pose)
    mge_pc_trans = mge_quat_rotate(mge_pc, mge_pose)
    print(torch_pc_trans)
    print(mge_pc_trans)

    ################### [torch_quat_transform] vs [mge_quat_transform] ################### pass
    torch_pc_trans = torch_quat_transform(torch_pose, torch_pc)
    mge_pc_trans = mge_quat_transform(mge_pose, mge_pc)
    print(torch_pc_trans)
    print(mge_pc_trans)
