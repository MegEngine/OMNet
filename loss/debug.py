from megengine.functional import metric
import torch
import numpy as np
import megengine as mge
import megengine.functional as F
import functools
from megengine.core.tensor.array_method import _reduce
from megengine.tensor import Tensor
from megengine.functional.elemwise import abs, log, not_equal
from megengine.functional.nn import indexing_one_hot, logsigmoid, logsumexp, relu
from pytorch3d import transforms as p3d_transforms
from scipy.spatial.transform import Rotation


def _reduce_output(loss_fn):
    r"""
    Wrapper to apply canonical reductions to loss outputs.
    """
    @functools.wraps(loss_fn)
    def reduced_loss_fn(*args, reduction="mean", **kwargs):
        loss = loss_fn(*args, **kwargs)
        if reduction == "none":
            return loss
        elif reduction in ("mean", "sum"):
            return _reduce(reduction)(loss)
        else:
            raise ValueError("{} is not a valid value for reduction".format(reduction))

    return reduced_loss_fn


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
        weight = weight / F.sum(weight)
        class_weight = weight[label.flatten()].reshape(label.shape)

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
            return F.sum((logZ - primary_term) * class_weight, axis=1, keepdims=True) / F.sum(class_weight, axis=1, keepdims=True)
    if not with_logits:
        pred = log(pred)
    if weight is None:
        return logZ - ls * pred.mean(axis) - (1 - ls) * primary_term
    else:
        return F.sum((logZ - ls * pred.mean(axis) -
                      (1 - ls) * primary_term) * class_weight, axis=1, keepdims=True) / F.sum(class_weight, axis=1, keepdims=True)


def np_knn(pts, random_pt, k):
    distance = np.sum((pts - random_pt)**2, axis=1)
    idx = np.argsort(distance)[:k]  # (k,)
    return idx


def torch_knn(pts, random_pt, k):
    distance = torch.sum((pts - random_pt)**2, dim=1)
    idx = distance.topk(k=k, dim=0, largest=False)[1]  # (batch_size, num_points, k)
    return idx


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


def mge_generate_overlap_mask(points_src, points_ref, mask_src, mask_ref, transform_gt):
    points_src[F.logical_not(mask_src.astype("bool")), :] = 50.0
    points_ref[F.logical_not(mask_ref.astype("bool")), :] = 100.0
    points_src = mge_transform(transform_gt, points_src)
    points_src = F.expand_dims(points_src, axis=2)
    points_ref = F.expand_dims(points_ref, axis=1)
    dist_matrix = F.sqrt(F.sum(F.square(points_src - points_ref), axis=-1))  # (B, N, N)
    dist_s2r = F.min(dist_matrix, axis=2)
    dist_r2s = F.min(dist_matrix, axis=1)
    print("mge_dist_s2r: ", dist_s2r)
    print("mge_dist_r2s: ", dist_r2s)
    overlap_src_mask = dist_s2r < 0.1  # (B, N)
    overlap_ref_mask = dist_r2s < 0.1  # (B, N)
    return overlap_src_mask, overlap_ref_mask


def torch_generate_overlap_mask(points_src, points_ref, mask_src, mask_ref, transform_gt):
    from cuda.chamfer_distance import ChamferDistance
    points_src[torch.logical_not(mask_src), :] = 50.0
    points_ref[torch.logical_not(mask_ref), :] = 100.0
    dist_s2r, dist_r2s = ChamferDistance()(torch_transform(transform_gt, points_src), points_ref)
    dist_s2r = torch.sqrt(dist_s2r)
    dist_r2s = torch.sqrt(dist_r2s)
    print("torch_dist_s2r: ", dist_s2r)
    print("torch_dist_r2s: ", dist_r2s)
    overlap_src_mask = dist_s2r < 0.1  # (B, N)
    overlap_ref_mask = dist_r2s < 0.1  # (B, N)
    return overlap_src_mask, overlap_ref_mask


def torch_loss(endpoints):
    loss = {}
    for i in range(2):
        # cls loss
        src_cls_pair, ref_cls_pair = endpoints['all_src_cls_pair'][i], endpoints['all_ref_cls_pair'][i]
        src_cls = torch.nn.CrossEntropyLoss(weight=torch.tensor([0.7, 0.3]))(src_cls_pair[1], src_cls_pair[0].long())
        ref_cls = torch.nn.CrossEntropyLoss(weight=torch.tensor([0.7, 0.3]))(ref_cls_pair[1], ref_cls_pair[0].long())
        loss['cls_{}'.format(i)] = (src_cls + ref_cls) / 2.0
        # reg loss
        pose_pair = endpoints["all_pose_pair"][i]
        loss["quat_{}".format(i)] = torch.nn.L1Loss()(pose_pair[0][:, :4], pose_pair[1][:, :4]) * 1
        loss["translate_{}".format(i)] = torch.nn.MSELoss()(pose_pair[0][:, 4:], pose_pair[1][:, 4:]) * 4

    # total loss
    discount_factor = 1.0  # Early iterations will be discounted
    total_loss = []
    for k in loss:
        discount = discount_factor**(4 - int(k[k.rfind("_") + 1:]) - 1)
        total_loss.append(loss[k].float() * discount)
    loss["total"] = torch.sum(torch.stack(total_loss), dim=0)
    print(loss)


def mge_loss(endpoints):
    loss = {}
    for i in range(2):
        # mask loss
        src_cls_pair, ref_cls_pair = endpoints["all_src_cls_pair"][i], endpoints["all_ref_cls_pair"][i]
        src_cls = F.nn.frequency_weighted_cross_entropy(src_cls_pair[1], src_cls_pair[0], weight=mge.tensor([0.7, 0.3]))
        ref_cls = F.nn.frequency_weighted_cross_entropy(ref_cls_pair[1], ref_cls_pair[0], weight=mge.tensor([0.7, 0.3]))
        loss["cls_{}".format(i)] = (src_cls + ref_cls) / 2.0
        # reg loss
        pose_pair = endpoints["all_pose_pair"][i]
        loss["quat_{}".format(i)] = F.nn.l1_loss(pose_pair[0][:, :4], pose_pair[1][:, :4]) * 1
        loss["translate_{}".format(i)] = F.nn.square_loss(pose_pair[0][:, 4:], pose_pair[1][:, 4:]) * 4
    # total loss
    total_losses = []
    for k in loss:
        total_losses.append(loss[k])
    loss["total"] = F.sum(F.concat(total_losses))
    print(loss)


def torch_dcm2euler(mats, seq, degrees=True):
    if seq == "xyz":
        eulers = p3d_transforms.matrix_to_euler_angles(mats, "ZYX")
    elif seq == "zyx":
        eulers = p3d_transforms.matrix_to_euler_angles(mats, "XYZ")
    eulers = eulers[:, [2, 1, 0]]
    if degrees:
        eulers = eulers / np.pi * 180
    return eulers


def mge_dcm2euler(mats, seq, degrees=True):
    mats = mats.numpy()
    eulers = []
    for i in range(mats.shape[0]):
        r = Rotation.from_matrix(mats[i])
        eulers.append(r.as_euler(seq, degrees=degrees))
    return mge.tensor(np.stack(eulers))


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


def torch_metrics(endpoints):
    metrics = {}
    gt_transforms = endpoints["transform_pair"][0]
    pred_transforms = endpoints["transform_pair"][1]

    # Euler angles, Individual translation errors (Deep Closest Point convention)
    r_gt_euler_deg = torch_dcm2euler(gt_transforms[:, :3, :3], seq="zyx")
    r_pred_euler_deg = torch_dcm2euler(pred_transforms[:, :3, :3], seq="zyx")

    t_gt = gt_transforms[:, :3, 3]
    t_pred = pred_transforms[:, :3, 3]

    r_mse = torch.mean((r_gt_euler_deg - r_pred_euler_deg)**2, dim=1)
    r_mae = torch.mean(torch.abs(r_gt_euler_deg - r_pred_euler_deg), dim=1)
    t_mse = torch.mean((t_gt - t_pred)**2, dim=1)
    t_mae = torch.mean(torch.abs(t_gt - t_pred), dim=1)

    r_rmse = torch.mean(torch.sqrt(r_mse))
    t_rmse = torch.mean(torch.sqrt(t_mse))
    r_mae = torch.mean(r_mae)
    t_mae = torch.mean(t_mae)

    # Rotation, translation errors (isotropic, i.e. doesn"t depend on error
    # direction, which is more representative of the actual error)
    concatenated = torch_concatenate(torch_inverse(gt_transforms), pred_transforms)
    rot_trace = concatenated[:, 0, 0] + concatenated[:, 1, 1] + concatenated[:, 2, 2]
    residual_rotdeg = torch.acos(torch.clamp(0.5 * (rot_trace - 1), min=-1.0, max=1.0)) * 180.0 / np.pi
    residual_transmag = concatenated[:, :, 3].norm(dim=-1)
    err_r = torch.mean(residual_rotdeg)
    err_t = torch.mean(residual_transmag)

    # weighted score of isotropic errors
    score = err_r * 0.01 + err_t
    metrics = {"R_RMSE": r_rmse, "R_MAE": r_mae, "t_RMSE": t_rmse, "t_MAE": t_mae, "Err_R": err_r, "Err_t": err_t, "score": score}
    return metrics


def mge_metrics(endpoints, ):
    metrics = {}
    gt_transforms = endpoints["transform_pair"][0]
    pred_transforms = endpoints["transform_pair"][1]

    # Euler angles, Individual translation errors (Deep Closest Point convention)
    r_gt_euler_deg = mge_dcm2euler(gt_transforms[:, :3, :3], seq="zyx")
    r_pred_euler_deg = mge_dcm2euler(pred_transforms[:, :3, :3], seq="zyx")

    t_gt = gt_transforms[:, :3, 3]
    t_pred = pred_transforms[:, :3, 3]

    r_mse = F.mean((r_gt_euler_deg - r_pred_euler_deg)**2, axis=1)
    r_mae = F.mean(F.abs(r_gt_euler_deg - r_pred_euler_deg), axis=1)
    t_mse = F.mean((t_gt - t_pred)**2, axis=1)
    t_mae = F.mean(F.abs(t_gt - t_pred), axis=1)

    r_rmse = F.mean(F.sqrt(r_mse))
    t_rmse = F.mean(F.sqrt(t_mse))
    r_mae = F.mean(r_mae)
    t_mae = F.mean(t_mae)

    # Rotation, translation errors (isotropic, i.e. doesn"t depend on error
    # direction, which is more representative of the actual error)
    concatenated = mge_concatenate(mge_inverse(gt_transforms), pred_transforms)
    rot_trace = concatenated[:, 0, 0] + concatenated[:, 1, 1] + concatenated[:, 2, 2]
    residual_rotdeg = F.acos(F.clip(0.5 * (rot_trace - 1), -1.0, 1.0)) * 180.0 / np.pi
    residual_transmag = F.norm(concatenated[:, :, 3], axis=-1)
    err_r = F.mean(residual_rotdeg)
    err_t = F.mean(residual_transmag)

    # weighted score of isotropic errors
    score = err_r * 0.01 + err_t

    metrics = {"R_RMSE": r_rmse, "R_MAE": r_mae, "t_RMSE": t_rmse, "t_MAE": t_mae, "Err_R": err_r, "Err_t": err_t, "score": score}
    return metrics


# ############################# [torch_cross_entropy] vs [mge_cross_entropy] #############################
# np_pred = np.random.randn(2, 2, 5)
# np_label = np.array([[1, 0, 1, 0, 1], [1, 0, 1, 0, 1]])
# torch_pred = torch.from_numpy(np_pred)
# torch_label = torch.from_numpy(np_label).long()
# torch_loss = torch.nn.CrossEntropyLoss(weight=torch.tensor([0.7, 0.3]).double())(torch_pred, torch_label)
# # torch_loss = torch.nn.CrossEntropyLoss()(torch_pred, torch_label)
# mge_pred = mge.tensor(np_pred)
# print(mge_pred.dtype)
# mge_label = mge.tensor(np_label).astype(np.int32)
# mge_loss = F.nn.frequency_weighted_cross_entropy(mge_pred, mge_label, weight=mge.tensor([0.7, 0.3]))
# # mge_loss = frequency_weighted_cross_entropy(mge_pred, mge_label)
# print(torch_loss)
# print(mge_loss)

# ############################# [torch_knn] vs [np_knn] #############################
# np_pc = np.random.randn(1024, 3).astype(np.float32)
# np_random_p = np.random.random(size=(1, 3)) + np.array([[500, 500, 500]]) * np.random.choice([1, -1, 1, -1])
# np_random_p = np_random_p.astype(np.float32)
# torch_pc = torch.from_numpy(np_pc)
# torch_random_p = torch.from_numpy(np_random_p)
# np_idx = np_knn(np_pc, np_random_p, k=768)
# torch_idx = torch_knn(torch_pc, torch_random_p, k=768)
# print(np_idx[:10])
# print(torch_idx[:10])
# print(np.sum(np_idx == torch_idx.numpy()))
# print(np_pc[np_idx[np_idx != torch_idx.numpy()]])

# ############################# [torch_generate_overlap_mask] vs [mge_generate_overlap_mask] #############################
# xyz_src = np.random.rand(2, 1024, 3).astype(np.float32)
# xyz_ref = np.random.rand(2, 1024, 3).astype(np.float32)
# torch_xyz_src = torch.from_numpy(xyz_src)
# torch_xyz_ref = torch.from_numpy(xyz_ref)

# mge_xyz_src = mge.tensor(xyz_src)
# mge_xyz_ref = mge.tensor(xyz_ref)
# print(mge_xyz_src)
# src_pred_mask = np.random.randint(0, 2, (2, 1024)).astype(np.float32)
# ref_pred_mask = np.random.randint(0, 2, (2, 1024)).astype(np.float32)
# torch_src_pred_mask = torch.from_numpy(src_pred_mask)
# torch_ref_pred_mask = torch.from_numpy(ref_pred_mask)
# mge_src_pred_mask = mge.tensor(src_pred_mask)
# mge_ref_pred_mask = mge.tensor(ref_pred_mask)
# torch_transform_gt = torch.Tensor([[[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]]]).repeat(2, 1, 1)
# mge_transform_gt = F.tile(mge.tensor([[[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]]]), (2, 1, 1))
# print("mge_src_pred_mask", mge_src_pred_mask)
# torch_overlap_src_mask, torch_overlap_ref_mask = torch_generate_overlap_mask(torch_xyz_src.clone(), torch_xyz_ref.clone(),
#                                                                              torch_src_pred_mask, torch_ref_pred_mask, torch_transform_gt)
# mge_overlap_src_mask, mge_overlap_ref_mask = mge_generate_overlap_mask(F.copy(mge_xyz_src), F.copy(mge_xyz_ref), mge_src_pred_mask,
#                                                                        mge_ref_pred_mask, mge_transform_gt)
# print("mge_src_pred_mask", mge_src_pred_mask)
# # print(torch_overlap_src_mask)
# # print(mge_overlap_src_mask)
# print(mge_xyz_src)

# ############################# [torch_loss] vs [mge_loss] #############################
# np.random.seed(0)
# torch_end_points = {}
# all_src_cls_pair = [[torch.from_numpy(np.random.rand(2, 3)).long(),
#                      torch.from_numpy(np.random.rand(2, 2, 3)).float()],
#                     [torch.from_numpy(np.random.rand(2, 3)).long(),
#                      torch.from_numpy(np.random.rand(2, 2, 3)).float()]]
# all_ref_cls_pair = [[torch.from_numpy(np.random.rand(2, 3)).long(),
#                      torch.from_numpy(np.random.rand(2, 2, 3)).float()],
#                     [torch.from_numpy(np.random.rand(2, 3)).long(),
#                      torch.from_numpy(np.random.rand(2, 2, 3)).float()]]
# torch_end_points["all_src_cls_pair"] = all_src_cls_pair
# torch_end_points["all_ref_cls_pair"] = all_ref_cls_pair
# all_pose_pair = [[torch.from_numpy(np.random.rand(2, 7)), torch.from_numpy(np.random.rand(2, 7))],
#                  [torch.from_numpy(np.random.rand(2, 7)), torch.from_numpy(np.random.rand(2, 7))]]
# torch_end_points["all_pose_pair"] = all_pose_pair

# np.random.seed(0)
# mge_end_points = {}
# all_src_cls_pair = [[mge.tensor(np.random.rand(2, 3)), mge.tensor(np.random.rand(2, 2, 3))],
#                     [mge.tensor(np.random.rand(2, 3)), mge.tensor(np.random.rand(2, 2, 3))]]
# all_ref_cls_pair = [[mge.tensor(np.random.rand(2, 3)), mge.tensor(np.random.rand(2, 2, 3))],
#                     [mge.tensor(np.random.rand(2, 3)), mge.tensor(np.random.rand(2, 2, 3))]]
# mge_end_points["all_src_cls_pair"] = all_src_cls_pair
# mge_end_points["all_ref_cls_pair"] = all_ref_cls_pair
# all_pose_pair = [[mge.tensor(np.random.rand(2, 7)), mge.tensor(np.random.rand(2, 7))],
#                  [mge.tensor(np.random.rand(2, 7)), mge.tensor(np.random.rand(2, 7))]]
# mge_end_points["all_pose_pair"] = all_pose_pair
# torch_loss(torch_end_points)
# mge_loss(mge_end_points)

# ############################# [torch_metrics] vs [mge_metrics] #############################
# anglex = np.random.uniform() * np.pi * 2
# angley = np.random.uniform() * np.pi * 2
# anglez = np.random.uniform() * np.pi * 2
# cosx = np.cos(anglex)
# cosy = np.cos(angley)
# cosz = np.cos(anglez)
# sinx = np.sin(anglex)
# siny = np.sin(angley)
# sinz = np.sin(anglez)
# Rx = np.array([[1, 0, 0], [0, cosx, -sinx], [0, sinx, cosx]])
# Ry = np.array([[cosy, 0, siny], [0, 1, 0], [-siny, 0, cosy]])
# Rz = np.array([[cosz, -sinz, 0], [sinz, cosz, 0], [0, 0, 1]])
# R = (Rx @ Ry @ Rz)[None, ...]  # (1, 3, 3)
# t = np.random.uniform(-0.5, 0.5, 3)[None, :, None]  # (1, 3, 1)
# a_trans = np.concatenate((R, t), axis=2)  # (1, 3, 4)

# anglex = np.random.uniform() * np.pi * 2
# angley = np.random.uniform() * np.pi * 2
# anglez = np.random.uniform() * np.pi * 2
# cosx = np.cos(anglex)
# cosy = np.cos(angley)
# cosz = np.cos(anglez)
# sinx = np.sin(anglex)
# siny = np.sin(angley)
# sinz = np.sin(anglez)
# Rx = np.array([[1, 0, 0], [0, cosx, -sinx], [0, sinx, cosx]])
# Ry = np.array([[cosy, 0, siny], [0, 1, 0], [-siny, 0, cosy]])
# Rz = np.array([[cosz, -sinz, 0], [sinz, cosz, 0], [0, 0, 1]])
# R = (Rx @ Ry @ Rz)[None, ...]  # (1, 3, 3)
# t = np.random.uniform(-0.5, 0.5, 3)[None, :, None]  # (1, 3, 1)
# b_trans = np.concatenate((R, t), axis=2)  # (1, 3, 4)

# torch_gt_trans = torch.from_numpy(a_trans)
# torch_pred_trans = torch.from_numpy(b_trans)
# mge_gt_trans = mge.tensor(a_trans)
# mge_pred_trans = mge.tensor(b_trans)

# torch_endpoints = {"transform_pair": [torch_gt_trans, torch_pred_trans]}
# mge_endpoints = {"transform_pair": [mge_gt_trans, mge_pred_trans]}

# torch_metric = torch_metrics(torch_endpoints)
# mge_metric = mge_metrics(mge_endpoints)
# print(torch_metric)
# print(mge_metric)
