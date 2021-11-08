import megengine as mge
import megengine.module as nn
import megengine.functional as F
from model.module import Encoder, Fusion, Decoder, Regression
from common import se3, quaternion
import math


class OMNet(nn.Module):
    def __init__(self, params):
        super(OMNet, self).__init__()
        self.num_iter = params.titer
        self.encoder = [Encoder() for _ in range(self.num_iter)]
        self.fusion = [Fusion() for _ in range(self.num_iter)]
        self.decoder = [Decoder() for _ in range(self.num_iter)]
        self.regression = [Regression() for _ in range(self.num_iter)]
        self.overlap_dist = params.overlap_dist

        for m in self.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
                nn.init.msra_normal_(m.weight, a=math.sqrt(5))
                if m.bias is not None:
                    fan_in, _ = nn.init.calculate_fan_in_and_fan_out(m.weight)
                    bound = 1 / math.sqrt(fan_in)
                    nn.init.uniform_(m.bias, -bound, bound)
            # elif isinstance(m, nn.BatchNorm1d):
            #     nn.init.ones_(m.weight)
            #     nn.init.zeros_(m.bias)

    def generate_overlap_mask(self, points_src, points_ref, mask_src, mask_ref, transform_gt):
        points_src[F.logical_not(mask_src.astype("bool")), :] = 50.0
        points_ref[F.logical_not(mask_ref.astype("bool")), :] = 100.0
        points_src = se3.mge_transform(transform_gt, points_src)
        points_src = F.expand_dims(points_src, axis=2)
        points_ref = F.expand_dims(points_ref, axis=1)
        dist_matrix = F.sqrt(F.sum(F.square(points_src - points_ref), axis=-1))  # (B, N, N)
        dist_s2r = F.min(dist_matrix, axis=2)
        dist_r2s = F.min(dist_matrix, axis=1)
        overlap_src_mask = dist_s2r < self.overlap_dist  # (B, N)
        overlap_ref_mask = dist_r2s < self.overlap_dist  # (B, N)
        return overlap_src_mask, overlap_ref_mask

    def forward(self, data_batch):
        endpoints = {}

        xyz_src = data_batch["points_src"]
        xyz_ref = data_batch["points_ref"]
        transform_gt = data_batch["transform_gt"]
        pose_gt = data_batch["pose_gt"]

        # init endpoints
        all_src_cls_pair = []
        all_ref_cls_pair = []
        all_transform_pair = []
        all_pose_pair = []
        all_xyz_src_t = [xyz_src]

        # init params
        B, src_N, _ = xyz_src.shape
        _, ref_N, _ = xyz_ref.shape
        init_quat = F.tile(mge.tensor([1, 0, 0, 0], dtype="float32"), (B, 1))  # (B, 4)
        init_translate = F.tile(mge.tensor([0, 0, 0], dtype="float32"), (B, 1))  # (B, 3)
        pose_pred = F.concat((init_quat, init_translate), axis=1)  # (B, 7)

        # rename xyz_src
        xyz_src_iter = F.copy(xyz_src, device=xyz_src.device)

        for i in range(self.num_iter):
            # deley mask
            if i < 2:
                src_pred_mask = F.ones((B, src_N), dtype=xyz_src.dtype)
                ref_pred_mask = F.ones((B, ref_N), dtype=xyz_ref.dtype)

            # encoder
            src_encoder_feats, src_glob_feat = self.encoder[i](xyz_src_iter.transpose(0, 2, 1).detach(), F.expand_dims(src_pred_mask,
                                                                                                                       axis=1))
            ref_encoder_feats, ref_glob_feat = self.encoder[i](xyz_ref.transpose(0, 2, 1).detach(), F.expand_dims(ref_pred_mask, axis=1))
            # fusion
            src_concat_feat = F.concat(
                (src_encoder_feats[0], F.repeat(src_glob_feat, src_N, axis=2), F.repeat(ref_glob_feat, src_N, axis=2)), axis=1)
            ref_concat_feat = F.concat(
                (ref_encoder_feats[0], F.repeat(ref_glob_feat, ref_N, axis=2), F.repeat(src_glob_feat, ref_N, axis=2)), axis=1)
            _, src_fused_feat = self.fusion[i](src_concat_feat, F.expand_dims(src_pred_mask, axis=1))
            _, ref_fused_feat = self.fusion[i](ref_concat_feat, F.expand_dims(ref_pred_mask, axis=1))

            # decoder
            src_decoder_feats, src_cls_pred = self.decoder[i](src_fused_feat)
            ref_decoder_feats, ref_cls_pred = self.decoder[i](ref_fused_feat)

            # regression
            src_feat = F.concat(src_decoder_feats, axis=1) * F.expand_dims(src_pred_mask, axis=1)
            ref_feat = F.concat(ref_decoder_feats, axis=1) * F.expand_dims(ref_pred_mask, axis=1)
            concat_feat = F.concat((src_fused_feat, src_feat, ref_fused_feat, ref_feat), axis=1)
            concat_feat = F.max(concat_feat, axis=-1)
            pose_pred_iter = self.regression[i](concat_feat)  # (B, 7)
            xyz_src_iter = quaternion.mge_quat_transform(pose_pred_iter, xyz_src_iter.detach())
            pose_pred = quaternion.mge_transform_pose(pose_pred.detach(), pose_pred_iter)
            transform_pred = quaternion.mge_quat2mat(pose_pred)

            # compute overlap and cls gt
            overlap_src_mask, overlap_ref_mask = self.generate_overlap_mask(F.copy(xyz_src, device=xyz_src.device),
                                                                            F.copy(xyz_ref, device=xyz_ref.device), src_pred_mask,
                                                                            ref_pred_mask, transform_gt)
            # overlap_src_mask, overlap_ref_mask = self.generate_overlap_mask(xyz_src, xyz_ref, src_pred_mask, ref_pred_mask, transform_gt)
            src_cls_gt = F.ones((B, src_N)) * overlap_src_mask
            ref_cls_gt = F.ones((B, ref_N)) * overlap_ref_mask
            src_pred_mask = F.argmax(src_cls_pred, axis=1)
            ref_pred_mask = F.argmax(ref_cls_pred, axis=1)

            # add endpoints
            all_src_cls_pair.append([src_cls_gt, src_cls_pred])
            all_ref_cls_pair.append([ref_cls_gt, ref_cls_pred])
            all_transform_pair.append([transform_gt, transform_pred])
            all_pose_pair.append([pose_gt, pose_pred])
            all_xyz_src_t.append(xyz_src_iter)

        endpoints["all_src_cls_pair"] = all_src_cls_pair
        endpoints["all_ref_cls_pair"] = all_ref_cls_pair
        endpoints["all_transform_pair"] = all_transform_pair
        endpoints["all_pose_pair"] = all_pose_pair
        endpoints["transform_pair"] = [transform_gt, transform_pred]
        endpoints["pose_pair"] = [pose_gt, pose_pred]
        endpoints["all_xyz_src_t"] = all_xyz_src_t

        return endpoints


def fetch_net(params):
    if params.net_type == "omnet":
        net = OMNet(params)

    else:
        raise NotImplementedError
    return net
