import functools
import gorilla
import pointgroup_ops
import spconv.pytorch as spconv
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_max, scatter_mean
from sklearn.cluster import DBSCAN
import numpy as np

from spformer.utils import cuda_cast, rle_encode
from .backbone import ResidualBlock, UBlock
from .loss import Criterion
from .query_decoder import QueryDecoder
from .pseudo_labeler import PseudoLabeler


@gorilla.MODELS.register_module()
class SPFormer(nn.Module):

    def __init__(
        self,
        input_channel: int = 6,
        blocks: int = 5,
        block_reps: int = 2,
        media: int = 32,
        normalize_before=True,
        return_blocks=True,
        pool='mean',
        num_class=18,
        decoder=None,
        criterion=None,
        test_cfg=None,
        norm_eval=False,
        fix_module=[],
        total_epochs = 400,
    ):
        super().__init__()

        # backbone and pooling
        self.input_conv = spconv.SparseSequential(
            spconv.SubMConv3d(
                input_channel,
                media,
                kernel_size=3,
                padding=1,
                bias=False,
                indice_key='subm1',
            ))
        block = ResidualBlock
        norm_fn = functools.partial(nn.BatchNorm1d, eps=1e-4, momentum=0.1)
        block_list = [media * (i + 1) for i in range(blocks)]
        self.unet = UBlock(
            block_list,
            norm_fn,
            block_reps,
            block,
            indice_key_id=1,
            normalize_before=normalize_before,
            return_blocks=return_blocks,
        )
        self.output_layer = spconv.SparseSequential(norm_fn(media), nn.ReLU(inplace=True))
        self.pool = pool
        self.num_class = num_class
        self.total_epochs = total_epochs

        # decoder
        self.decoder = QueryDecoder(**decoder, in_channel=media, num_class=num_class)

        # criterion
        self.criterion = Criterion(**criterion, num_class=num_class)

        self.ps_labeler = PseudoLabeler(d_model=media)

        self.test_cfg = test_cfg
        self.norm_eval = norm_eval
        for module in fix_module:
            module = getattr(self, module)
            module.eval()
            for param in module.parameters():
                param.requires_grad = False

    def train(self, mode=True):
        super(SPFormer, self).train(mode)
        if mode and self.norm_eval:
            for m in self.modules():
                # trick: eval have effect on BatchNorm1d only
                if isinstance(m, nn.BatchNorm1d):
                    m.eval()

    def forward(self, batch, mode='loss'):
        if mode == 'loss':
            return self.loss(**batch)
        elif mode == 'predict':
            return self.predict(**batch)

    @cuda_cast
    def loss(self, scan_ids, voxel_coords, p2v_map, v2p_map, spatial_shape, feats, superpoints, batch_offsets, 
            point_batch_offsets, coords_float, superpoints_list, insts):
        batch_size = len(batch_offsets) - 1
        voxel_feats = pointgroup_ops.voxelization(feats, v2p_map)
        input = spconv.SparseConvTensor(voxel_feats, voxel_coords.int(), spatial_shape, batch_size)

        voxel_rgb_feats = voxel_feats[:, :3].clone()

        sp_feats, sp_coords, sp_rgb_feats, points_feat = self.extract_feat(input, superpoints, p2v_map, coords_float, voxel_rgb_feats)
        out = self.decoder(sp_feats, batch_offsets, sp_coords)

        out["batch_offsets"] = batch_offsets

        ps_insts, ps_prob_labels, one_box_points_preds, one_box_points_labs = self.get_ps_insts(
            scan_ids, coords_float, superpoints_list, insts, points_feat, self.num_class, point_batch_offsets)
        
        sp_ps_prob_labels = self.sp_pool(ps_prob_labels, superpoints)

        out["sp_prob_labels"] = sp_ps_prob_labels
        out["one_box_points_preds"] = one_box_points_preds
        out["one_box_points_labs"] = one_box_points_labs

        loss, loss_dict = self.criterion(out, ps_insts)
        return loss, loss_dict

    @cuda_cast
    def predict(self, scan_ids, voxel_coords, p2v_map, v2p_map, spatial_shape, feats, superpoints,
                batch_offsets, point_batch_offsets, coords_float, superpoints_list, insts):
        batch_size = len(batch_offsets) - 1
        voxel_feats = pointgroup_ops.voxelization(feats, v2p_map)
        input = spconv.SparseConvTensor(voxel_feats, voxel_coords.int(), spatial_shape, batch_size)

        voxel_rgb_feats = voxel_feats[:, :3].clone()

        sp_feats, sp_coords, sp_rgb_feats, _ = self.extract_feat(input, superpoints, p2v_map, coords_float, voxel_rgb_feats)
        out = self.decoder(sp_feats, batch_offsets, sp_coords)

        out["batch_offsets"] = batch_offsets
        out["sp_coords"] = sp_coords
        out["sp_rgb_feats"] = sp_rgb_feats

        ret = self.predict_by_feat(scan_ids, out, superpoints)
        return ret

    def predict_by_feat(self, scan_ids, out, superpoints):
        pred_labels = out['labels']
        pred_masks = out['masks']
        #pred_scores = out['scores']

        scores = F.softmax(pred_labels[0], dim=-1)[:, :-1]
        labels = torch.arange(
            self.num_class, device=scores.device).unsqueeze(0).repeat(self.decoder.num_query, 1).flatten(0, 1)

        scores, topk_idx = scores.flatten(0, 1).topk(self.test_cfg.topk_insts, sorted=False)
        labels = labels[topk_idx]
        labels += 1

        topk_idx = torch.div(topk_idx, self.num_class, rounding_mode='floor')

        mask_pred = pred_masks[0]
        mask_pred = mask_pred[topk_idx]
        mask_pred_sigmoid = mask_pred.sigmoid()
        # mask_pred before sigmoid()
        mask_pred = (mask_pred > 0).float()  # [n_p, M]
        mask_scores = (mask_pred_sigmoid * mask_pred).sum(1) / (mask_pred.sum(1) + 1e-6)
        scores = scores * mask_scores
        # get mask
        mask_pred = mask_pred[:, superpoints].int()

        # score_thr
        score_mask = scores > self.test_cfg.score_thr
        scores = scores[score_mask]  # (n_p,)
        labels = labels[score_mask]  # (n_p,)
        mask_pred = mask_pred[score_mask]  # (n_p, N)

        # npoint thr
        mask_pointnum = mask_pred.sum(1)
        npoint_mask = mask_pointnum > self.test_cfg.npoint_thr
        scores = scores[npoint_mask]  # (n_p,)
        labels = labels[npoint_mask]  # (n_p,)
        mask_pred = mask_pred[npoint_mask]  # (n_p, N)

        cls_pred = labels.cpu().numpy()
        score_pred = scores.cpu().numpy()
        mask_pred = mask_pred.cpu().numpy()

        pred_instances = []
        for i in range(cls_pred.shape[0]):
            pred = {}
            pred['scan_id'] = scan_ids[0]
            pred['label_id'] = cls_pred[i]
            pred['conf'] = score_pred[i]
            # rle encode mask to save memory
            pred['pred_mask'] = rle_encode(mask_pred[i])
            pred_instances.append(pred)
        return dict(scan_id=scan_ids[0], pred_instances=pred_instances)

    def sp_pool(self, x, superpoints):
        if self.pool == 'mean':
            x = scatter_mean(x, superpoints, dim=0)
        elif self.pool == 'max':
            x, _ = scatter_max(x, superpoints, dim=0)
        return x

    def extract_feat(self, x, superpoints, v2p_map, coords_float, voxel_rgb_feats):
        # backbone
        x = self.input_conv(x)
        x, _ = self.unet(x)
        x = self.output_layer(x)
        x = x.features[v2p_map.long()]  # (B*N, media)
        sp_rgb_feats = rgb_feats = voxel_rgb_feats[v2p_map.long()]
        points_feat = x

        sp_coords = coords_float
        # superpoint pooling
        x = self.sp_pool(x, superpoints)  # (B*M, media)
        sp_coords = self.sp_pool(coords_float, superpoints)  # (M, 3)
        sp_rgb_feats = self.sp_pool(rgb_feats, superpoints) 

        return x, sp_coords, sp_rgb_feats, points_feat

    def get_ps_insts(self, scan_ids, coords_float, superpoints_list, insts, points_feat, instance_classes=18, point_batch_offsets=None):
        ps_insts = []
        ps_prob_labels = []
        one_box_points_preds = []
        one_box_points_labs = []
        for b, (scan_id, superpoint, gt_inst) in enumerate(zip(scan_ids, superpoints_list, insts)):
            b_s, b_e = point_batch_offsets[b], point_batch_offsets[b + 1]
            coord_float = coords_float[b_s:b_e]
            point_feat = points_feat[b_s:b_e]
            ps_inst, ps_prob_label, one_box_points_pred, one_box_points_lab = self.ps_labeler(scan_id, coord_float, superpoint, point_feat, gt_inst, instance_classes)
            ps_insts.append(ps_inst)
            ps_prob_labels.append(ps_prob_label)
            one_box_points_preds.append(one_box_points_pred)
            one_box_points_labs.append(one_box_points_lab)
        ps_prob_labels = torch.cat(ps_prob_labels, dim=0)
        one_box_points_preds = torch.cat(one_box_points_preds, dim=0)
        one_box_points_labs = torch.cat(one_box_points_labs, dim=0)
        
        return ps_insts, ps_prob_labels, one_box_points_preds, one_box_points_labs

