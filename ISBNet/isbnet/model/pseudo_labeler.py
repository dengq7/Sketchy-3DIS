import torch
import torch.nn.functional as F
import torch.nn as nn
import torch_scatter
import numpy as np
from tqdm import tqdm


class PseudoLabeler(nn.Module):

    def __init__(self, d_model=32):
        super().__init__()
        self.box_cls = nn.Sequential(nn.Linear(d_model*3, d_model), nn.ReLU(), nn.Linear(d_model, 2))
        self.ignore_classes = [0, 1]

    def is_within_bb_torch(self, points, bb_min, bb_max):
        return torch.all(points >= bb_min, dim=-1) & torch.all(points <= bb_max, dim=-1)


    def is_box1_in_box2(self, box1, box2, offset=0.05):
        return torch.all((box1[:3] + offset) >= box2[:3]) & torch.all((box1[3:] - offset) <= box2[3:])


    def spp_align_label(self, spp, label, n_classes=-1, bb_occupancy_spp=None, prob_label=None):
        if n_classes == -1:
            n_classes = torch.max(label) + 1

        # breakpoint()
        onehot_label = F.one_hot(label.long(), num_classes=n_classes).permute(1, 0)

        n_label, n_points = onehot_label.shape[:2]
        _, spp_ids = torch.unique(spp, return_inverse=True)

        count_label_spp = torch_scatter.scatter(
            onehot_label, spp_ids.expand(n_label, n_points), dim=-1, reduce="sum"
        )  # n_labels, n_spp
        # spp_mask = (mean_spp_inst > 0.5)

        # count_label_spp_sum = (count_label_spp).sum(dim=0)

        if bb_occupancy_spp is not None:
            count_label_spp[1:, :] = count_label_spp[1:, :] * (bb_occupancy_spp == 1).float()

        label_spp = torch.argmax(count_label_spp, dim=0)  # n_spp

        refined_label = label_spp[spp_ids]

        if prob_label is None:
            return refined_label

        prob_label_spp = torch_scatter.scatter(prob_label, spp_ids, reduce="mean")

        refined_prob_label = prob_label_spp[spp_ids]
        return refined_label, refined_prob_label


    def gen_pseudo_label_box2mask(
        self, coords_float, spp, instance_cls, instance_box, instance_box_volume, instance_classes=18, points_feat=None, dataset_name="scannetv2"
    ):
        n_points_ = len(coords_float)
        n_boxes = len(instance_box)

        if n_boxes == 0:
            return None, None

        bb_occupancy = self.is_within_bb_torch(
            coords_float[:, None, :], instance_box[None, :, :3] - 0.005, instance_box[None, :, 3:] + 0.005
        )  # N_points, N_box

        # none_cond = (instance_cls[in_s:in_t] == -100)
        # bb_occupancy[:, none_cond] = 0

        inst_per_point = torch.ones(n_points_, dtype=torch.long, device=coords_float.device) * -100
        # activations_per_point = [np.argwhere(bb_occupancy[:, i] == 1) for i in range(len(scene['positions']))]
        # number of BBs a point is in
        num_BBs_per_point = bb_occupancy.long().sum(dim=1)

        # NOTE process points within multi boxes
        point_inds, box_inds = torch.nonzero(bb_occupancy[(num_BBs_per_point > 1), :], as_tuple=True)
        volume_box = instance_box_volume[box_inds]
        min_volume, argmin_volume = torch_scatter.scatter_min(volume_box, point_inds, dim=0)
        assert len(min_volume) == (num_BBs_per_point > 1).sum()
        corres_multibox_active = box_inds[argmin_volume]

        corres_box_active = torch.nonzero(bb_occupancy[(num_BBs_per_point == 1), :], as_tuple=True)[1]

        inst_per_point[(num_BBs_per_point == 1)] = corres_box_active  # NOTE: 1 box -> assign corres box
        inst_per_point[(num_BBs_per_point == 0)] = -1  # NOTE: nobox -> background -1

        if points_feat != None:

            fp_point_inds, fp_corres_box_active = torch.nonzero(bb_occupancy[(num_BBs_per_point >= 1), :], as_tuple=True)

            segments_feat = torch_scatter.scatter_mean(points_feat[fp_point_inds], fp_corres_box_active, dim=0)   # N_Segment, d_model

            one_plus_point_inds, one_plus_point_active = torch.unique(point_inds, return_inverse=True)

            one_plus_points_feat = points_feat[one_plus_point_inds]  # N_1+_points_num, d_model

            _diff_cos = F.cosine_similarity(one_plus_points_feat[:, None, :], segments_feat[None, :, :], dim=-1)    # N_1+_points_num, N_Segment
            
            _diff_cos_sorted_vale, _diff_cos_sorted_index = torch.max(_diff_cos, dim=-1)    # N_1+_points_num, 1

            corres_multibox_active[one_plus_point_inds] = _diff_cos_sorted_index
            
        inst_per_point[
            (num_BBs_per_point > 1)
        ] = corres_multibox_active  # NOTE: multibox -> assign box with the smallest volume

        if dataset_name == "scannetv2":

            _, spp = torch.unique(spp, return_inverse=True)

            inst_per_point = torch.where(inst_per_point >= 0, inst_per_point + 1, 0)
            inst_per_point = self.spp_align_label(spp, inst_per_point, n_classes=n_boxes + 1)
            inst_per_point = torch.where(inst_per_point > 0, inst_per_point - 1, -1).int()

        ps_semantic_label = torch.ones(n_points_, dtype=torch.long, device=coords_float.device) * -100
        ps_instance_label = torch.ones(n_points_, dtype=torch.long, device=coords_float.device) * -100
        fp_ps_sem_label = instance_cls[inst_per_point[inst_per_point >= 0].long()]
        ps_semantic_label[inst_per_point >= 0] = fp_ps_sem_label
        ps_semantic_label[inst_per_point == -1] = instance_classes

        ps_instance_label[inst_per_point >= 0] = inst_per_point[inst_per_point >= 0].long()

        return ps_semantic_label, ps_instance_label

    
    def batch_giou_cross(self, boxes1, boxes2):
        # boxes1: N, 6
        # boxes2: M, 6
        # out: N, M
        boxes1 = boxes1[:, None, :]
        boxes2 = boxes2[None, :, :]
        intersection = torch.prod(
            torch.clamp(
                (torch.min(boxes1[..., 3:], boxes2[..., 3:]) - torch.max(boxes1[..., :3], boxes2[..., :3])), min=0.0
            ),
            -1,
        )  # N

        boxes1_volumes = torch.prod(torch.clamp((boxes1[..., 3:] - boxes1[..., :3]), min=0.0), -1)
        boxes2_volumes = torch.prod(torch.clamp((boxes2[..., 3:] - boxes2[..., :3]), min=0.0), -1)

        union = boxes1_volumes + boxes2_volumes - intersection
        iou = intersection / (union + 1e-6)

        volumes_bound = torch.prod(
            torch.clamp(
                (torch.max(boxes1[..., 3:], boxes2[..., 3:]) - torch.min(boxes1[..., :3], boxes2[..., :3])), min=0.0
            ),
            -1,
        )  # N

        giou = iou - (volumes_bound - union) / (volumes_bound + 1e-6)

        return iou, giou


    def fit_gp_spp(self, coords_float_spp, feats_spp, b1_inds, b2_inds, intersect_inds):
        device = feats_spp.device

        # intersect_coords = coords_float_spp[intersect_inds]
        intersect_feats = feats_spp[intersect_inds]
        # intersect_centroid = intersect_coords.mean(0)

        # b1_coords = coords_float_spp[b1_inds]
        b1_feats = feats_spp[b1_inds]
        b1_feats_mean = torch.mean(b1_feats, dim=0)

        # b2_coords = coords_float_spp[b2_inds]
        b2_feats = feats_spp[b2_inds]
        b2_feats_mean = torch.mean(b2_feats, dim=0)

        b1_mutex_feats_offset1 = b1_feats - b1_feats_mean
        b1_mutex_feats_offset2 = b1_feats - b2_feats_mean
        b2_mutex_feats_offset1 = b2_feats - b1_feats_mean
        b2_mutex_feats_offset2 = b2_feats - b2_feats_mean
        b1_mutex_cat_offset = torch.cat((b1_feats, b1_mutex_feats_offset1, b1_mutex_feats_offset2), dim=-1)
        b2_mutex_cat_offset = torch.cat((b2_feats, b2_mutex_feats_offset1, b2_mutex_feats_offset2), dim=-1)

        train_x = torch.cat([b1_mutex_cat_offset, b2_mutex_cat_offset], dim=0)

        one_box_points_lab = torch.cat(
            [torch.zeros(b1_feats.shape[0], device=device), torch.ones(b2_feats.shape[0], device=device)], dim=0
        )
        b1_inter_feats_offset = intersect_feats - b1_feats_mean
        b2_inter_feats_offset = intersect_feats - b2_feats_mean
        intersect_feats_cat_offset = torch.cat((intersect_feats, b1_inter_feats_offset, b2_inter_feats_offset), dim=-1)
        f_pred = self.box_cls(intersect_feats_cat_offset)
        f_pred_softmax = f_pred.softmax(dim=-1)
        pred_labels = torch.argmax(f_pred_softmax, dim=-1)
        pred_probs_new = torch.where(pred_labels == 1, f_pred_softmax[:,1], f_pred_softmax[:, 0])
        one_box_points_pred = self.box_cls(train_x)
        return pred_probs_new, pred_labels, one_box_points_pred, one_box_points_lab

    def filter_bg_within_box(self, coords_float_spp, feats_spp, b_inds, box):
        b_coords = coords_float_spp[b_inds]   # b1, 3
        center_coord = (box[0:3] + box[3:]) / 2     # 3

        b_feats = feats_spp[b_inds]   # b1, d
        center_feat = torch.mean(b_feats, dim=0)   # d
        #.unsqueeze(0).repeat(b1_feats.shape[0], 1)  

        ins_extent = box[3:] - box[0:3]     # 3
        coord_dis = (torch.abs(b_coords - center_coord) / ins_extent).mean(-1)     # b1
        coord_dis_exp = torch.exp(-1.0 * coord_dis)     # b1
        feat_dis = torch.cosine_similarity(b_feats, center_feat, dim=-1)   # b1
        dis = coord_dis_exp * feat_dis    # b1
        threshold_coord = coord_dis_exp.mean(-1)
        threshold_feat = feat_dis.mean(-1)
        threshold = threshold_coord * threshold_feat

        uncertain_inds = torch.where(dis < threshold)
        pred_probs_new = dis[uncertain_inds]
        filter_inds = b_inds[uncertain_inds]
        
        return filter_inds, pred_probs_new

    def gen_pseudo_label_mlp(
        self, 
        coords_float,
        mask_feats,
        spp,
        instance_cls,
        instance_box,
        instance_box_volume,
        wall_box=[],
        wall_box_volume=[],
        instance_classes=18,
        dataset_name="scannetv2",
        ground_h=0.1,
        thresh_spp_occu=0.8,
    ):
        max_num = 1000000
        n_points = len(coords_float)
        n_fg_instances = len(instance_box)

        unique_spps, spp = torch.unique(spp, return_inverse=True)
        n_spps = len(unique_spps)

        gp_feats = mask_feats.float()

        min_range = torch.min(coords_float, dim=0)[0]
        max_range = torch.max(coords_float, dim=0)[0]
        floor_min_z = min_range[2]
        floor_max_z = floor_min_z + ground_h
        floor_box = torch.tensor([min_range[0], min_range[1], min_range[2], max_range[0], max_range[1], floor_max_z])[
            None, :
        ].to(
            coords_float.device
        )  # 1x6
        floor_box_volume = torch.prod(torch.clamp(floor_box[:, 3:] - floor_box[:, :3], min=0.001), dim=1)

        if len(wall_box) > 0:
            boxes = torch.cat([instance_box, wall_box, floor_box])
            boxes_cls = torch.cat(
                [
                    instance_cls,
                    torch.ones((len(wall_box) + 1), dtype=instance_cls.dtype, device=instance_cls.device)
                    * instance_classes,
                ],
                dim=0,
            )
            boxes_volume = torch.cat([instance_box_volume, wall_box_volume, floor_box_volume], dim=0)
        else:
            boxes = torch.cat([instance_box, floor_box])
            boxes_cls = torch.cat(
                [instance_cls, torch.ones(1, dtype=instance_cls.dtype, device=instance_cls.device) * instance_classes],
                dim=0,
            )
            boxes_volume = torch.cat([instance_box_volume, floor_box_volume], dim=0)

        n_boxes = len(boxes)

        bb_occupancy = self.is_within_bb_torch(
            coords_float[:, None, :], boxes[None, :, :3] - 0.005, boxes[None, :, 3:] + 0.005
        )  # N_points, N_box
        # num_BBs_per_point = bb_occupancy.long().sum(dim=1)

        coords_float_spp = torch_scatter.scatter(
            coords_float, spp[:, None].expand(-1, coords_float.shape[1]), dim=0, reduce="mean"
        )
        gp_feats_spp = torch_scatter.scatter(gp_feats, spp[:, None].expand(-1, gp_feats.shape[1]), dim=0, reduce="mean")

        bb_occupancy_spp = torch_scatter.scatter(
            bb_occupancy.float(), spp[:, None].expand(n_points, n_boxes), dim=0, reduce="mean"
        )  # n_spp, n_label
        bb_occupancy_spp = bb_occupancy_spp >= thresh_spp_occu
        n_bbs_per_spp = bb_occupancy_spp.long().sum(dim=1)

        inst_per_point = torch.ones(n_spps, dtype=torch.int, device=coords_float.device) * -100
        is_determined = torch.zeros(n_spps, dtype=torch.long, device=coords_float.device)
        ps_prob_label = torch.zeros(n_spps, dtype=torch.float, device=coords_float.device)

        # no box -> -1, 1 box -> corres

        corres_box_active = torch.nonzero(bb_occupancy_spp[(n_bbs_per_spp == 1)], as_tuple=True)[1]

        inst_per_point[(n_bbs_per_spp == 1)] = corres_box_active.int()  # NOTE: 1 box -> assign corres box
        ps_prob_label[(n_bbs_per_spp == 1)] = 1
        is_determined[(n_bbs_per_spp == 1)] = max_num

        # filter the low confidence background points within bounding box
        for b in range(n_boxes):
            #b_inds = torch.nonzero((inst_per_point == b) & (n_bbs_per_spp == 1)).view(-1)
            b_inds = torch.nonzero((inst_per_point == b) & (n_bbs_per_spp != 0)).view(-1)
            if len(b_inds) == 0:
                continue
            filter_inds, pred_probs_new = self.filter_bg_within_box(coords_float_spp, gp_feats_spp, b_inds, boxes[b])
            # inst_per_point[filter_inds] = -1
            if len(filter_inds) == 0:
                continue
            ps_prob_label[filter_inds] = pred_probs_new

        # inst_one_hot[:, torch.nonzero(num_BBs_per_point == 0).view(-1)] = 1
        # inst_one_hot_prob[:, (num_BBs_per_point == 0)] = 1
        inst_per_point[(n_bbs_per_spp == 0)] = -1  # NOTE: nobox -> background -1
        ps_prob_label[(n_bbs_per_spp == 0)] = 1
        is_determined[(n_bbs_per_spp == 0)] = max_num

        cross_box_iou, _ = self.batch_giou_cross(boxes, boxes)  # N_box, N_box
        cross_box_iou.fill_diagonal_(0.0)

        box_visited = torch.zeros(n_boxes, dtype=torch.bool, device=coords_float.device)

        one_box_points_preds = []
        one_box_points_labs = []

        #for b1 in tqdm(range(n_boxes)):
        for b1 in range(n_boxes):
            b1_ious = cross_box_iou[b1]

            overlap_cond = (b1_ious > 0.0001) & (box_visited == 0)
            overlap_inds = torch.nonzero(overlap_cond).view(-1).int()
            n_overlap_ = len(overlap_inds)

            if n_overlap_ == 0:
                box_visited[b1] = 1
                continue

            for b2 in overlap_inds:
                assert b1 != b2
                intersect_cond = (bb_occupancy_spp[:, b1] == 1) & (bb_occupancy_spp[:, b2] == 1)

                intersect_inds = torch.nonzero(intersect_cond).view(-1)
                num_intersect_points = len(intersect_inds)

                if num_intersect_points == 0:
                    continue

                if self.is_box1_in_box2(boxes[b1], boxes[b2], offset=0.1):
                    inst_per_point[intersect_inds] = b1
                    is_determined[intersect_inds] = max_num
                    ps_prob_label[intersect_inds] = 1
                    box_visited[b1] = 1
                    break

                if self.is_box1_in_box2(boxes[b2], boxes[b1], offset=0.1):
                    inst_per_point[intersect_inds] = b2
                    is_determined[intersect_inds] = max_num
                    ps_prob_label[intersect_inds] = 1
                    box_visited[b2] = 1
                    continue

                if b1_ious[b2] >= 0.6:
                    continue

                b1_inds = torch.nonzero((inst_per_point == b1) & (n_bbs_per_spp == 1)).view(-1)
                b2_inds = torch.nonzero((inst_per_point == b2) & (n_bbs_per_spp == 1)).view(-1)

                if len(b1_inds) == 0 or len(b2_inds) == 0:
                    continue

                # try:
                pred_probs_new, pred_labels, one_box_points_pred, one_box_points_lab = self.fit_gp_spp(
                    coords_float_spp, gp_feats_spp, b1_inds, b2_inds, intersect_inds
                )

                one_box_points_preds.append(one_box_points_pred)
                one_box_points_labs.append(one_box_points_lab)

                overwrite_cond = ps_prob_label[intersect_inds] < pred_probs_new

                if pred_labels.shape != torch.Size([]):
                    inst_per_point[intersect_inds[overwrite_cond][pred_labels[overwrite_cond] == 1]] = b2
                    inst_per_point[intersect_inds[overwrite_cond][pred_labels[overwrite_cond] == 0]] = b1
                    ps_prob_label[intersect_inds[overwrite_cond]] = pred_probs_new[overwrite_cond]
                    is_determined[intersect_inds[overwrite_cond]] = len(intersect_inds)

            box_visited[b1] = 1

        point_inds, box_inds = torch.nonzero(
            bb_occupancy_spp[(n_bbs_per_spp > 1) & (is_determined == 0), :], as_tuple=True
        )
        min_volume, argmin_volume = torch_scatter.scatter_min(boxes_volume[box_inds], point_inds, dim=0)
        # sum_volume = torch_scatter.scatter(boxes_volume[box_inds].float(), point_inds, dim=0, reduce="sum")

        corres_multibox_active = box_inds[argmin_volume]

        # inst_one_hot[corres_multibox_active, point_inds[argmin_volume]] = 1
        # inst_one_hot_prob[corres_multibox_active, point_inds[argmin_volume]] = 1.0

        inst_per_point[
            (n_bbs_per_spp > 1) & (is_determined == 0)
        ] = corres_multibox_active.int()  # NOTE: multibox -> assign box with the smallest volume
        ps_prob_label[(n_bbs_per_spp > 1) & (is_determined == 0)] = 1.0
        # ps_prob_label[(n_bbs_per_spp > 1) & (is_determined == 0)] = 1 - (min_volume / sum_volume).float()

        ps_semantic_label_spp = torch.ones(n_spps, dtype=torch.int, device=coords_float.device) * -100
        ps_instance_label_spp = torch.ones(n_spps, dtype=torch.int, device=coords_float.device) * -100

        ps_semantic_label_spp[inst_per_point >= 0] = boxes_cls[inst_per_point[inst_per_point >= 0].long()].int()
        ps_semantic_label_spp[inst_per_point == -1] = instance_classes

        ps_instance_label_spp[inst_per_point >= 0] = inst_per_point[inst_per_point >= 0]

        ps_instance_label_spp[ps_instance_label_spp >= n_fg_instances] = -100
        ps_semantic_label_spp[ps_instance_label_spp >= n_fg_instances] = instance_classes

        ps_semantic_label = ps_semantic_label_spp[spp]
        ps_instance_label = ps_instance_label_spp[spp]
        ps_prob_label = ps_prob_label[spp].float()

        if len(one_box_points_preds) != 0:
            one_box_points_preds = torch.cat(one_box_points_preds, dim=0)
            one_box_points_labs = torch.cat(one_box_points_labs, dim=0)
        else:
            one_box_points_preds = torch.tensor([]).to(coords_float.device)
            one_box_points_labs = torch.tensor([]).to(coords_float.device)

        return ps_semantic_label, ps_instance_label, ps_prob_label, one_box_points_preds, one_box_points_labs


    def forward(self, coord_float, point_feat, superpoint, instance_cls, instance_box, instance_box_volume,instance_classes):
        ps_semantic_label, ps_instance_label, ps_prob_label, one_box_points_preds, one_box_points_labs = self.gen_pseudo_label_mlp(
                coord_float, point_feat, superpoint, instance_cls, instance_box, instance_box_volume, instance_classes=instance_classes)
        return ps_semantic_label, ps_instance_label, ps_prob_label, one_box_points_preds, one_box_points_labs
