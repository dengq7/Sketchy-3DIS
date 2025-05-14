import torch


def get_soft_mask_and_weights(pred_hard_bbox, sp_coords, pred_masks, batch_offsets):
    pred_soft_masks = []
    pred_soft_weights = []
    for i in range(len(batch_offsets) - 1):
        start_id, end_id = batch_offsets[i], batch_offsets[i + 1]
        sp_coord = sp_coords[start_id:end_id].unsqueeze(0).repeat(pred_hard_bbox[i].shape[0], 1, 1)      # (q, M, 3)
        pred_bboxs_coord =  pred_hard_bbox[i]        # (q, 6)
        pred_bboxs_min = pred_bboxs_coord[:, 0:3]    # (q, 3)
        pred_bboxs_max = pred_bboxs_coord[:, 3:]     # (q, 3)

        pred_bboxs_min_coord = pred_bboxs_min.unsqueeze(1).repeat(1, sp_coord.shape[1], 1)    # (q, M, 3)
        pred_bboxs_max_coord = pred_bboxs_max.unsqueeze(1).repeat(1, sp_coord.shape[1], 1)     # (q, M, 3)

        coord_dis1 = pred_bboxs_min_coord -  sp_coord     # (q, M, 3)
        coord_dis2 = sp_coord -  pred_bboxs_max_coord     # (q, M, 3)
        coord_dis = coord_dis1 * coord_dis2     # (q, M, 3)
        pred_bbox_mask = torch.eq(torch.mean(torch.ge(coord_dis, 0.).float(), dim=-1), 1.0).float()     # (q, M)
        
        pred_mask_binary = (pred_masks[i].sigmoid() >= 0.5).float()
        pred_bbox_mask_binary = (pred_bbox_mask > 0.5).float()

        intersection = (pred_mask_binary * pred_bbox_mask_binary).sum(-1)
        union = pred_bbox_mask_binary.sum(-1) + pred_mask_binary.sum(-1) - intersection
        score = intersection / (union + 1e-6)
        iou_scores = torch.exp(1.0 * score)        # (q, )
        #normalized
        norm_iou_scores = iou_scores / iou_scores.mean()  # (q, )
        norm_iou_scores = norm_iou_scores.unsqueeze(1).repeat(1, pred_masks[i].shape[1])
        pred_soft_mask = pred_masks[i] * norm_iou_scores * pred_bbox_mask  # (q, M)
        pred_soft_mask = torch.where(torch.le(pred_soft_mask, 0.), -20.0, pred_soft_mask.double())      # (q, M)
        pred_soft_masks.append(pred_soft_mask)      # (b, q, M)
        
        pred_soft_mask_binary = (pred_soft_mask.sigmoid() > 0.5).bool()

        boundary_points_indexes = torch.eq(torch.eq(pred_soft_mask_binary, pred_mask_binary).float(), 0.).float()   # (q, M)
        points2minbox_dis, _ =  torch.min(torch.abs(coord_dis1), dim=-1)    # (q, M)
        points2maxbox_dis, _ =  torch.min(torch.abs(coord_dis2), dim=-1)    # (q, M)
        points2box_dis = torch.where(torch.le(points2minbox_dis, points2maxbox_dis), points2minbox_dis, points2maxbox_dis)     # (q, M)
        points2box_exp = torch.exp(-1.0 * points2box_dis)
        pred_soft_weight = points2box_exp * boundary_points_indexes       # (q, M)
        pred_soft_weight = pred_soft_weight + 1.0       # (q, M)
        pred_soft_weights.append(pred_soft_weight)  # (b, q, M)
    return pred_soft_masks, pred_soft_weights


def get_proposal_feats(pred_hard_bbox, sp_coords, sp_feats, pred_bbox_scales, batch_offsets):
    """
    pred_hard_bbox: [bsz, num_queries, 6]
    sp_coords: [N, 3]
    sp_feats: [N, d_model]
    pred_bbox_scales: [bsz, num_queries, 1]
    """
    prop_feats = sp_feats.new_zeros(sp_feats.shape)     # [N, d_model]
    pred_bbox_scales = pred_bbox_scales.repeat(1, 1, sp_coords.shape[1])    #[bsz, num_queries, 3]
    for i in range(len(batch_offsets) - 1):
        start_id, end_id = batch_offsets[i], batch_offsets[i + 1]
        sp_coord = sp_coords[start_id:end_id].unsqueeze(0).repeat(pred_hard_bbox.shape[1], 1, 1)      # (num_queries, N, 3)
        pred_bboxs_coord =  pred_hard_bbox[i]        # (q, 6)
        pred_bboxs_min = pred_bboxs_coord[:, 0:3]    # (q, 3)
        pred_bboxs_max = pred_bboxs_coord[:, 3:]     # (q, 3)

        # get features by scaled bbox
        pred_center_by_bboxs = (pred_bboxs_min + pred_bboxs_max) / 2.0     # (q, 3)
        scaled_bboxs_min = pred_center_by_bboxs - (pred_center_by_bboxs - pred_bboxs_min) * pred_bbox_scales[i]       # (q, 3)
        scaled_bboxs_maxd = pred_center_by_bboxs + (pred_bboxs_max - pred_center_by_bboxs) * pred_bbox_scales[i]       # (q, 3)
        scaled_bboxs_min_coord = scaled_bboxs_min.unsqueeze(1).repeat(1, sp_coord.shape[1], 1)    # (q, M, 3)
        scaled_bboxs_max_coord = scaled_bboxs_maxd.unsqueeze(1).repeat(1, sp_coord.shape[1], 1)    # (q, M, 3)
        scaled_coord_dis1 = scaled_bboxs_min_coord -  sp_coord     # (q, M, 3)
        scaled_coord_dis2 = sp_coord -  scaled_bboxs_max_coord     # (q, M, 3)
        scaled_coord_dis = scaled_coord_dis1 * scaled_coord_dis2     # (q, M, 3)
        scaled_pred_bbox_mask = torch.eq(torch.mean(torch.ge(scaled_coord_dis, 0.).float(), dim=-1), 1.0).float()     # (q, M)

        scaled_pred_bbox_mask_binary = (scaled_pred_bbox_mask > 0.5).float()  # (q, M)
        scaled_pred_bbox_mask_binary = scaled_pred_bbox_mask_binary.unsqueeze(2).repeat(1, 1, sp_feats.shape[1]) # (q, M, d_model)
        scaled_sp_feats = sp_feats[start_id:end_id].unsqueeze(0).repeat(scaled_pred_bbox_mask_binary.shape[0], 1, 1)   # (q, M, d_model)
        scaled_bbox_sp_feats_binary = scaled_sp_feats * scaled_pred_bbox_mask_binary   # (q, M, d_model)
        scaled_sp_feats = torch.mean(scaled_bbox_sp_feats_binary, dim=0)    # (M, d_model)
        prop_feats[start_id:end_id] = scaled_sp_feats  # [N, d_model]
    return prop_feats


def get_medium_feats(pred_hard_bbox, sp_coords, sp_feats, pred_masks, batch_offsets):
    """
    pred_hard_bbox: [bsz, num_queries, 6]
    sp_coords: [N, 3]
    sp_feats: [N, d_model]
    """
    medium_feats = sp_feats.new_zeros(sp_feats.shape)     # [N, d_model]
    for i in range(len(batch_offsets) - 1):
        start_id, end_id = batch_offsets[i], batch_offsets[i + 1]
        sp_coord = sp_coords[start_id:end_id].unsqueeze(0).repeat(pred_hard_bbox.shape[1], 1, 1)      # (num_queries, M, 3)
        pred_bboxs_coord =  pred_hard_bbox[i]        # (q, 6)
        pred_bboxs_min = pred_bboxs_coord[:, 0:3]    # (q, 3)
        pred_bboxs_max = pred_bboxs_coord[:, 3:]     # (q, 3)

        pred_bboxs_min_coord = pred_bboxs_min.unsqueeze(1).repeat(1, sp_coord.shape[1], 1)    # (q, M, 3)
        pred_bboxs_max_coord = pred_bboxs_max.unsqueeze(1).repeat(1, sp_coord.shape[1], 1)     # (q, M, 3)

        coord_dis1 = pred_bboxs_min_coord -  sp_coord     # (q, M, 3)
        coord_dis2 = sp_coord -  pred_bboxs_max_coord     # (q, M, 3)
        coord_dis = coord_dis1 * coord_dis2     # (q, M, 3)
        pred_bbox_mask = torch.eq(torch.mean(torch.ge(coord_dis, 0.).float(), dim=-1), 1.0).float()     # (q, M)
        
        pred_mask = pred_masks[i]   # (M)

        pred_mask_binary = (pred_mask.sigmoid() >= 0.5).float()
        pred_bbox_mask_binary = (pred_bbox_mask > 0.5).bool()

        intersection = (pred_mask_binary * pred_bbox_mask_binary).sum(-1)
        union = pred_bbox_mask_binary.sum(-1) + pred_mask_binary.sum(-1) - intersection
        score = intersection / (union + 1e-6)
        iou_scores = torch.exp(1.0 * score)        # (q, )
        #normalized
        norm_iou_scores = iou_scores / iou_scores.mean()  # (q, )
        norm_iou_scores = norm_iou_scores.unsqueeze(1).repeat(1, pred_mask.shape[1])    # (q, M)

        #pred_soft_mask = pred_mask * norm_iou_scores * pred_bbox_mask  # (q, M)
        pred_soft_mask = pred_mask * norm_iou_scores  # (q, M)
        #pred_soft_mask = torch.where(torch.le(pred_soft_mask, 0.), -20.0, pred_soft_mask.double())      # (q, M)
        
        pred_soft_mask_binary = (pred_soft_mask.sigmoid() > 0.5).bool()    # (q, M)

        medium_mask_binary = pred_bbox_mask_binary | pred_soft_mask_binary       # (q, M)
        medium_mask_binary = medium_mask_binary.unsqueeze(2).repeat(1, 1, sp_feats.shape[1]) # (q, M, d_model)
        med_feat = sp_feats[start_id:end_id].unsqueeze(0).repeat(medium_mask_binary.shape[0], 1, 1)   # (q, M, d_model)
        med_feat_binary = med_feat * medium_mask_binary   # (q, M, d_model)
        med_feat = torch.mean(med_feat_binary, dim=0)    # (M, d_model)
        medium_feats[start_id:end_id] = med_feat  # [N, d_model]
    return medium_feats
