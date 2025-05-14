import torch
import torch.nn as nn
from torch_scatter import scatter_min, scatter_mean, scatter_max, scatter
import numpy as np
from torch.distributions import Normal, Independent
from torch.autograd import Variable

from .shape_util import get_medium_feats, get_proposal_feats

class GlobalCrossAttentionLayer(nn.Module):

    def __init__(self, d_model=256, nhead=8, dropout=0.0):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward(self, source, query, batch_offsets, attn_masks=None, pe=None):
        """
        source (B*N, d_model)
        batch_offsets List[int] (b+1)
        query Tensor (b, n_q, d_model)
        """
        B = len(batch_offsets) - 1
        outputs = []
        query = self.with_pos_embed(query, pe)
        for i in range(B):
            start_id = batch_offsets[i]
            end_id = batch_offsets[i + 1]
            k = v = source[start_id:end_id].unsqueeze(0)  # (1, n, d_model)
            if attn_masks:
                output, _ = self.attn(query[i].unsqueeze(0), k, v, attn_mask=attn_masks[i])  # (1, 100, d_model)
            else:
                output, _ = self.attn(query[i].unsqueeze(0), k, v)
            self.dropout(output)
            output = output + query[i]
            self.norm(output)
            outputs.append(output)
        outputs = torch.cat(outputs, dim=0)  # (b, 100, d_model)
        return outputs


class MediumCrossAttentionLayer(nn.Module):
    def __init__(self, d_model=256, nhead=8, dropout=0.0):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward(self, source, query, batch_offsets, attn_masks=None, pe=None):
        """
        source (B*N, d_model)
        batch_offsets List[int] (b+1)
        query Tensor (b, n_q, d_model)
        """
        B = len(batch_offsets) - 1
        outputs = []
        query = self.with_pos_embed(query, pe)
        for i in range(B):
            start_id = batch_offsets[i]
            end_id = batch_offsets[i + 1]
            k = v = source[start_id:end_id].unsqueeze(0)  # (1, n, d_model)
            if attn_masks:
                output, _ = self.attn(query[i].unsqueeze(0), k, v, attn_mask=attn_masks[i])  # (1, 100, d_model)
            else:
                output, _ = self.attn(query[i].unsqueeze(0), k, v)
            self.dropout(output)
            output = output + query[i]
            self.norm(output)
            outputs.append(output)
        outputs = torch.cat(outputs, dim=0)  # (b, 100, d_model)
        return outputs


class PropCrossAttentionLayer(nn.Module):

    def __init__(self, d_model=256, nhead=8, dropout=0.0):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward(self, source, query, batch_offsets, attn_masks=None, pe=None):
        """
        source (B*N, d_model)
        batch_offsets List[int] (b+1)
        query Tensor (b, n_q, d_model)
        """
        B = len(batch_offsets) - 1
        outputs = []
        query = self.with_pos_embed(query, pe)
        for i in range(B):
            start_id = batch_offsets[i]
            end_id = batch_offsets[i + 1]
            k = v = source[start_id:end_id].unsqueeze(0)  # (1, n, d_model)
            if attn_masks:
                output, _ = self.attn(query[i].unsqueeze(0), k, v, attn_mask=attn_masks[i])  # (1, 100, d_model)
            else:
                output, _ = self.attn(query[i].unsqueeze(0), k, v)
            self.dropout(output)
            output = output + query[i]
            self.norm(output)
            outputs.append(output)
        outputs = torch.cat(outputs, dim=0)  # (b, 100, d_model)
        return outputs
        
class SelfAttentionLayer(nn.Module):

    def __init__(self, d_model=256, nhead=8, dropout=0.0):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            d_model,
            nhead,
            dropout=dropout,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward(self, x, pe=None):
        """
        x Tensor (b, 100, c)
        """
        q = k = self.with_pos_embed(x, pe)
        output, _ = self.attn(q, k, x)
        output = self.dropout(output) + x
        output = self.norm(output)
        return output


class FFN(nn.Module):

    def __init__(self, d_model, hidden_dim, dropout=0.0, activation_fn='relu'):
        super().__init__()
        if activation_fn == 'relu':
            self.net = nn.Sequential(
                nn.Linear(d_model, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, d_model),
                nn.Dropout(dropout),
            )
        elif activation_fn == 'gelu':
            self.net = nn.Sequential(
                nn.Linear(d_model, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, d_model),
                nn.Dropout(dropout),
            )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        output = self.net(x)
        output = output + x
        output = self.norm(output)
        return output


class QueryDecoder(nn.Module):
    """
    in_channels List[int] (4,) [64,96,128,160]
    """

    def __init__(
        self,
        num_layer=6,
        num_query=100,
        num_class=18,
        in_channel=32,
        d_model=256,
        nhead=8,
        hidden_dim=1024,
        dropout=0.0,
        activation_fn='relu',
        iter_pred=False,
        attn_mask=False,
        pe=False,
    ):
        super().__init__()
        self.num_layer = num_layer
        self.num_query = num_query
        self.input_proj = nn.Sequential(nn.Linear(in_channel, d_model), nn.LayerNorm(d_model), nn.ReLU())
        self.query = nn.Embedding(num_query, d_model)
        if pe:
            self.pe = nn.Embedding(num_query, d_model)
        self.global_cross_attn_layers = nn.ModuleList([])
        self.medium_cross_layers = nn.ModuleList([])
        self.prop_cross_attn_layers = nn.ModuleList([])
        self.self_attn_layers = nn.ModuleList([])
        self.ffn_layers = nn.ModuleList([])
        for i in range(num_layer):
            self.global_cross_attn_layers.append(GlobalCrossAttentionLayer(d_model, nhead, dropout))
            self.medium_cross_layers.append(MediumCrossAttentionLayer(d_model, nhead, dropout))
            self.prop_cross_attn_layers.append(PropCrossAttentionLayer(d_model, nhead, dropout))
            self.self_attn_layers.append(SelfAttentionLayer(d_model, nhead, dropout))
            self.ffn_layers.append(FFN(d_model, hidden_dim, dropout, activation_fn))
        self.out_norm = nn.LayerNorm(d_model)
        self.out_cls = nn.Sequential(nn.Linear(d_model, d_model), nn.ReLU(), nn.Linear(d_model, num_class + 1))
        self.out_bbox = nn.Sequential(nn.Linear(d_model, d_model), nn.ReLU(), nn.Linear(d_model, 6))
        self.out_bbox_scales = nn.Sequential(nn.Linear(d_model, d_model), nn.ReLU(), nn.Linear(d_model, 1))
        self.x_mask = nn.Sequential(nn.Linear(in_channel, d_model), nn.ReLU(), nn.Linear(d_model, d_model))
        self.iter_pred = iter_pred
        self.attn_mask = attn_mask
        
    def get_mask(self, query, mask_feats, batch_offsets):
        pred_masks = []
        attn_masks = []
       
        for i in range(len(batch_offsets) - 1):
            start_id, end_id = batch_offsets[i], batch_offsets[i + 1]
            mask_feat = mask_feats[start_id:end_id]
            pred_mask = torch.einsum('nd,md->nm', query[i], mask_feat)
            if self.attn_mask:
                attn_mask = (pred_mask.sigmoid() < 0.5).bool()
                attn_mask[torch.where(attn_mask.sum(-1) == attn_mask.shape[-1])] = False
                attn_mask = attn_mask.detach()
                attn_masks.append(attn_mask)
            pred_masks.append(pred_mask)
        return pred_masks, attn_masks

    
    def prediction_head(self, query, mask_feats, batch_offsets):
        query = self.out_norm(query)    # (b, q, d_model)
        pred_labels = self.out_cls(query)
        pred_bboxs = self.out_bbox(query)   # (b, q, 6)
        pred_bbox_scales = self.out_bbox_scales(query)
        pred_masks, attn_masks = self.get_mask(query, mask_feats, batch_offsets)
        return pred_labels, pred_masks, attn_masks, pred_bboxs, pred_bbox_scales
    
    def reparametrize(self, mu, sigma):
        std = sigma.mul(0.5).exp_()
        eps = torch.cuda.FloatTensor(std.size()).normal_()
        eps = Variable(eps)
        return eps.mul(std).add_(mu)
    
    def forward_iter_pred(self, x, batch_offsets, sp_coords):
        """
        x [B*M, inchannel]
        """
        prediction_labels = []
        prediction_masks = []
        prediction_bbox_scales = []
        prediction_bboxs = []
        inst_feats = self.input_proj(x)
        mask_feats = self.x_mask(x)
        B = len(batch_offsets) - 1
        query = self.query.weight.unsqueeze(0).repeat(B, 1, 1)  # (b, n, d_model)
        if getattr(self, 'pe', None):
            pe = self.pe.weight.unsqueeze(0).repeat(B, 1, 1)
        else:
            pe = None
        pred_labels, pred_masks, attn_masks, pred_bboxs, pred_bbox_scales = self.prediction_head(query, mask_feats, batch_offsets)
        medium_feats = get_medium_feats(pred_bboxs, sp_coords, inst_feats, pred_masks, batch_offsets)
        prop_feats = get_proposal_feats(pred_bboxs, sp_coords, inst_feats, pred_bbox_scales, batch_offsets)
        prediction_labels.append(pred_labels)
        prediction_masks.append(pred_masks)
        prediction_bbox_scales.append(pred_bbox_scales)
        prediction_bboxs.append(pred_bboxs)
        for i in range(self.num_layer):
            query = self.global_cross_attn_layers[i](inst_feats, query, batch_offsets, attn_masks, pe)
            query = self.medium_cross_layers[i](medium_feats, query, batch_offsets, pe)
            query = self.prop_cross_attn_layers[i](prop_feats, query, batch_offsets, pe)
            query = self.self_attn_layers[i](query, pe)
            query = self.ffn_layers[i](query)
            pred_labels, pred_masks, attn_masks, pred_bboxs, pred_bbox_scales = self.prediction_head(query, mask_feats, batch_offsets)
            
            if i != self.num_layer-1:
                medium_feats = get_medium_feats(pred_bboxs, sp_coords, inst_feats, pred_masks, batch_offsets)
                prop_feats = get_proposal_feats(pred_bboxs, sp_coords, inst_feats, pred_bbox_scales, batch_offsets)

            prediction_labels.append(pred_labels)
            prediction_masks.append(pred_masks)
            prediction_bbox_scales.append(pred_bbox_scales)
            prediction_bboxs.append(pred_bboxs)
        '''
        return {
            'labels':
            pred_labels,
            'masks':
            pred_masks,
            'bbox_scales':
            pred_bbox_scales,
            'bboxs':
            pred_bboxs,
            'aux_outputs': [{
                'labels': a,
                'masks': b,
                'bbox_scales': c,
                'bboxs': d,
            } for a, b, c, d in zip(
                prediction_labels[:-1],
                prediction_masks[:-1],
                prediction_bbox_scales[:-1],
                prediction_bboxs[:-1],
            )],
        }
        '''
        return {
            'labels':
            prediction_labels[1],
            'masks':
            prediction_masks[1],
            'bbox_scales':
            prediction_bbox_scales[1],
            'bboxs':
            prediction_bboxs[1],
            'aux_outputs': [{
                'labels': a,
                'masks': b,
                'bbox_scales': c,
                'bboxs': d,
            } for a, b, c, d in zip(
                prediction_labels[:-1],
                prediction_masks[:-1],
                prediction_bbox_scales[:-1],
                prediction_bboxs[:-1],
            )],
        }
    def forward(self, x, batch_offsets, sp_coords):
        if self.iter_pred:
            return self.forward_iter_pred(x, batch_offsets, sp_coords)
