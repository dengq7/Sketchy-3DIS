import glob
import math
import numpy as np
import os.path as osp
import pointgroup_ops
import scipy
import scipy.interpolate as interpolate
import scipy.ndimage as ndimage
import torch
import torch_scatter
from torch.utils.data import Dataset
from typing import Dict, Sequence, Tuple, Union

from ..utils import Instances3D


class ScanNetDataset(Dataset):

    CLASSES = ('cabinet', 'bed', 'chair', 'sofa', 'table', 'door', 'window', 'bookshelf', 'picture', 'counter', 'desk',
               'curtain', 'refrigerator', 'shower curtain', 'toilet', 'sink', 'bathtub', 'otherfurniture')
    NYU_ID = (3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39)

    def __init__(self,
                 data_root,
                 prefix,
                 suffix,
                 voxel_cfg=None,
                 training=True,
                 with_label=True,
                 mode=4,
                 with_elastic=True,
                 use_xyz=True,
                 logger=None,
                 with_normal=True,
                 lamda=-10):
        self.data_root = data_root
        self.prefix = prefix
        self.suffix = suffix
        self.voxel_cfg = voxel_cfg
        self.training = training
        self.with_label = with_label
        self.mode = mode
        self.with_elastic = with_elastic
        self.use_xyz = use_xyz
        self.logger = logger
        self.filenames = self.get_filenames()
        self.logger.info(f'Load {self.prefix} dataset: {len(self.filenames)} scans')
        self.with_normal = with_normal
        self.lamda = lamda
    '''
    def get_filenames(self):
        filenames = glob.glob(osp.join(self.data_root, self.prefix, '*' + self.suffix))
        assert len(filenames) > 0, 'Empty dataset.'
        filenames = sorted(filenames)
        # filenames = filenames[:12]
        return filenames
    '''
    def get_filenames(self):
        if self.prefix == 'trainval':
            filenames = glob.glob(osp.join(self.data_root, "train", '*' + self.suffix)) + \
                glob.glob(osp.join(self.data_root, "val", '*' + self.suffix))
        else:
            filenames = glob.glob(osp.join(self.data_root, self.prefix, '*' + self.suffix))
        assert len(filenames) > 0, 'Empty dataset.'
        filenames = sorted(filenames)
        # filenames = filenames[:12]
        return filenames

    def load(self, filename):
        scan_id = osp.basename(filename).replace(self.suffix, "")
        spp_filename = osp.join(self.data_root, "superpoints", scan_id + ".pth")
        superpoint = torch.load(spp_filename)

        if self.with_normal:
            normal = torch.load(self.data_root, "normals", scan_id + ".pth")
        else:
            normal = None

        if self.with_label:
            coords, colors, dummy_sem_label, dummy_inst_label = torch.load(filename)
        else:
            
            xyz, rgb, superpoint = torch.load(filename)
            dummy_sem_label = np.zeros(xyz.shape[0], dtype=np.float32)
            dummy_inst_label = np.zeros(xyz.shape[0], dtype=np.float32)
        return xyz, rgb, superpoint, dummy_sem_label, dummy_inst_label, normal

    def __len__(self):
        return len(self.filenames)

    def transform_train(self, xyz, rgb, superpoint, semantic_label, instance_label, normal=None):
        xyz_middle, normal = self.data_aug(xyz, True, True, True, normal)
        rgb += np.random.randn(3) * 0.1
        xyz = xyz_middle * self.voxel_cfg.scale
        if self.with_elastic:
            xyz = self.elastic(xyz, 6, 40.)
            xyz = self.elastic(xyz, 20, 160.)
        xyz = xyz - xyz.min(0)
        xyz, valid_idxs = self.crop(xyz)
        xyz_middle = xyz_middle[valid_idxs]
        xyz = xyz[valid_idxs]
        rgb = rgb[valid_idxs]
        semantic_label = semantic_label[valid_idxs]
        superpoint = np.unique(superpoint[valid_idxs], return_inverse=True)[1]
        instance_label = self.get_cropped_inst_label(instance_label, valid_idxs)
        if normal is not None:
            normal = normal[valid_idxs]
        return xyz, xyz_middle, rgb, superpoint, semantic_label, instance_label, normal

    def transform_test(self, xyz, rgb, superpoint, semantic_label=None, instance_label=None, normal=None):
        xyz_middle = xyz
        xyz = xyz_middle * self.voxel_cfg.scale
        xyz -= xyz.min(0)
        valid_idxs = np.ones(xyz.shape[0], dtype=bool)
        superpoint = np.unique(superpoint[valid_idxs], return_inverse=True)[1]
        if instance_label is not None:
            instance_label = self.get_cropped_inst_label(instance_label, valid_idxs)
        return xyz, xyz_middle, rgb, superpoint, semantic_label, instance_label, normal

    def data_aug(self, xyz, jitter=False, flip=False, rot=False, normal=None):
        m = np.eye(3)
        if jitter:
            m += np.random.randn(3, 3) * 0.1
        if flip:
            m[0][0] *= np.random.randint(0, 2) * 2 - 1  # flip x randomly
        if rot:
            theta = np.random.rand() * 2 * math.pi
            m = np.matmul(
                m,
                [[math.cos(theta), math.sin(theta), 0], [-math.sin(theta), math.cos(theta), 0], [0, 0, 1]])  # rotation
        if normal is not None:
            normal = np.matmul(normal, m)
        return np.matmul(xyz, m), normal

    def crop(self, xyz: np.ndarray) -> Union[np.ndarray, np.ndarray]:
        r"""
        crop the point cloud to reduce training complexity

        Args:
            xyz (np.ndarray, [N, 3]): input point cloud to be cropped

        Returns:
            Union[np.ndarray, np.ndarray]: processed point cloud and boolean valid indices
        """
        xyz_offset = xyz.copy()
        valid_idxs = xyz_offset.min(1) >= 0
        assert valid_idxs.sum() == xyz.shape[0]

        full_scale = np.array([self.voxel_cfg.spatial_shape[1]] * 3)
        room_range = xyz.max(0) - xyz.min(0)
        while valid_idxs.sum() > self.voxel_cfg.max_npoint:
            offset = np.clip(full_scale - room_range + 0.001, None, 0) * np.random.rand(3)
            xyz_offset = xyz + offset
            valid_idxs = (xyz_offset.min(1) >= 0) * ((xyz_offset < full_scale).sum(1) == 3)
            full_scale[:2] -= 32

        return xyz_offset, valid_idxs

    def elastic(self, xyz, gran, mag):
        """Elastic distortion (from point group)

        Args:
            xyz (np.ndarray): input point cloud
            gran (float): distortion param
            mag (float): distortion scalar

        Returns:
            xyz: point cloud with elastic distortion
        """
        blur0 = np.ones((3, 1, 1)).astype('float32') / 3
        blur1 = np.ones((1, 3, 1)).astype('float32') / 3
        blur2 = np.ones((1, 1, 3)).astype('float32') / 3

        bb = np.abs(xyz).max(0).astype(np.int32) // gran + 3
        noise = [np.random.randn(bb[0], bb[1], bb[2]).astype('float32') for _ in range(3)]
        noise = [ndimage.filters.convolve(n, blur0, mode='constant', cval=0) for n in noise]
        noise = [ndimage.filters.convolve(n, blur1, mode='constant', cval=0) for n in noise]
        noise = [ndimage.filters.convolve(n, blur2, mode='constant', cval=0) for n in noise]
        noise = [ndimage.filters.convolve(n, blur0, mode='constant', cval=0) for n in noise]
        noise = [ndimage.filters.convolve(n, blur1, mode='constant', cval=0) for n in noise]
        noise = [ndimage.filters.convolve(n, blur2, mode='constant', cval=0) for n in noise]
        ax = [np.linspace(-(b - 1) * gran, (b - 1) * gran, b) for b in bb]
        interp = [interpolate.RegularGridInterpolator(ax, n, bounds_error=0, fill_value=0) for n in noise]

        def g(xyz_):
            return np.hstack([i(xyz_)[:, None] for i in interp])

        return xyz + g(xyz) * mag

    def get_cropped_inst_label(self, instance_label: np.ndarray, valid_idxs: np.ndarray) -> np.ndarray:
        r"""
        get the instance labels after crop operation and recompact

        Args:
            instance_label (np.ndarray, [N]): instance label ids of point cloud
            valid_idxs (np.ndarray, [N]): boolean valid indices

        Returns:
            np.ndarray: processed instance labels
        """
        instance_label = instance_label[valid_idxs]
        j = 0
        while j < instance_label.max():
            if len(np.where(instance_label == j)[0]) == 0:
                instance_label[instance_label == instance_label.max()] = j
            j += 1
        return instance_label

    def get_gt_instance3D(self, instance_label, semantic_label, superpoint, scan_id, coord_float):
        num_insts = instance_label.max().item() + 1
        num_points = len(instance_label)
        gt_labels, instance_boxes = [], []
        instance_box_volume = []

        gt_inst = torch.zeros(num_points, dtype=torch.int64)
        for i in range(num_insts):
            idx = torch.where(instance_label == i)
            assert len(torch.unique(semantic_label[idx])) == 1
            sem_id = semantic_label[idx][0]
            if sem_id == -100:
                # sem_id = 1
                # gt_inst[idx] = sem_id * 1000 + i + 1
                continue
            gt_label = sem_id
            gt_labels.append(gt_label)

            coord_float_ = coord_float[idx]

            min_xyz_i = coord_float_.min(0)[0]
            max_xyz_i = coord_float_.max(0)[0]
            
            '''
            coord_extent = 0.1 * (max_xyz_i - min_xyz_i)
            ## S1: scaled
            max_xyz_i = max_xyz_i + 0.5 * coord_extent
            min_xyz_i = min_xyz_i - 0.5 * coord_extent
            '''
            '''
            ## S2:  scaled + translated
            max_xyz_i = max_xyz_i + 1.0 * coord_extent
            '''
            '''
            ## S3:  scaled + rotated
            max_xyz_i = max_xyz_i + 0.5 * coord_extent
            min_xyz_i = min_xyz_i - 0.5* coord_extent
            axis = ((max_xyz_i + min_xyz_i)/2).cpu().numpy()
            radian = 5 * math.pi/180
            rot_matrix = scipy.linalg.expm(np.cross(np.eye(3), axis / scipy.linalg.norm(axis) * radian))
            min_xyz_i = np.matmul(min_xyz_i, rot_matrix)
            max_xyz_i = np.matmul(max_xyz_i, rot_matrix)
            '''
            '''
            ## S4:  scaled + translated + rotated
            max_xyz_i = max_xyz_i + 0.5 * coord_extent
            axis = ((max_xyz_i + min_xyz_i)/2).cpu().numpy()
            radian = 5 * math.pi/180
            rot_matrix = scipy.linalg.expm(np.cross(np.eye(3), axis / scipy.linalg.norm(axis) * radian))
            min_xyz_i = np.matmul(min_xyz_i, rot_matrix)
            max_xyz_i = np.matmul(max_xyz_i, rot_matrix)
            '''

            # if sem_label == -100: continue
            instance_boxes.append(torch.cat([min_xyz_i, max_xyz_i], dim=0))
            coord_extent = torch.clip((max_xyz_i - min_xyz_i), min=0.0, max=None)
            instance_box_volume.append(torch.prod(coord_extent))
        if instance_boxes:
            instance_boxes = torch.stack(instance_boxes, dim=0)
            instance_box_volume = torch.stack(instance_box_volume, dim=0)
        else:
            instance_boxes = torch.tensor([])
            instance_box_volume = torch.tensor([])
        gt_labels = torch.tensor(gt_labels)

        inst = Instances3D(num_points, gt_instances=gt_inst.numpy())
        inst.gt_labels = gt_labels.long()
        inst.gt_boxes = instance_boxes.float()
        inst.instance_box_volume = instance_box_volume.float()
        return inst

    def __getitem__(self, index: int) -> Tuple:
        filename = self.filenames[index]
        scan_id = osp.basename(filename).replace(self.suffix, '')
        data = self.load(filename)
        
        data = self.transform_train(*data) if self.training else self.transform_test(*data)
        xyz, xyz_middle, rgb, superpoint, semantic_label, instance_label, normal = data
        '''
        xyz, rgb, superpoint, semantic_label, instance_label, normal = data
        xyz_middle = xyz
        '''
        coord = torch.from_numpy(xyz).long()
        coord_float = torch.from_numpy(xyz_middle).float()
        feat = torch.from_numpy(rgb).float()
        superpoint = torch.from_numpy(superpoint)
        if normal is not None:
            normal = torch.from_numpy(normal).float()
        semantic_label = torch.from_numpy(semantic_label).long()
        semantic_label = torch.where(semantic_label < 2, -100, semantic_label - 2)
        instance_label = torch.from_numpy(instance_label).long()
        gt_inst = self.get_gt_instance3D(instance_label, semantic_label, superpoint, scan_id, coord_float)
        return scan_id, coord, coord_float, feat, superpoint, gt_inst, normal

    def collate_fn(self, batch: Sequence[Sequence]) -> Dict:
        scan_ids, coords, coords_float, feats, superpoints, insts = [], [], [], [], [], []
        batch_offsets = [0]
        superpoint_bias = 0
        normals = []
        point_batch_offsets = [0]
        point_superpoint_bias = 0
        superpoints_list = []

        for i, data in enumerate(batch):
            scan_id, coord, coord_float, feat, superpoint, inst, normal = data

            superpoints_list.append(superpoint)

            superpoint += superpoint_bias
            superpoint_bias = superpoint.max().item() + 1
            batch_offsets.append(superpoint_bias)

            point_superpoint_bias += coord.shape[0]
            point_batch_offsets.append(point_superpoint_bias)

            scan_ids.append(scan_id)
            coords.append(torch.cat([torch.LongTensor(coord.shape[0], 1).fill_(i), coord], 1))
            coords_float.append(coord_float)
            feats.append(feat)
            superpoints.append(superpoint)
            insts.append(inst)
            normals.append(normal)

        # merge all scan in batch
        batch_offsets = torch.tensor(batch_offsets, dtype=torch.int)  # int [B+1]
        point_batch_offsets = torch.tensor(point_batch_offsets, dtype=torch.int)  # int [B+1]
        coords = torch.cat(coords, 0)  # long [B*N, 1 + 3], the batch item idx is put in b_xyz[:, 0]
        coords_float = torch.cat(coords_float, 0)  # float [B*N, 3]
        feats = torch.cat(feats, 0)  # float [B*N, 3]
        superpoints = torch.cat(superpoints, 0).long()  # long [B*N, ]
        if self.use_xyz:
            feats = torch.cat((feats, coords_float), dim=1)
        
        if self.with_normal:
            normals = torch.cat(normals, dim=0)
            feats = torch.cat([feats, normals], dim=1)

        # voxelize
        spatial_shape = np.clip((coords.max(0)[0][1:] + 1).numpy(), self.voxel_cfg.spatial_shape[0], None)  # long [3]
        voxel_coords, p2v_map, v2p_map = pointgroup_ops.voxelization_idx(coords, len(batch), self.mode)

        return {
            'scan_ids': scan_ids,
            'voxel_coords': voxel_coords,
            'p2v_map': p2v_map,
            'v2p_map': v2p_map,
            'spatial_shape': spatial_shape,
            'feats': feats,
            'superpoints': superpoints,
            'batch_offsets': batch_offsets,
            'point_batch_offsets': point_batch_offsets,
            'coords_float': coords_float,
            'superpoints_list': superpoints_list,
            'insts': insts,
        }
