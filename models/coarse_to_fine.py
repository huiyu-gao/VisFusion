import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsparse.tensor import PointTensor
from loguru import logger

from models.modules import SPVCNN
from .local_fusion import VisibleFusion
from .sparsify import sparsify
from .global_fusion import GlobalFusion
from utils import apply_log_transform


class ReconNet(nn.Module):
    def __init__(self, cfg):
        super(ReconNet, self).__init__()
        self.cfg = cfg
        self.n_scales = len(cfg.THRESHOLDS) - 1

        alpha = int(self.cfg.BACKBONE2D.ARC.split('-')[-1])
        ch_in = [80 * alpha + 1, 96 + 40 * alpha + 2 + 1, 48 + 24 * alpha + 2 + 1, 24 + 24 + 2 + 1]
        channels = [96, 48, 24]

        self.visible_fusion = VisibleFusion(cfg)
        # fuse to global
        if cfg.FUSION.FUSION_ON:
            self.gru_fusion = GlobalFusion(cfg, channels)
            self.global_tsdf = GlobalFusion(cfg, direct_substitute=True)

        self.sp_convs = nn.ModuleList()
        self.local_tsdf_preds = nn.ModuleList()
        self.local_occ_preds = nn.ModuleList()
        self.global_tsdf_preds = nn.ModuleList()
        self.global_occ_preds = nn.ModuleList()

        for i in range(len(cfg.THRESHOLDS)):
            self.sp_convs.append(SPVCNN(num_classes=1,
                                        in_channels=ch_in[i],
                                        pres=1,
                                        cr=1 / 2 ** i,
                                        vres=self.cfg.VOXEL_SIZE * 2 ** (self.n_scales - i),
                                        dropout=self.cfg.SPARSEREG.DROPOUT))
            self.local_tsdf_preds.append(nn.Linear(channels[i], 1))
            self.local_occ_preds.append(nn.Linear(channels[i], 1))
            self.global_tsdf_preds.append(nn.Linear(channels[i], 1))
            self.global_occ_preds.append(nn.Linear(channels[i], 1))

    def generate_grid(self, n_vox, interval):
        """ Create voxel grid.

        Args:
            n_vox: (list), number of voxels along axis x, y, z, dim: [n_x, x_y, x_z]
            interval: (int), interval to generate grid

        Returns:
            grid: (Tensor), generated grid, dim: (1, 3, (n_x/interval)*(n_y/interval)*(n_z/interval))
        """

        with torch.no_grad():
            grid_range = [torch.arange(0, n_vox[axis], interval, dtype=torch.float, device='cuda:0') for axis in range(3)]
            grid = torch.stack(torch.meshgrid(grid_range[0], grid_range[1], grid_range[2]))
            grid = grid.unsqueeze(0)
            grid = grid.view(1, 3, -1)

        return grid

    def upsample(self, pre_feat, pre_coords, interval, num=8):
        """ Upsample coords and features.

        Args:
            pre_feat: (Tensor), features from last level, dim: (num of voxels, C)
            pre_coords: (Tensor), coordinates from last level, dim: (num of voxels, 4) (4 : Batch ind, x, y, z)
            interval: (int), interval of voxels, interval = scale ** 2
            num: (int), 1 -> 8

        Returns:
            up_feat: (Tensor), upsampled features, dim: (num of voxels * 8, C)
            up_coords: (Tensor), upsampled coordinates, dim: (num of voxels * 8, 4) (4 : Batch ind, x, y, z)
        """

        with torch.no_grad():
            pos_list = [1, 2, 3, [1, 2], [1, 3], [2, 3], [1, 2, 3]]
            n, c = pre_feat.shape
            up_feat = pre_feat.unsqueeze(1).expand(-1, num, -1).contiguous()
            up_coords = pre_coords.unsqueeze(1).repeat(1, num, 1).contiguous()
            for i in range(num - 1):
                up_coords[:, i + 1, pos_list[i]] += interval

            up_feat = up_feat.view(-1, c)
            up_coords = up_coords.view(-1, 4)

        return up_feat, up_coords

    def get_target(self, coords, inputs, scale):
        """ Get ground truth TSDF and occupancy for voxels.

        Args:
            coords: (Tensor), coordinates of voxels, dim: (num of voxels, 4) (4 : Batch ind, x, y, z)
            inputs: (list), inputs['tsdf_list'/'occ_list']: ground truth volume list, dim: [(B, DIM_X, DIM_Y, DIM_Z)]
            scale: (int), scale for voxel size

        Returns:
            tsdf_target: (Tensor), TSDF ground truth for each predicted voxels, dim: (num of voxels,)
            occ_target: (Tensor), occupancy ground truth for each predicted voxels, dim: (num of voxels,)
        """

        with torch.no_grad():
            tsdf_target = inputs['tsdf_list'][scale]
            occ_target = inputs['occ_list'][scale]
            coords_down = coords.detach().clone().long()
            coords_down[:, 1:] = torch.div(coords[:, 1:], 2 ** scale, rounding_mode='trunc')
            tsdf_target = tsdf_target[coords_down[:, 0], coords_down[:, 1], coords_down[:, 2], coords_down[:, 3]]
            occ_target = occ_target[coords_down[:, 0], coords_down[:, 1], coords_down[:, 2], coords_down[:, 3]]

        return tsdf_target, occ_target

    def forward(self, features, inputs, outputs, compute_loss=False):
        """ Coarse-to-fine reconstruction in current fragment.

        Args:
            features: (list), features for each image: eg. list[0]: pyramid features for image0.
                              dim: [(B, C0, H, W), (B, C1, H/2, W/2), (B, C2, H/2, W/2)]
            inputs: (dict), meta data from dataloader
            outputs: {}

        Returns:
            outputs: dict: {
                'coords': (Tensor), coordinates of voxels, dim: (number of voxels, 4) (4 : batch ind, x, y, z)
                'tsdf': (Tensor), TSDF of voxels, dim: (number of voxels, 1)
                }
            loss_dict: dict: {
                'loss_X': (Tensor), multi level loss
                }
        """

        bs = features[0][0].shape[0]
        pre_feat = None
        pre_coords = None
        loss_dict = {}

        # coarse to fine
        for i in range(self.cfg.N_LAYER):
            interval = 2 ** (self.n_scales - i)
            scale = self.n_scales - i
            loss_dict[f'loss_{i}'] = torch.Tensor([0.0]).cuda()[0]

            if i == 0:
                # generate new coords
                coords = self.generate_grid(self.cfg.N_VOX, interval)[0]
                up_coords = []
                for b in range(bs):
                    up_coords.append(torch.cat([torch.ones(1, coords.shape[-1], device=coords.device) * b, coords]))
                up_coords = torch.cat(up_coords, dim=1).permute(1, 0).contiguous()
            else:
                # upsample coords and features
                up_feat, up_coords = self.upsample(pre_feat, pre_coords, interval)
                
                up_near_feat, up_near_coords = None, None
                if self.cfg.FUSION.FUSION_ON and (len(near_feat) != 0):
                        # consider near voxels located outside the camera frustums to help geenrate coherent surface
                        up_near_feat, up_near_coords = self.upsample(near_feat, near_coords, interval)

                # upsample TSDF
                tsdf_local = up_feat[:, -2:-1] * 2
                tsdf_local[tsdf_local.abs() > 1] = 1
                coords_local = up_coords

            # visibility-aware local feature fusion
            feats = torch.stack([feat[scale] for feat in features])
            volume, view_weights, view_mask_targets, count = self.visible_fusion(up_coords, feats, inputs,
                                                                    self.cfg.VOXEL_SIZE, scale, output_gt=compute_loss)
            grid_mask = count > 0

            # compute view select loss
            if compute_loss:
                weights_loss = self.compute_view_select_loss(view_weights, view_mask_targets.float(), grid_mask)
                loss_dict[f'loss_{i}'] += weights_loss

            # concatenate features from last stage
            if i == 0:
                feat = volume
            else:
                feat = torch.cat([volume, up_feat], dim=1)

                if up_near_feat is not None:
                    up_near_feat = torch.cat([torch.zeros(up_near_feat.shape[0], volume.shape[1], dtype=volume.dtype, 
                                                          device=volume.device), up_near_feat], dim=1)
                    feat = torch.cat([feat, up_near_feat], dim=0)
                    up_coords = torch.cat([up_coords, up_near_coords], dim=0).detach().clone().float()

            # convert to aligned camera coordinate
            r_coords = up_coords.detach().clone().float()
            for b in range(bs):
                batch_ind = torch.nonzero(up_coords[:, 0] == b).squeeze(1)
                coords_batch = up_coords[batch_ind][:, 1:].float()
                coords_batch = coords_batch * self.cfg.VOXEL_SIZE + inputs['vol_origin_partial'][b].float()
                coords_batch = torch.cat((coords_batch, torch.ones_like(coords_batch[:, :1])), dim=1)
                coords_batch = coords_batch @ inputs['world_to_aligned_camera'][b, :3, :].permute(1, 0).contiguous()
                r_coords[batch_ind, 1:] = coords_batch

            # batch index is in the last position
            r_coords = r_coords[:, [1, 2, 3, 0]]

            # sparse 3D convolution
            point_feat = PointTensor(feat, r_coords)
            feat = self.sp_convs[i](point_feat)

            if i != 0 and up_near_feat is not None:
                feat = feat[:up_feat.shape[0]]
                up_coords = up_coords[:up_feat.shape[0]]

            # predict local TSDF and occupancy
            tsdf = self.local_tsdf_preds[i](feat)
            occ = self.local_occ_preds[i](feat)

            # compute local TSDF and occupancy loss
            if compute_loss:
                tsdf_target, occ_target = self.get_target(up_coords, inputs, scale)
                local_loss = self.compute_loss(tsdf, occ, tsdf_target, occ_target, grid_mask, self.cfg.POS_WEIGHT)
                loss_dict[f'loss_{i}'] += local_loss

            # find surface voxels for local sparsification
            if i < self.cfg.N_LAYER - 1:
                occupancy = sparsify(up_coords, torch.sigmoid(occ), feats, inputs, self.cfg.VOXEL_SIZE, scale,
                                            self.cfg.TOP_K_OCC[i], self.cfg.N_VOX, mode=self.cfg.FUSION.SPARSIFY)
            else:
                occupancy = torch.ones_like(feat[:, 0], dtype=torch.bool)
            occupancy[grid_mask == False] = False

            for b in range(bs):
                batch_ind = torch.nonzero(up_coords[occupancy][:, 0] == b).squeeze(1)
                if len(batch_ind) == 0:
                    logger.warning('no valid points: scale {}, batch {}'.format(i, b))
                    return outputs, loss_dict
            
            if self.cfg.FUSION.FUSION_ON:
                # global feature fusion via ConvGRU
                pre_coords, feat, tsdf_target, occ_target, tsdf_pred, near_coords, near_feat = self.gru_fusion(
                                                        up_coords, feat, inputs, i, grid_mask, occupancy,
                                                        global_tsdf=self.global_tsdf.global_volume[i] if i > 0 else None,
                                                        local_tsdf=(coords_local, tsdf_local) if i > 0 else None)

                # predict global TSDF and occupancy
                occ = self.global_occ_preds[i](feat)
                tsdf = self.global_tsdf_preds[i](feat)
                if tsdf_pred is not None:
                    tsdf = tsdf_pred + tsdf

                # compute global TSDF and occupancy loss
                if compute_loss and tsdf_target is not None:
                    global_loss = self.compute_loss(tsdf, occ, tsdf_target, occ_target, pos_weight=self.cfg.POS_WEIGHT)
                    loss_dict[f'loss_{i}'] += global_loss

                # avoid out of memory: sample points if num of points is too large
                if self.training and len(pre_coords) > self.cfg.TRAIN_NUM_SAMPLE[i] * bs:
                    choice = np.random.choice(len(pre_coords), self.cfg.TRAIN_NUM_SAMPLE[i] * bs, replace=False)
                    pre_coords, feat, tsdf, occ = pre_coords[choice], feat[choice], tsdf[choice], occ[choice]
                if self.training and near_feat.shape[0] > feat.shape[0]//4:
                    choice = np.random.choice(near_feat.shape[0], feat.shape[0]//4, replace=False)
                    near_feat, near_coords = near_feat[choice], near_coords[choice]
                    
            else:
                pre_coords = up_coords[occupancy]
                feat = feat[occupancy]
                tsdf = tsdf[occupancy]
                occ = occ[occupancy]

            pre_feat = torch.cat([feat, tsdf, occ], dim=1)
            if self.cfg.FUSION.FUSION_ON:
                near_feat = torch.cat([near_feat, torch.zeros_like(near_feat[..., :2])], dim=-1)

            pre_mask = occ.squeeze() > self.cfg.THRESHOLDS[i]
            tsdf[~pre_mask] = 1

            if self.cfg.FUSION.FUSION_ON:
                # TSDF should be in range [-1, 1]
                tsdf = tsdf.detach()
                tsdf[tsdf > 1] = 1
                tsdf[tsdf < -1] = -1
                # store predicted TSDF
                self.global_tsdf(pre_coords, tsdf, inputs, i)

            if self.training:
                torch.cuda.empty_cache()

            if i == self.cfg.PASS_LAYERS:
                outputs['coords'] = pre_coords
                outputs['tsdf'] = tsdf

                return outputs, loss_dict

    @staticmethod
    def compute_loss(tsdf, occ, tsdf_target, occ_target, mask=None, pos_weight=1.0, loss_weight=(1, 1)):
        """
        Args:
            tsdf: (Tensor), predicted TSDF, (N, 1)
            occ: (Tensor), predicted occupancy, (N, 1)
            tsdf_target: (Tensor),ground truth TSDF, (N, 1)
            occ_target: (Tensor), ground truth occupancy, (N, 1)
            loss_weight: (Tuple)
            mask: (Tensor), mask voxels which cannot be seen by all views
            pos_weight: (float)
        
        Returns:
            loss: (float) final loss
        """

        # compute occupancy/TSDF loss
        tsdf = tsdf.view(-1)
        occ = occ.view(-1)
        tsdf_target = tsdf_target.view(-1)
        occ_target = occ_target.view(-1)
        if mask is not None:
            mask = mask.view(-1)
            tsdf = tsdf[mask]
            occ = occ[mask]
            tsdf_target = tsdf_target[mask]
            occ_target = occ_target[mask]

        n_all = occ_target.shape[0]
        n_p = occ_target.sum()
        if n_p == 0:
            logger.warning('target: no valid voxel when computing loss')
            return torch.Tensor([0.0]).cuda()[0] * tsdf.sum()
        w_for_1 = (n_all - n_p).float() / n_p
        w_for_1 *= pos_weight

        # compute occ BCE loss
        occ_loss = F.binary_cross_entropy_with_logits(occ, occ_target.float(), pos_weight=w_for_1)

        # compute TSDF L1 loss
        tsdf = apply_log_transform(tsdf[occ_target])
        tsdf_target = apply_log_transform(tsdf_target[occ_target])
        tsdf_loss = torch.mean(torch.abs(tsdf - tsdf_target))

        # compute final loss
        loss = loss_weight[0] * occ_loss + loss_weight[1] * tsdf_loss

        return loss


    @staticmethod
    def compute_view_select_loss(view_weights, view_weights_target, mask=None):
        """
        Args:
            view_weights: (Tensor), predicted view_weights, dim: (N, 1)
            view_weights_target: (Tensor), ground truth view_targets, dim: (N, 1)
            mask: (Tensor), mask voxels which cannot be seen by any view
            pos_weight: (float)

        Returns:
            loss: final loss
        """

        if mask is not None:
            mask = mask.view(-1)
            view_weights = view_weights[mask]
            view_weights_target = view_weights_target[mask]

        loss = torch.Tensor([0.0]).cuda()[0]
        if mask.sum() != 0:
            # compute visibility L2 loss
            occ_view_mean = view_weights_target / (view_weights_target.sum(dim=1, keepdims=True) + 1e-10)
            loss = (occ_view_mean - view_weights).pow(2).mean()

        return loss