""" Partially copied from [NeuralRecon](https://github.com/zju3dv/NeuralRecon) by Jiaming Sun and Yiming Xie. """

import torch
import torch.nn as nn
from torchsparse.tensor import PointTensor
from utils import sparse_to_dense_channel, sparse_to_dense_torch
from .modules import ConvGRU


class GlobalFusion(nn.Module):
    """
    Two functionalities of this class:
    1. GRU Fusion module. Update hidden state features with ConvGRU.
    2. Substitute TSDF in the global volume when direct_substitute = True.
    """

    def __init__(self, cfg, ch_in=None, direct_substitute=False):
        super(GlobalFusion, self).__init__()
        self.cfg = cfg
        # replace TSDF in global TSDF volume by direct substitute corresponding voxels
        self.direct_substitude = direct_substitute

        if direct_substitute:
            # TSDF
            self.ch_in = [1, 1, 1]
            self.feat_init = 1
        else:
            # features
            self.ch_in = ch_in
            self.feat_init = 0

        self.n_scales = len(cfg.THRESHOLDS) - 1
        self.scene_name = [None, None, None]
        self.global_origin = [None, None, None]
        self.global_volume = [None, None, None]
        self.target_tsdf_volume = [None, None, None]

        if direct_substitute:
            self.fusion_nets = None
        else:
            self.fusion_nets = nn.ModuleList()
            self.tsdf_mlp = nn.ModuleList()
            for i, ch in enumerate(ch_in):
                self.fusion_nets.append(ConvGRU(hidden_dim=ch,
                                                input_dim=ch,
                                                pres=1,
                                                vres=self.cfg.VOXEL_SIZE * 2 ** (self.n_scales - i)))
                self.tsdf_mlp.append(nn.Linear(in_features=1, out_features=ch))

    def reset(self, i):
        self.global_volume[i] = PointTensor(torch.Tensor([]), torch.Tensor([]).view(0, 3).long()).cuda()
        self.target_tsdf_volume[i] = PointTensor(torch.Tensor([]), torch.Tensor([]).view(0, 3).long()).cuda()

    def convert2dense(self, current_coords, current_values, coords_target_global, tsdf_target, relative_origin,
                      layer, grid_mask, occupancy, global_tsdf=None, local_tsdf=None):
        '''
        1. convert sparse features/TSDFs to dense feature/TSDF volume;
        2. combine current feature coordinates and previous coordinates within FBV from global hidden state to get
        new feature coordinates (updated_coords);
        3. fuse ground truth TSDF.

        Args:
            current_coords: (Tensor), current coordinates, dim: (num of voxels, 3)
            current_values: (Tensor), current features/TSDFs, dim: (num of voxels, C)
            coords_target_global: (Tensor), ground truth coordinates, dim: (N', 3)
            tsdf_target: (Tensor), TSDF ground truth, dim: (N',)
            relative_origin: (Tensor), origin in global volume, dim: (3,)
            layer: (int) layer index, from coarse to fine: 0, 1, 2
            grid_mask: mask: 1 represents the voxel is located in at least 1 camera frustum, dim: (num of voxels,)
            occupancy: mask: 0 represents the voxel should be sparsified, dim: (num of voxels,)
            global_tsdf: (sparse Tensor), global_tsdf.C: coords, dim: (N', 4)
                                          global_tsdf.F: stored TSDF predicted by previous fragments, dim: (N', 1)
            local_tsdf: (tuple), tuple[0]: coords, dim: (num of voxels, 4)
                                 tuple[1]: upsampled TSDF predicted by previous layers, dim: (num of voxels, 1)
        
        Returns:
            updated_coords: (Tensor), coordinates after combination, dim: (N", 3)
            current_volume: (Tensor), current dense feature/TSDF volume, dim: (DIM_X, DIM_Y, DIM_Z, C)
            global_volume: (Tensor), global dense feature/TSDF volume, dim: (DIM_X, DIM_Y, DIM_Z, C)
            target_volume: (Tensor), dense target TSDF volume, dim: (DIM_X, DIM_Y, DIM_Z, 1)
            valid: mask: 1 represents in grid mask, dim: (num of voxels,)
            valid_target: gt mask: 1 represents in current FBV, dim: (num of voxels,)
            global_tsdf_pred: dense TSDF volume predicted by previous fragments, dim: (DIM_X, DIM_Y, DIM_Z, 1)
            local_tsdf_pred: dense TSDF volume predicted by previous layers, dim: (DIM_X, DIM_Y, DIM_Z, 1)
            near_coords: (Tensor), coordinates of voxels that in current FBV but not in grid mask, dim: (M, 3)
            near_feat: (Tensor), global features of voxels that in current FBV but not in grid mask, dim: (M, C)
        '''
        
        # global map
        global_coords = self.global_volume[layer].C
        global_value = self.global_volume[layer].F
        global_tsdf_target = self.target_tsdf_volume[layer].F
        global_coords_target = self.target_tsdf_volume[layer].C

        # mask voxels that are out of the FBV
        dim = torch.div(torch.Tensor(self.cfg.N_VOX).cuda(), 2 ** (self.n_scales - layer), rounding_mode='trunc').int()
        dim_list = dim.data.cpu().numpy().tolist()
        global_coords = global_coords - relative_origin
        valid = ((global_coords < dim) & (global_coords >= 0)).all(dim=-1)

        if self.direct_substitude:
            grid_mask = torch.ones_like(current_coords[:, 0]).bool()

        # find overlapping voxels in the global map
        valid_volume = sparse_to_dense_torch(current_coords[grid_mask], 1, dim_list, 0, global_value.device, dtype=torch.bool)
        value = valid_volume[global_coords[valid][:, 0], global_coords[valid][:, 1], global_coords[valid][:, 2]]
        local_visible = torch.zeros_like(global_coords[:, 0], dtype=torch.bool)
        local_visible[valid] = value
        valid = local_visible

        # sparse to dense: features
        global_volume = sparse_to_dense_channel(global_coords[valid], global_value[valid], dim_list, self.ch_in[layer],
                                                self.feat_init, global_value.device)
        current_volume = sparse_to_dense_channel(current_coords, current_values, dim_list, self.ch_in[layer],
                                                    self.feat_init, global_value.device)
        
        if not self.direct_substitude:
            valid_current_volume = sparse_to_dense_torch(current_coords[occupancy], 1, dim_list, 0, global_value.device, dtype=torch.bool)
            updated_coords = torch.nonzero((global_volume != 0).any(-1) | valid_current_volume)

            # find the voxels in the global map that are in the current FBV but not in the grid mask
            valid_global = ((global_coords < dim) & (global_coords >= 0)).all(dim=-1)
            near_mask = valid_global & (valid == False)
            near_coords, near_feat = global_coords[near_mask], global_value[near_mask]

        else:
            updated_coords = torch.nonzero((global_volume.abs() < 1).any(-1) | (current_volume.abs() < 1).any(-1))
            near_coords, near_feat = None, None

        # sparse to dense: predicted TSDF
        global_tsdf_pred = None
        local_tsdf_pred = None
        if global_tsdf is not None:
            global_coords_pred = global_tsdf.C
            global_tsdf_pred = global_tsdf.F
            global_coords_pred = global_coords_pred - relative_origin
            valid_pred = ((global_coords_pred < dim) & (global_coords_pred >= 0)).all(dim=-1)
            global_tsdf_pred = sparse_to_dense_channel(global_coords_pred[valid_pred], global_tsdf_pred[valid_pred], 
                                                       dim_list, 1, -100, global_tsdf_pred.device)
        if local_tsdf is not None:
            local_tsdf_pred = sparse_to_dense_channel(local_tsdf[0], local_tsdf[1], dim_list, 1, 1, local_tsdf[1].device)

        # fuse ground truth
        if tsdf_target is not None:
            # mask voxels that are out of the FBV
            global_coords_target = global_coords_target - relative_origin
            valid_target = ((global_coords_target < dim) & (global_coords_target >= 0)).all(dim=-1)
            # combine current TSDF and global TSDF
            coords_target = torch.cat([global_coords_target[valid_target], coords_target_global])[:, :3]
            tsdf_target = torch.cat([global_tsdf_target[valid_target], tsdf_target.unsqueeze(-1)])
            # sparse to dense
            target_volume = sparse_to_dense_channel(coords_target, tsdf_target, dim_list, 1, 1, tsdf_target.device)
        else:
            target_volume = valid_target = None

        return updated_coords, current_volume, global_volume, target_volume, valid, valid_target, \
               global_tsdf_pred, local_tsdf_pred, near_coords, near_feat

    def update_map(self, values, coords, target_volume, valid, valid_target, relative_origin, layer):
        ''' Replace Hidden state/TSDF in global Hidden state/TSDF volume by direct substitute corresponding voxels.

        Args:
            values: (Tensor), fused features, dim: (num of voxels, C)
            coords: (Tensor), updated coords, dim: (num of voxels, 3)
            target_volume: (Tensor), TSDF volume, dim: (DIM_X, DIM_Y, DIM_Z, 1)
            valid: (Tensor), mask: 1 represents in current FBV, dim: (num of voxels,)
            valid_target: (Tensor), gt mask: 1 represents in current FBV, dim: (num of voxels,)
            relative_origin: (Tensor), origin in global volume, dim: (3,)
            layer: (int), layer index
        '''

        # pred
        self.global_volume[layer].F = torch.cat([self.global_volume[layer].F[valid == False], values])
        coords = coords + relative_origin
        self.global_volume[layer].C = torch.cat([self.global_volume[layer].C[valid == False], coords])

        # target
        if target_volume is not None:
            target_volume = target_volume.squeeze()
            self.target_tsdf_volume[layer].F = torch.cat([self.target_tsdf_volume[layer].F[valid_target == False],
                                                    target_volume[target_volume.abs() < 1].unsqueeze(-1)])
            target_coords = torch.nonzero(target_volume.abs() < 1) + relative_origin
            self.target_tsdf_volume[layer].C = torch.cat([self.target_tsdf_volume[layer].C[valid_target == False], target_coords])


    def save_mesh(self, layer, outputs, scene):
        if outputs is None:
            outputs = dict()
        if "scene_name" not in outputs:
            outputs['origin'] = []
            outputs['scene_tsdf'] = []
            outputs['scene_name'] = []
        # only keep the newest result
        if scene in outputs['scene_name']:
            # delete old
            idx = outputs['scene_name'].index(scene)
            del outputs['origin'][idx]
            del outputs['scene_tsdf'][idx]
            del outputs['scene_name'][idx]

        # scene name
        outputs['scene_name'].append(scene)

        fuse_coords = self.global_volume[layer].C
        tsdf = self.global_volume[layer].F.squeeze(-1)
        max_c = torch.max(fuse_coords, dim=0)[0][:3]
        min_c = torch.min(fuse_coords, dim=0)[0][:3]
        outputs['origin'].append(min_c * self.cfg.VOXEL_SIZE * (2 ** (self.n_scales - layer)))

        ind_coords = fuse_coords - min_c
        dim_list = (max_c - min_c + 1).int().data.cpu().numpy().tolist()
        empty_value = 1
        if self.cfg.SINGLE_LAYER_MESH:
            empty_value = -1
        tsdf_volume = sparse_to_dense_torch(ind_coords, tsdf, dim_list, empty_value, tsdf.device)
        outputs['scene_tsdf'].append(tsdf_volume)

        return outputs

    def forward(self, coords, values_in, inputs, layer=2, grid_mask=None, occupancy=None, outputs=None, 
                save_mesh=False, global_tsdf=None, local_tsdf=None):
        '''
        Args:
            coords: (Tensor), coordinates of voxels, dim: (num of voxels, 4) (4 : batch ind, x, y, z)
            values_in: (Tensor), features/TSDF, dim: (num of voxels, C)
            inputs: (dict), meta data from dataloader
            layer: (int) layer index, from coarse to fine: 0, 1, 2
            grid_mask: mask: 1 represents the voxel is located in at least 1 camera frustum, dim: (num of voxels,)
            occupancy: mask: 0 represents the voxel should be sparsified, dim: (num of voxels,)

            if direct_substitude:
                outputs: dict: {'coords': (Tensor), coordinates of voxels, dim: (number of voxels, 4)
                                'tsdf': (Tensor), TSDF of voxels, dim: (number of voxels, 1)}
                save_mesh: (bool), whether or not to save the reconstructed mesh of current sample
            else:
                global_tsdf: (sparse Tensor), global_tsdf.C: coords, dim: (N', 4)
                                              global_tsdf.F: stored TSDF predicted by previous fragments, dim: (N', 1)

                local_tsdf: (tuple), tuple[0]: coords, dim: (num of voxels, 4)
                                     tuple[1]: upsampled TSDF predicted by previous layers, dim: (num of voxels, 1)

        Returns:
            if direct_substitude and save_mesh:
                outputs: dict: {
                    'coords': (Tensor), coordinates of voxels, dim: (number of voxels, 4)
                    'tsdf': (Tensor), TSDF of voxels, dim: (number of voxels, 1)
                    'origin': (list), origin of the predicted partial volume, dim: [(3,)]
                    'scene_tsdf': (list), predicted TSDF volume, dim: [(nx, ny, nz)]
                    'scene_name': (list), name of each scene in 'scene_tsdf', dim: [string]
                    }
            else:
                updated_coords_all: (Tensor), updated coordinates, dim: (N", 4) (4 : Batch ind, x, y, z)
                values_all: (Tensor), features after gru fusion, dim: (N", C)
                tsdf_target_all: (Tensor), TSDF ground truth, dim: (N", 1)
                occ_target_all: (Tensor), occupancy ground truth, dim: (N", 1)
                tsdf_pred_all: (Tensor), previous TSDF predictions, dim: (N", 1)
                near_coords_all: (Tensor), coordinates of voxels that in current FBV but not in grid mask, dim: (M, 4)
                near_feat_all: (Tensor), global features of voxels that in current FBV but not in grid mask, dim: (M, C)
        '''

        # delete computational graph to save memory
        if self.global_volume[layer] is not None:
            self.global_volume[layer] = self.global_volume[layer].detach()
        if global_tsdf is not None:
            global_tsdf = global_tsdf.detach()
        if local_tsdf is not None:
            local_coords, local_tsdf = local_tsdf[0].detach(), local_tsdf[1].detach()

        batch_size = len(inputs['fragment'])
        interval = 2 ** (self.n_scales - layer)


        updated_coords_all, values_all = None, None
        tsdf_target_all, occ_target_all = None, None
        near_coords_all, near_feat_all = None, None
        tsdf_pred_all = None

        # incremental fusion
        for batch in range(batch_size):
            scene = inputs['scene'][batch]  # scene name
            global_origin = inputs['vol_origin'][batch]  # origin of global volume
            origin = inputs['vol_origin_partial'][batch]  # origin of local volume

            if scene != self.scene_name[layer] and self.scene_name[layer] is not None and self.direct_substitude:
                outputs = self.save_mesh(layer, outputs, self.scene_name[layer])

            # if this fragment is from new scene, reinitialize backend map
            if self.scene_name[layer] is None or scene != self.scene_name[layer]:
                self.scene_name[layer] = scene
                self.reset(layer)
                self.global_origin[layer] = global_origin

            # each level has its corresponding voxel size
            voxel_size = self.cfg.VOXEL_SIZE * interval

            # relative origin in global volume
            relative_origin = (origin - self.global_origin[layer]) / voxel_size
            relative_origin = relative_origin.cuda().long()

            batch_ind = torch.nonzero(coords[:, 0] == batch).squeeze(1)
            if len(batch_ind) == 0:
                continue

            coords_batch = torch.div(coords[batch_ind, 1:].long(), interval, rounding_mode='trunc')
            values = values_in[batch_ind]
            grid_mask_batch = grid_mask[batch_ind] if grid_mask is not None else None
            occupancy_batch = occupancy[batch_ind] if occupancy is not None else None

            if local_tsdf is not None:
                local_batch_ind = torch.nonzero(local_coords[:, 0] == batch).squeeze(1)
                local_coords_batch = torch.div(local_coords[local_batch_ind, 1:].long(), interval, rounding_mode='trunc')
                local_tsdf_batch = local_tsdf[local_batch_ind]

            if 'occ_list' in inputs.keys():
                # get partial gt
                occ_target = inputs['occ_list'][self.n_scales - layer][batch]
                tsdf_target = inputs['tsdf_list'][self.n_scales -  layer][batch][occ_target]
                coords_target = torch.nonzero(occ_target)
            else:
                coords_target = tsdf_target = None

            # convert to dense
            updated_coords, current_volume, global_volume, target_volume, valid, valid_target, \
            global_tsdf_pred, local_tsdf_pred, near_coords, near_feat = self.convert2dense(coords_batch, values,       
                                coords_target, tsdf_target, relative_origin, layer, grid_mask_batch, occupancy_batch,
                                global_tsdf, (local_coords_batch, local_tsdf_batch) if local_tsdf is not None else None)

            # dense to sparse: get features using new feature coordinates (updated_coords)
            values = current_volume[updated_coords[:, 0], updated_coords[:, 1], updated_coords[:, 2]]
            global_values = global_volume[updated_coords[:, 0], updated_coords[:, 1], updated_coords[:, 2]]

            # dense to sparse: get previous TSDF predictions for learning TSDF residuals;
            # for some voxels that have global TSDFs predicted by previous fragments at the same layer, use global one 
            # instead of local one since local TSDF is upsampled from previous layers (lower resolution).
            tsdf_pred = None
            if local_tsdf_pred is not None:
                tsdf_pred = local_tsdf_pred[updated_coords[:, 0], updated_coords[:, 1], updated_coords[:, 2]]
            if global_tsdf_pred is not None:
                global_tsdf_pred = global_tsdf_pred[updated_coords[:, 0], updated_coords[:, 1], updated_coords[:, 2]]
                tsdf_pred[global_tsdf_pred>=-1] = global_tsdf_pred[global_tsdf_pred>=-1]

            # combine features with previous TSDF predictions
            if local_tsdf_pred is not None:
                values += self.tsdf_mlp[layer](tsdf_pred)

            # get fused ground truth
            if target_volume is not None:
                tsdf_target = target_volume[updated_coords[:, 0], updated_coords[:, 1], updated_coords[:, 2]]
                occ_target = tsdf_target.abs() < 1
            else:
                tsdf_target = occ_target = None

            if not self.direct_substitude:
                # leverage near voxels (stored in the global map) that not in any camera frustums (grid mask) 
                # but in current FBV as input of ConvGRU to help generate more complete and coherent surface
                updated_coords_frag = torch.cat((updated_coords, near_coords), dim=0)
                global_values_frag = torch.cat((global_values, near_feat), dim=0)
                values_frag = torch.cat((values, torch.zeros_like(near_feat)), dim=0)

                # convert to aligned camera coordinate
                r_coords = updated_coords_frag.detach().clone().float()
                r_coords = r_coords.permute(1, 0).contiguous().float() * voxel_size + origin.unsqueeze(-1).float()
                r_coords = torch.cat((r_coords, torch.ones_like(r_coords[:1])), dim=0)
                r_coords = inputs['world_to_aligned_camera'][batch, :3, :] @ r_coords
                r_coords = torch.cat([r_coords, torch.zeros(1, r_coords.shape[-1], device=r_coords.device)])
                r_coords = r_coords.permute(1, 0).contiguous()

                h = PointTensor(global_values_frag, r_coords)
                x = PointTensor(values_frag, r_coords)

                values_frag = self.fusion_nets[layer](h, x)
                values = values_frag[:len(values)]

            # update global volume (direct substitute)
            self.update_map(values, updated_coords, target_volume, valid, valid_target, relative_origin, layer)

            if updated_coords_all is None:
                updated_coords_all = torch.cat([torch.ones_like(updated_coords[:, :1]) * batch, updated_coords * interval], dim=1)
                values_all = values
                tsdf_target_all = tsdf_target
                occ_target_all = occ_target
                if tsdf_pred is not None:
                    tsdf_pred_all = tsdf_pred

                if not self.direct_substitude:
                    near_coords_all = torch.cat([torch.ones_like(near_coords[:, :1]) * batch, near_coords * interval], dim=1)
                    near_feat_all = near_feat

            else:
                updated_coords = torch.cat([torch.ones_like(updated_coords[:, :1]) * batch, updated_coords * interval], dim=1)
                updated_coords_all = torch.cat([updated_coords_all, updated_coords])
                values_all = torch.cat([values_all, values])

                if tsdf_target_all is not None:
                    tsdf_target_all = torch.cat([tsdf_target_all, tsdf_target])
                    occ_target_all = torch.cat([occ_target_all, occ_target])
                if tsdf_pred is not None:
                    tsdf_pred_all = torch.cat([tsdf_pred_all, tsdf_pred])

                if not self.direct_substitude:
                    near_coords = torch.cat([torch.ones_like(near_coords[:, :1]) * batch, near_coords * interval], dim=1)
                    near_coords_all = torch.cat([near_coords_all, near_coords])
                    near_feat_all = torch.cat([near_feat_all, near_feat])

            if self.direct_substitude and save_mesh:
                outputs = self.save_mesh(layer, outputs, self.scene_name[layer])

        if self.direct_substitude:
            return outputs
        else:
            return updated_coords_all, values_all, tsdf_target_all, occ_target_all, tsdf_pred_all, \
                   near_coords_all, near_feat_all