import torch
import torch.nn as nn
from torch.nn.functional import grid_sample
from torchsparse.tensor import PointTensor
from models.modules import SCNN
from ops.get_visibility_target import get_visibility_target


class VisibleFusion(nn.Module):

    def __init__(self, cfg):
        super(VisibleFusion, self).__init__()
        self.cfg = cfg
        self.n_scales = len(cfg.THRESHOLDS) - 1
        self.similarity_dim = cfg.N_VIEWS * (cfg.N_VIEWS - 1)

        self.view_select = nn.ModuleList()
        for i in range(len(cfg.THRESHOLDS)):
            self.view_select.append(SCNN(num_classes=1,
                                         in_channels=self.similarity_dim,
                                         out_channels=cfg.N_VIEWS,
                                         dropout=self.cfg.SPARSEREG.DROPOUT))

    def get_view_mask(self, coords, voxel_size, scale, imgs, proj_mats):
        """ Get the mask of views for each voxel. 
            True for view i means the voxel could be projected into the i-th image (within the image range).

        Args:
            coords: (Tensor), coordinates of voxels, dim: (num of views, 4, num of voxels) (4 : batch ind, x, y, z)
            voxel_size: (float), size of voxel in meter
            scale: (int), scale for voxel size
            imgs: (Tensor), images, dim: (number of views, C, H, W)
            proj_mats: (Tensor), projection matrics, dim: (num of views, 4, 4)

        Returns:
            mask: (Tensor), True means the voxel is projected within the image, dim: (num of views, num of voxels)
        """

        n_views, _, im_h, im_w = imgs.shape
        rs_grid_corners = coords[:, :, None].clone()

        # get the coords of voxel corners
        corners = torch.tensor([[0.5, 0.5, 0.5, 0.0], [0.5, 0.5, -0.5, 0.0],
                            [0.5, -0.5, 0.5, 0.0], [0.5, -0.5, -0.5, 0.0],
                            [-0.5, 0.5, 0.5, 0.0], [-0.5, 0.5, -0.5, 0.0],
                            [-0.5, -0.5, 0.5, 0.0], [-0.5, -0.5, -0.5, 0.0]])
        corners = corners.transpose(1, 0).to(device=coords.device, dtype=coords.dtype)[None, :, :, None]
        rs_grid_corners = rs_grid_corners + corners * voxel_size * (2 ** scale)
        rs_grid_corners = rs_grid_corners.reshape([n_views, 4, -1])

        # project to image
        im_p_corners = proj_mats @ rs_grid_corners
        im_x_corners, im_y_corners, im_z_corners = im_p_corners[:, 0], im_p_corners[:, 1], im_p_corners[:, 2]
        im_x_corners /= im_z_corners
        im_y_corners /= im_z_corners
        im_grid_corners = torch.stack([2 * im_x_corners / (im_w - 1) - 1, 2 * im_y_corners / (im_h - 1) - 1], dim=-1)
        im_grid_corners = im_grid_corners.reshape([n_views, 8, coords.shape[2], 2])

        # assign True if any of the 8 corners could be projected within the image
        mask = ((im_grid_corners.abs() <= 1.1).all(dim=-1) & (
                    im_z_corners.reshape([n_views, 8, coords.shape[2]]) > 0)).any(dim=1)

        return mask

    def forward(self, coords, feats, inputs, voxel_size, scale, output_gt=False):
        """ Form a 3D (sparse) feature volume by fusing image features from multiple views
            according to the visibility of voxels.

        Args:
            coords: (Tensor), coordinates of voxels, dim: (num of voxels, 4) (4 : batch ind, x, y, z)
            feats: (Tensor), image feature maps, dim: (num of views, batch size, C, H, W)
            inputs: (dict), meta data from dataloader
            voxel_size: (float), size of voxel in meter
            scale: (int), scale for voxel size
            output_gt: (bool), whether or not to calculate and output the ground truth visibility

        Returns:
            feature_volume: (Tensor), 3D (sparse) feature volume, dim: (num of voxels, C + 1)
            view_weights: (Tensor), fusion weights for different views, dim: (num of voxels, num of views)
            visibility_target_all: (Tensor), ground truth visibility, dim: (num of voxels, num of views)
            count: (Tensor), number of times each voxel can be seen (ignore occlusion), dim: (num of voxels,)
        """
        
        n_views, bs, c, h, w = feats.shape
        device = feats.device
        n_points_all = coords.shape[0]

        feature_volume_all = torch.zeros(n_points_all, n_views, c, device=device)
        visibility_values_all = torch.zeros(n_points_all, self.similarity_dim, device=device)
        mask_all = torch.zeros(n_points_all, n_views, dtype=torch.bool, device=device)
        im_z_norm_all = torch.zeros(n_points_all, 1, device=device)
        visibility_target_all = torch.zeros(n_points_all, n_views, dtype=torch.bool, device=device) if output_gt else None

        c_coords = coords.detach().clone().float()
        for batch in range(bs):
            batch_ind = torch.nonzero(coords[:, 0] == batch).squeeze(1)
            n_points = len(batch_ind)

            coords_batch = coords[batch_ind][:, 1:].view(-1, 3) # [n_points, 3]
            feats_batch = feats[:, batch] # [n_views, c, h, w]
            proj_feat_batch = inputs['proj_matrices'][batch, :, scale+1] # [n_views, 4, 4]
            proj_img_batch = inputs['proj_matrices'][batch, :, 0] # [n_views, 4, 4]
            origin_batch = inputs['vol_origin_partial'][batch].unsqueeze(0)  # [1, 3]

            # convert to world coordinates
            world_coords_batch = coords_batch * voxel_size + origin_batch.float()
            w_coords = world_coords_batch.unsqueeze(0).expand(n_views, -1, -1) # [n_views, n_points, 3]
            w_coords =w_coords.permute(0, 2, 1).contiguous() # [n_views, 3, n_points]
            w_coords = torch.cat([w_coords, torch.ones([n_views, 1, n_points], device=device)], dim=1)

            # get the mask of views for each voxel.
            mask = self.get_view_mask(w_coords, voxel_size, scale, inputs['imgs'][batch], proj_img_batch)

            if output_gt:
                vis_target = get_visibility_target(w_coords, voxel_size, scale, inputs['depths'][batch], 
                                                   proj_img_batch, mask, margin=3)
                visibility_target_all[batch_ind] = vis_target.permute(1, 0)

            # project to feature maps
            im_p = proj_feat_batch @ w_coords # [n_views, 4, n_points]
            im_x, im_y, im_z = im_p[:, 0], im_p[:, 1], im_p[:, 2]
            im_x /= im_z
            im_y /= im_z

            # extract features from feature maps
            im_grid = torch.stack([2 * im_x / (w - 1) - 1, 2 * im_y / (h - 1) - 1], dim=-1) # [n_views, n_points, 2]
            im_grid = im_grid.view(n_views, 1, -1, 2) # [n_views, 1, n_points, 2]
            features = grid_sample(feats_batch, im_grid, padding_mode='border', align_corners=True)
            features = features.view(n_views, c, -1) # [n_views, c, n_points]

            # remove nan
            mask = mask.view(n_views, -1) # [n_views, n_points]
            features[mask.unsqueeze(1).expand(-1, c, -1) == False] = 0
            im_z[mask == False] = 0

            feature_volume_all[batch_ind] = features.permute(2, 0, 1).contiguous()
            mask_all[batch_ind] = mask.permute(1, 0)

            # calculate similarity maps
            feature_norm = torch.linalg.norm(features, dim=1) # [n_views, n_points]
            feature_norm = features / (feature_norm.unsqueeze(1) + 1e-10) # [n_views, c, n_points]
            similarity_map = feature_norm.permute(2, 0, 1) @ feature_norm.permute(2, 1, 0) # [n_points, n_views, n_views]
            del features, feature_norm

            # remove diagonal entries and flatten the maps as vectors
            visibility_values = similarity_map.reshape(n_points, -1)[:, :-1].reshape(
                                        n_points, n_views - 1, n_views + 1)[:, :, 1:].reshape(n_points, -1)
            visibility_values_all[batch_ind] = visibility_values
            del similarity_map, visibility_values

            # normalize depth values
            im_z = im_z.sum(dim=0).unsqueeze(1) / (mask.sum(dim=0) + 1e-10).unsqueeze(1) # [n_points, 1] mean of views
            im_z_mean = im_z[im_z > 0].mean()
            im_z_std = torch.norm(im_z[im_z > 0] - im_z_mean) + 1e-5
            im_z_norm = (im_z - im_z_mean) / im_z_std
            im_z_norm[im_z <= 0] = 0
            im_z_norm_all[batch_ind] = im_z_norm

            # convert to aligned camera coordinate
            c_coords_batch = torch.cat((world_coords_batch, torch.ones_like(world_coords_batch[:, :1])), dim=1)
            c_coords_batch = c_coords_batch @ inputs['world_to_aligned_camera'][batch, :3, :].permute(1, 0).contiguous()
            c_coords[batch_ind, 1:] = c_coords_batch

        # predict weights for different views by sparse 3D convolution
        c_coords = c_coords[:, [1, 2, 3, 0]]
        point_feat = PointTensor(visibility_values_all, c_coords)
        view_weights = self.view_select[self.n_scales - scale](point_feat, pres=1, vres=self.cfg.VOXEL_SIZE * 2 ** scale)

        # mask out voxels outside camera frustums
        view_weights = view_weights * mask_all

        # feature fusion
        feature_volume = (feature_volume_all * view_weights.unsqueeze(2)).sum(1)
        del feature_volume_all

        feature_volume = torch.cat([feature_volume, im_z_norm_all], dim=1)
        count = mask_all.sum(dim=1).float()

        return feature_volume, view_weights, visibility_target_all, count