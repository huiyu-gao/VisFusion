import torch
from utils import sparse_to_dense_torch
import torch.nn.functional as F


def sparsify(coords, occ, feats, inputs, voxel_size, scale, K, N_VOX, mode='continuous_k', near_far=[0.2, 3.5]):
    """ Ray-based local sparsification of the current feature volume.

    Args:
        coords: (Tensor), coordinates of voxels, dim: (num of voxels, 4) (4 : batch ind, x, y, z)
        occ: (Tensor), occupancy values of voxels, dim: (num of voxels, 1)
        feats: (Tensor), image feature maps, dim: (num of views, batch size, C, H, W)
        inputs: (dict), meta data from dataloader
        voxel_size: (float), size of voxel in meter
        scale: (int), scale for voxel size
        K: (int), number of voxels to keep along each ray
        N_VOX: (list), number of voxels along axis x, y, z at finest level, dim: [n_x, x_y, x_z]
        mode: (string), 'continuous_k' or 'top_k'
        near_far: (list), range to sample points along each ray

    Returns:
        occupancy_mask_all: (Tensor), a mask indicates which voxels should be reserved, dim: (num of voxels,)
    """

    N_views, bs, c, H, W = feats.shape
    device = occ.device
    D = int(N_VOX[2] / 2 ** scale * 1.5)
    default_occ_value = -100

    occupancy_mask_all = torch.zeros(occ.shape[0], dtype=torch.bool, device=device)

    for batch in range(bs):
        batch_ind = torch.nonzero(coords[:, 0] == batch).squeeze(1)

        coords_batch = coords[batch_ind, 1:].view(-1, 3) # [n_points, 3]
        occ_batch = occ[batch_ind].squeeze(1)
        origin_batch = inputs['vol_origin_partial'][batch].unsqueeze(0) # [1, 3]
        proj_feat_batch = inputs['proj_matrices_inv'][batch, :, scale+1] # [n_views, 4, 4]

        # create grid in camera coordinate system
        near, far = near_far
        ys, xs = torch.meshgrid(torch.linspace(0, H - 1, H, device=device), torch.linspace(0, W - 1, W, device=device)) # HW
        near_plane = torch.stack((xs, ys, torch.ones_like(xs, device=device)), dim=-1) * near
        far_plane = torch.stack((xs, ys, torch.ones_like(xs, device=device)), dim=-1) * far

        # sample points along each pixel ray
        linspace_z = torch.linspace(0.0, 1.0, D, device=device).view(D, 1, 1, 1)
        pts_volume = (1.0 - linspace_z) * near_plane + linspace_z * far_plane # [D, H, W, 3]
        pts_volume = pts_volume.view(-1, 3).T

        # inverse of project matrix
        R = proj_feat_batch[:, :3, :3] # [n_views, 3, 3]
        T = proj_feat_batch[:, :3, 3:] # [n_views, 3, 1]

        # transform points from camera to world
        pts_volume = pts_volume[None].expand(N_views, -1, -1)
        pts_volume = R @ pts_volume + T # [n_views, 3, n_samples]
        pts_volume = pts_volume.permute(0, 2, 1).contiguous().view(N_views, D, H, W, 3) # [n_views, D, H, W, 3]

        # build occupancy volume
        dim = torch.div(torch.Tensor(N_VOX), 2 ** scale, rounding_mode='trunc').int()
        dim_list = dim.data.cpu().numpy().tolist()
        occ_coords = torch.div(coords_batch.long(), 2 ** scale, rounding_mode='trunc') # [n_points, 3]

        # sparse to dense
        occ_volume = sparse_to_dense_torch(occ_coords, occ_batch, dim_list, default_occ_value, device)
        occ_mask = sparse_to_dense_torch(occ_coords, 1, dim_list, 0, device, dtype=torch.bool)

        # grid sample expects coords in [-1,1]
        pts_volume = torch.round((pts_volume - origin_batch) / voxel_size / 2 ** scale).long() # [n_views, D, H, W, 3]
        pts_volume_n = 2 * pts_volume / (torch.Tensor(dim_list).cuda() - 1) - 1
        pts_volume_n = pts_volume_n[:, :, :, :, [2, 1, 0]]
        occ_volume_n = occ_volume[None, None, ...].expand(N_views, -1, -1, -1, -1)

        # get corresponding occupancy values for each sampled points
        pts_occ = F.grid_sample(occ_volume_n, pts_volume_n, mode='nearest', padding_mode='border', 
                                align_corners=False).squeeze(1) # [n_views, D, H, W]

        # mask out voxels outside local fragment
        valid_mask = (pts_volume_n.abs() <= 1).sum(dim=-1) == 3 # [n_views, D, H, W]
        del pts_volume_n
        valid_mask[pts_occ == default_occ_value] = False
        pts_occ[valid_mask == False] = default_occ_value # [n_views, D, H, W]
        
        # mask out duplicated voxels along each ary
        invalid_mask = (pts_volume[:, 1:, :, :, :] == pts_volume[:, :-1, :, :, :]).all(dim=-1) # [n_views, D-1, H, W]
        invalid_mask = torch.cat([torch.zeros(invalid_mask[:, :1].shape, dtype=torch.bool, device=device), 
                                  invalid_mask], dim=1) # [n_views, D, H, W]
        pts_occ[invalid_mask] = default_occ_value # [n_views, D, H, W]

        if mode == 'continuous_k':
            # for each pixel ray, keep consecutive k voxels with the highest sum of occupancies

            valid_mask = valid_mask & (invalid_mask == False)
            del invalid_mask

            # let the valid voxels be adjacent
            _, indices = torch.sort(valid_mask.float(), dim=1, descending=True, stable=True)
            sorted_pts_occ = torch.gather(pts_occ, dim=1, index=indices)

            # consider neighbouring voxels using a sliding window
            left, right = int((K + 1) / 2), int(K / 2)
            sorted_pts_occ = torch.cat([torch.zeros(N_views, left, H, W, device=device), sorted_pts_occ], dim=1)
            sorted_pts_occ = torch.cat([sorted_pts_occ, torch.ones(N_views, right, H, W, device=device) * default_occ_value], dim=1)
            sorted_pts_occ_sum = torch.cumsum(sorted_pts_occ, dim=1)
            sorted_pts_occ_sum = sorted_pts_occ_sum[:, K:] - sorted_pts_occ_sum[:, :-K] # [n_views, D, H, W]
            del sorted_pts_occ

            # find the indices of surface voxels with the highest sum of occupancies along each ray
            surface_index = torch.argmax(sorted_pts_occ_sum, dim=1)
            surface_index[surface_index < (left - 1)] = left
            surface_index[surface_index > (D - right - 1)] = D - right - 1
            surface_index = surface_index[:, None].expand(-1, D, -1, -1) # [n_views, D, H, W]
            del sorted_pts_occ_sum

            # get the surface mask
            surface_mask = torch.linspace(0.0, D - 1, D, device=device).view(1, D, 1, 1).repeat(N_views, 1, H, W)
            surface_mask = ((surface_mask - surface_index) > - left) & ((surface_mask - surface_index) <= right)
            _, indices = torch.sort(indices, dim=1)
            surface_mask = torch.gather(surface_mask, dim=1, index=indices)
            surface_mask = valid_mask & surface_mask # [n_views, D, H, W]
            del surface_index, indices

            # only consider rays that have high probability to reach a surface within the sampling range
            sorted, _ = torch.topk(pts_occ, K, dim=1) # [n_views, K, H, W]
            surface_mask = surface_mask & (sorted.sum(1) > K * 0.3).unsqueeze(1).expand(-1, D, -1, -1)

            # get the coordinates of these voxels
            voxel_coords = pts_volume[surface_mask]
            del surface_mask, pts_volume

        elif mode == 'top_k':
            # for each pixel ray, keep k voxels with top k occupancies along this ray

            # find the indices of voxels with top k occupancy values along each ray
            sorted, indices = torch.topk(pts_occ, K, dim=1) # [n_views, K, H, W]

            # get the coordinates of these voxels
            voxel_grid = torch.stack((torch.meshgrid(torch.linspace(0, N_views - 1, N_views),
                                        torch.linspace(0, K - 1, K),
                                        torch.linspace(0, H - 1, H),
                                        torch.linspace(0, W - 1, W))), dim=-1) # [n_views, K, H, W, 4]
            voxel_grid[:, :, :, :, 1] = indices
            voxel_grid = voxel_grid.reshape(-1, 4).long() # [n_views * K * H * W, 4]
            voxel_coords = pts_volume[voxel_grid[:, 0], voxel_grid[:, 1], voxel_grid[:, 2], voxel_grid[:, 3]] # [n_views * K * H * W, 3]
            surface_mask = (sorted.sum(1) > K * 0.3).unsqueeze(1).expand(-1, K, -1, -1).reshape(-1)
            voxel_coords = voxel_coords[surface_mask]

        # keep unique and valid voxel coordinate
        voxel_coords = torch.unique(voxel_coords, dim=0)
        if len(voxel_coords) != 0:
            valid = ((voxel_coords < dim.cuda()) & (voxel_coords >= 0)).all(dim=-1)
            voxel_coords = voxel_coords[valid]

        # get the mask for sparsifying
        occupancy_mask = sparse_to_dense_torch(voxel_coords, 1, dim_list, 0, device, dtype=torch.bool)
        occupancy_mask = occupancy_mask & occ_mask
        occupancy_mask = occupancy_mask[occ_coords[:, 0], occ_coords[:, 1], occ_coords[:, 2]]

        occupancy_mask_all[batch_ind] = occupancy_mask

    return occupancy_mask_all