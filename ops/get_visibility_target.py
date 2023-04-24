import torch
from torch.nn.functional import grid_sample


def get_visibility_target(coords, voxel_size, scale, depths_gt, proj_mats, mask, margin=3):
    """ Get the ground truth visibility mask.

    Args:
        coords: (Tensor), coordinates of voxels, dim: (num of views, 4, num of voxels) (4 : batch ind, x, y, z)
        voxel_size: (float), size of voxel in meter
        scale: (int), scale for voxel size
        depths_gt: (Tensor), depth images, dim: (number of views, H, W)
        proj_mats: (Tensor), projection matrics, dim: (num of views, 4, 4)
        mask: (Tensor), mask of views, dim: (number of views, num of voxels)
        margin: (int), number of voxels to truncate

    Returns:
        visibility_mask: (Tensor), ground truth visibility mask, dim: (num of views, num of voxels)
    """

    n_views, im_h, im_w = depths_gt.shape

    # project grid to depth images
    im_p = proj_mats @ coords # [n_views, 4, n_points]
    im_x, im_y, im_z = im_p[:, 0], im_p[:, 1], im_p[:, 2]
    im_x /= im_z
    im_y /= im_z

    # extract depths
    im_grid = torch.stack([2 * im_x / (im_w - 1) - 1, 2 * im_y / (im_h - 1) - 1], dim=-1) # [n_views, n_points, 2]
    im_grid = im_grid.view(n_views, 1, -1, 2) # [n_views, 1, n_points, 2]
    depths_gt = depths_gt.view(n_views, 1, im_h, im_w) # [n_views, 1,  H, W]
    depths = grid_sample(depths_gt, im_grid, mode='nearest', padding_mode='border', align_corners=True)
    depths = depths.view(n_views, -1) # [n_views, n_points]

    # mask out voxels outside camera frustums
    depths[mask == False] = 0

    # calculate tsdf
    sdf_trunc = margin * voxel_size * 2 ** scale
    tsdf = (depths - im_z) / sdf_trunc

    visibility_mask = mask & (depths > 0) & (tsdf >= -1) & (tsdf <= 1)
    visibility_mask[(visibility_mask.sum(0) == 1).unsqueeze(0).expand(n_views, -1)] = 0

    return visibility_mask
