import torch
import torch.nn as nn

from .backbone import MnasMulti
from .coarse_to_fine import ReconNet
from .global_fusion import GlobalFusion
from utils import tocuda


class VisFusion(nn.Module):
    def __init__(self, cfg):
        super(VisFusion, self).__init__()
        self.cfg = cfg.MODEL

        alpha = float(self.cfg.BACKBONE2D.ARC.split('-')[-1])
        self.pixel_mean = torch.Tensor(cfg.MODEL.PIXEL_MEAN).view(-1, 1, 1)
        self.pixel_std = torch.Tensor(cfg.MODEL.PIXEL_STD).view(-1, 1, 1)
        self.n_scales = len(self.cfg.THRESHOLDS) - 1

        # networks
        self.backbone2d = MnasMulti(alpha)
        self.coarse_to_fine_recon = ReconNet(cfg.MODEL)
        self.fuse_to_global = GlobalFusion(cfg.MODEL, direct_substitute=True)

    def normalizer(self, x):
        """ Normalizes the RGB images to the input range """
        return (x - self.pixel_mean.type_as(x)) / self.pixel_std.type_as(x)

    def forward(self, inputs, save_mesh=False, finetune_layer=None):
        """ Online 3D scene reconstruction.

        Args:
            inputs: dict: {
                'imgs':                    (Tensor), images,
                                                     dim: (batch size, number of views, C, H, W)
                'proj_matrices':           (Tensor), projection matrices to input images and feature maps,
                                                     dim: (batch size, number of views, 1 + number of scales, 4, 4)
                'proj_matrices_inv':       (Tensor), inverse matrices of the projection matrices,
                                                     dim: (batch size, number of views, 1 + number of scales, 4, 4)
                'vol_origin':              (Tensor), origin of the full voxel volume (xyz position of voxel (0, 0, 0)),
                                                     dim: (batch size, 3)
                'vol_origin_partial':      (Tensor), origin of the partial voxel volume (xyz position of voxel (0, 0, 0)),
                                                     dim: (batch size, 3)
                'world_to_aligned_camera': (Tensor), matrices: transform from world coords to aligned camera coords,
                                                     dim: (batch size, number of views, 4, 4)
                when we have ground truth:
                'depths':                  (Tensor), depth images,
                                                     dim: (batch size, number of views, H, W)
                'tsdf_list':               (list), TSDF ground truth for each level,
                                                   dim: [(batch size, DIM_X, DIM_Y, DIM_Z)]
                'occ_list':                (list), occupancy ground truth for each level,
                                                   dim: [(batch size, DIM_X, DIM_Y, DIM_Z)]
                others: unused in network
                }
            save_mesh: (bool), whether or not to save the reconstructed mesh of current sample
            finetune_layer: set as None (default) to train the whole model;
                            set as 0, 1 or 2 to fine tune a specific layer

        Returns:
            outputs: dict: {
                'coords':         (Tensor), coordinates of voxels, dim: (number of voxels, 4) (4 : batch ind, x, y, z)
                'tsdf':           (Tensor), TSDF of voxels, dim: (number of voxels, 1)
                if save_mesh:
                'origin':         (list), origin of the predicted partial volume, dim: [(3,)]
                'scene_tsdf':     (list), predicted TSDF volume, dim: [(nx, ny, nz)]
                }
            loss_dict: dict: {
                'loss_X':         (Tensor), multi level loss
                'total_loss':     (Tensor), total loss
                }
        """

        inputs = tocuda(inputs)
        outputs = {}
        compute_loss = ('tsdf_list' in inputs.keys())

        # image feature extraction
        # in: images; out: feature maps
        imgs = torch.unbind(inputs['imgs'], 1)
        features = [self.backbone2d(self.normalizer(img)) for img in imgs]

        # coarse-to-fine reconstruction in current fragment
        # in: image features; out: sparse coords and TSDF
        outputs, loss_dict = self.coarse_to_fine_recon(features, inputs, outputs, compute_loss)

        # fuse to global map
        if not self.training and 'coords' in outputs.keys():
            outputs = self.fuse_to_global(outputs['coords'], outputs['tsdf'], inputs, self.cfg.PASS_LAYERS,
                                          outputs=outputs, save_mesh=save_mesh)

        # gather loss
        if compute_loss:
            weighted_loss = 0.0
            for key, value in loss_dict.items():
                layer = int(key.split('_')[1])
                if finetune_layer is None or finetune_layer == layer:
                    weighted_loss += value * self.cfg.LW[layer]

            loss_dict.update({'total_loss': weighted_loss})

        return outputs, loss_dict