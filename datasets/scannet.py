import os
import numpy as np
import pickle
import cv2
from PIL import Image
from torch.utils.data import Dataset


class ScanNetDataset(Dataset):
    def __init__(self, datapath, mode, transforms, nviews, n_scales, scene=None, load_gt=True):
        super(ScanNetDataset, self).__init__()
        self.datapath = datapath
        self.mode = mode
        self.n_views = nviews
        self.transforms = transforms
        self.tsdf_file = 'all_tsdf_{}'.format(self.n_views)
        self.scene = scene

        assert self.mode in ["train", "val", "test"]
        self.metas = self.build_list()
        self.source_path = 'scans_test' if mode == 'test' else 'scans'

        self.n_scales = n_scales
        self.epoch = None
        self.tsdf_cashe = {}
        self.max_cashe = 100
        self.load_gt = load_gt

    def build_list(self):
        if self.scene is None:
            # load data for all scenes in the train/val/test split
            path = os.path.join(self.datapath, self.tsdf_file, 'fragments_{}.pkl'.format(self.mode))
        else:
            # load data for a specific scene
            path = os.path.join(self.datapath, self.tsdf_file, self.scene, 'fragments.pkl')
        with open(path, 'rb') as f:
            metas = pickle.load(f)
        return metas

    def __len__(self):
        return len(self.metas)

    def read_cam_file(self, filepath, vid):
        intrinsics = np.loadtxt(os.path.join(filepath, 'intrinsic', 'intrinsic_color.txt'), delimiter=' ')[:3, :3]
        intrinsics = intrinsics.astype(np.float32)
        poses = np.loadtxt(os.path.join(filepath, 'pose', '{}.txt'.format(str(vid))))
        return intrinsics, poses

    def read_img(self, filepath):
        img = Image.open(filepath)
        return img

    def read_depth(self, filepath):
        # Read depth image and camera pose
        depth_im = cv2.imread(filepath, -1).astype(np.float32)
        depth_im /= 1000.  # depth is saved in 16-bit PNG in millimeters
        return depth_im

    def read_scene_volumes(self, data_path, scene):
        if scene not in self.tsdf_cashe.keys():
            if len(self.tsdf_cashe) > self.max_cashe:
                self.tsdf_cashe = {}
            full_tsdf_list = []
            for l in range(self.n_scales + 1):
                # load full tsdf volume
                full_tsdf = np.load(os.path.join(data_path, scene, 'full_tsdf_layer{}.npz'.format(l)), allow_pickle=True)
                full_tsdf_list.append(full_tsdf.f.arr_0)
            self.tsdf_cashe[scene] = full_tsdf_list

        return self.tsdf_cashe[scene]

    def __getitem__(self, idx):
        meta = self.metas[idx]

        imgs = []
        poses_list = []
        intrinsics_list = []

        for i, vid in enumerate(meta['image_ids']):
            # load images
            imgs.append(self.read_img(os.path.join(self.datapath, self.source_path, meta['scene'], 'color', '{}.jpg'.format(vid))))

            # load intrinsics and extrinsics
            intrinsics, poses = self.read_cam_file(os.path.join(self.datapath, self.source_path, meta['scene']), vid)

            intrinsics_list.append(intrinsics)
            poses_list.append(poses)

        intrinsics = np.stack(intrinsics_list)
        poses = np.stack(poses_list)

        items = {
            'imgs': imgs,
            'intrinsics': intrinsics,
            'poses': poses,
            'vol_origin': meta['vol_origin'],
            'scene': meta['scene'],
            'fragment': meta['scene'] + '_' + str(meta['fragment_id']),
            'epoch': [self.epoch],
            'image_ids': meta['image_ids']
        }

        if self.load_gt:
            tsdf_list = self.read_scene_volumes(os.path.join(self.datapath, self.tsdf_file), meta['scene'])
            items['tsdf_list_full'] = tsdf_list

            depths = []
            for i, vid in enumerate(meta['image_ids']):
                depths.append(self.read_depth(os.path.join(self.datapath, self.source_path, meta['scene'], 'depth', '{}.png'.format(vid))))
            items['depths'] = depths

        if self.transforms is not None:
            items = self.transforms(items)

        return items
