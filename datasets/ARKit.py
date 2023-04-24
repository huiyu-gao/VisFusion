""" Copied from [NeuralRecon](https://github.com/zju3dv/NeuralRecon) by Jiaming Sun and Yiming Xie. """

import os
import numpy as np
import pickle
from PIL import Image
from torch.utils.data import Dataset


class ARKitDataset(Dataset):
    def __init__(self, datapath, mode, transforms, nviews, n_scales, scene=None, load_gt=False):
        super(ARKitDataset, self).__init__()
        self.datapath = datapath
        self.mode = mode
        self.n_views = nviews
        self.transforms = transforms

        assert self.mode in ["train", "val", "test"]
        self.metas = self.build_list()

        self.n_scales = n_scales
        self.epoch = None
        self.tsdf_cashe = {}
        self.max_cashe = 100

    def build_list(self):
        with open(os.path.join(self.datapath, 'fragments.pkl'), 'rb') as f:
            metas = pickle.load(f)
        return metas

    def __len__(self):
        return len(self.metas)

    def read_img(self, filepath):
        img = Image.open(filepath)
        return img

    def __getitem__(self, idx):
        meta = self.metas[idx]

        imgs = []
        for i, vid in enumerate(meta['image_ids']):
            # load images
            imgs.append(self.read_img(os.path.join(self.datapath, 'images', '{}.jpg'.format(vid))))

        intrinsics = np.stack( meta['intrinsics'])
        poses = np.stack(meta['extrinsics'])

        items = {
            'imgs': imgs,
            'intrinsics': intrinsics,
            'poses': poses,
            'scene': meta['scene'],
            'fragment': meta['scene'] + '_' + str(meta['fragment_id']),
            'epoch': [self.epoch],
            'vol_origin': np.array([0, 0, 0])
        }

        if self.transforms is not None:
            items = self.transforms(items)

        return items
