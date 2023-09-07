from pack_existing_segs import *
import pickle
import random
import torch
import torch.utils.data as D
from torchvision import transforms
import os
import glob
from PIL import Image
import PIL
import json
from pathlib import Path


class COCODataset(D.Dataset):
    def __init__(self, root="../Data/COCO17", train=True, length=None, indices=None, no_seg=False, fix_resize=None):
        self.root = Path(root)
        self.no_seg = no_seg
        self.fix_resize = fix_resize

        with open(self.root / 'annotations/captions_train2017.json', 'r') as f:
            images_info = json.load(f)
        self.file_name_to_id = dict()
        for image_info in images_info['images']:
            self.file_name_to_id[image_info['file_name']] = image_info['id']
            # for reverse search
            self.file_name_to_id[image_info['id']] = image_info['file_name']

        with open(self.root / 'cap_dict.json', 'r') as f:
            self.captions_dict = json.load(f)

        if train:
            self.image_files = glob.glob(
                os.path.join(self.root / 'train2017', "*.jpg"))
        else:
            raise NotImplementedError
            # self.image_files = glob.glob(os.path.join(self.root / 'val2017', "*.jpg"))

        assert length is None or indices is None, "Cannot specify both len and indices"
        assert length is not None or indices is not None, "Must specify either len or indices"
        if length is not None:
            self.image_files = random.sample(self.image_files, length)
        else:
            self.image_files = [self.image_files[i] for i in indices]

    def __getitem__(self, index):
        image_file = self.image_files[index]
        image = PIL.Image.open(image_file)
        image = image.convert('RGB')
        file_name = image_file.split('/')[-1]
        # image_tensor = transforms.ToTensor()(image)
        image_tensor = transforms.Compose([
            transforms.Resize((image.height // 2, image.width // 2)
                              ) if self.fix_resize is None else transforms.Resize(self.fix_resize),
            transforms.ToTensor()
        ])(image)

        if self.no_seg:
            seg_output = torch.empty((0, 0, 0))
        else:
            with open(self.root / 'train2017seg' / (str(self.file_name_to_id[file_name]) + '.pkl'), 'rb') as f:
                packed_seg_out = pickle.load(f)
            seg_output = unpack_new_seg_out(packed_seg_out)
            seg_output = transforms.Resize(image_tensor.shape[1:])(
                torch.from_numpy(seg_output)
                # .to(device)
                )

        return image_tensor, seg_output, self.file_name_to_id[file_name]

    def __len__(self):
        return len(self.image_files)
