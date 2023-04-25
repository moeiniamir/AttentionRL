import json
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from torchvision import transforms
import os
import timm
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import einops
import torch
import torch.nn as nn
from transformers import GPT2Config, GPT2LMHeadModel
from tokenizers import ByteLevelBPETokenizer
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchvision.transforms import Pad as ImagePad
from torchvision.transforms import Resize as ImageResize

MAX_SEQ_LENGTH = 32
BATCH_SIZE = 300
grid_size = [64, 64]
VOCAB_SIZE = 30_000

tokenizer = ByteLevelBPETokenizer()
tokenizer.train(files='Data/COCO17/all_text.txt', vocab_size=VOCAB_SIZE, min_frequency=5, special_tokens=[
    "<s>",
    "<pad>",
    "</s>",
    "<unk>",
    "<mask>",
])

class COCODataset:
  def __init__(self, cap_dict, image_info, image_path, cam=None, device=None):
    self.cap_dict = cap_dict
    self.index = sorted(list(self.cap_dict.keys()))
    self.index_set = set(self.index)
    self.image_id2name = dict()
    for item in image_info:
      if int(item['id']) in self.index_set:
        self.image_id2name[int(item['id'])] = item['file_name']
    self.image_files = [os.path.join(image_path, self.image_id2name[item]) for item in self.index]
    self.norm_mean = np.array([[[0.485, 0.456, 0.406]]])
    self.norm_std = np.array([[[0.229, 0.224, 0.225]]])
    self.transform = transforms.Compose([
      transforms.ToTensor(),
      transforms.Normalize(mean = [0.485, 0.456, 0.406], 
                           std = [0.229, 0.224, 0.225])
    ])
    self.cam = cam
    self.device = device
    self.downscale = 4
    
  def __getitem__(self, idx):
    image = Image.open(self.image_files[idx]).convert('RGB')
    image_array = np.array(image)
    h, w, _ = image_array.shape
    W = ((w + grid_size[0] - 1) // grid_size[0]) * grid_size[0] 
    H = ((h + grid_size[1] - 1) // grid_size[1]) * grid_size[1]
    image_array = np.pad(image_array, [((H-h)//2, (H-h)-(H-h)//2), ((W-w)//2, (W-w)-(W-w)//2), (0, 0)], mode='linear_ramp')
    image_tensor = torch.tensor(((image_array / 255 - self.norm_mean) / self.norm_std).transpose(2, 0, 1), dtype=torch.float32)
    captions = self.cap_dict[self.index[idx]]
    return image_tensor, captions

  def __len__(self):
    return len(self.index)
      
  def select_patches(self, image):
    # image = Image.fromarray(image_array)
    downsampler = ImageResize((image.shape[-2] // self.downscale, image.shape[-1] // self.downscale))
    upsampler = ImageResize((image.shape[-2], image.shape[-1]))
    res = upsampler(torch.tensor(self.cam(input_tensor=downsampler(image).to(self.device), targets=None))).numpy()
    image_grids = einops.rearrange(image, 'B C (N h) (M w) -> B N M C h w', h=grid_size[1], w=grid_size[0])
    res_grids = einops.rearrange(res, 'B (N h) (M w) -> B N M h w', h=grid_size[1], w=grid_size[0])
    candidate_grids = []
    B = res_grids.shape[0]
    u = np.mean(res_grids, axis=(-1, -2))
    u /= np.sum(u, axis=(-1, -2), keepdims=True)
    for i in range(B):
      choices = np.random.choice(range(u[i].size), p = u[i].flatten(), size = 5, replace=True)
      x, y = choices // u.shape[2], choices % u.shape[2]
      candidate_grids.append(image_grids[i][x, y])
    # candidate_grids = candidate_grids / 255
    # candidate_grids = (candidate_grids - np.array([[[[0.485, 0.456, 0.406]]]])) / np.array([[[[0.229, 0.224, 0.225]]]])
    m_input = torch.stack(candidate_grids).float()
    return m_input

  def caption2batch(self, captions, max_seq_length=MAX_SEQ_LENGTH):
    pad_id = tokenizer.encode('<pad>').ids[0]
    all_in = []
    all_out = []
    attention_mask = []
    max_tokens = 0
    for caption in captions:
      x_in = tokenizer.encode('<s>' + caption).ids
      x_out = tokenizer.encode(caption + '</s>').ids
      if len(x_in) > max_seq_length:
        x_in = [x_in[0]] + x_in[1:max_seq_length]
        x_out = x_out[:max_seq_length-1] + [x_out[-1]]
      all_in.append(x_in)
      all_out.append(x_out)
      max_tokens = max(max_tokens, len(x_in))
    for i in range(len(all_in)):
      attention_mask.append([1] * len(all_in[i]) + [0] * (max_tokens - len(all_in[i])))
      all_in[i] += [pad_id] * (max_tokens - len(all_in[i]))
      all_out[i] += [pad_id] * (max_tokens - len(all_out[i]))
    return torch.tensor(all_in, dtype=torch.int64), torch.tensor(all_out, dtype=torch.int64), torch.tensor(attention_mask, dtype=torch.int64)


  def collate_fn(self, data):
    images, captions = zip(*data)
    final_images = []
    u, v = 0, 0
    for i in range(len(images)):
      u, v = max(u, images[i].shape[1]), max(v, images[i].shape[2])
    #fill = torch.flatten(torch.tensor(-self.norm_mean / self.norm_std)).float()
    #print(fill)
    fill = 0
    for i in range(len(images)):
      left = (v - images[i].shape[2]) // 2
      right = v - (images[i].shape[2] + left)
      top = (u - images[i].shape[1]) // 2
      bottom = u - (images[i].shape[1] + top)
      
      pad = ImagePad(padding=[left, top, right, bottom], fill = fill, padding_mode='constant')
      final_images.append(pad(images[i]))
      # print(f'original shape: {images[i].shape}\n max: {[left, top, right, bottom]}\n pad: {[v - images[i].shape[2], u - images[i].shape[1]]}\n  final shape: {final_images[-1].shape}\n-------------\n')

    return torch.stack(final_images), captions
  
  def patch_batch(self, images, captions):
    patched_images = []
    selected_captions = []
    out = dict()
    patched_images = self.select_patches(images)
    for i, image in enumerate(images):
      # patched_images.append(self.select_patches(image))
      u = np.random.choice(range(len(captions[i])))
      selected_captions.append(captions[i][u])
    captions_in, captions_out, attention_mask = self.caption2batch(selected_captions)
    # patched_images = torch.stack(patched_images)
    out['patched_images'] = patched_images
    out['captions_in'] = captions_in
    out['captions_out'] = captions_out
    out['attention_mask'] = attention_mask
    out['selected_captions'] = selected_captions
    out['captions'] = captions
    return out
    