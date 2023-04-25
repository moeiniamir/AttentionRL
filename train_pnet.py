import json
#from pycocotools.coco import COCO
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
from dataloader import *
os.environ['TOKENIZERS_PARALLELISM'] = "false"

MAX_SEQ_LENGTH = 32
BATCH_SIZE = 160

transformer_hidden_size = 256

max_lr = 0.001
weight_decay = 5e-4
n_epochs = 30


captions_train = json.load(open('Data/COCO17/cap_dict_train.json', 'r'))
captions_train = {int(k):v for k, v in captions_train.items()}
captions_val = json.load(open('Data/COCO17/cap_dict_val.json', 'r'))
captions_val = {int(k):v for k, v in captions_val.items()}

#orig = COCO('Data/COCO17/captions_train2017.json')
orig = json.load(open('Data/COCO17/annotations/captions_train2017.json', 'r'))
#images = orig.loadImgs(sorted(list(captions.keys())))
images_train = sorted([item for item in orig['images'] if int(item['id']) in captions_train], key=lambda x: x['id'])

orig = json.load(open('Data/COCO17/annotations/captions_val2017.json', 'r'))
images_val = sorted([item for item in orig['images'] if int(item['id']) in captions_val], key=lambda x: x['id'])


device = 'cuda:0'
image_model = timm.create_model('efficientnet_b0', pretrained=False).to(device)
image_model.load_state_dict(torch.load('Models/eff-b0.pth'))
_ = image_model.eval()
cam = GradCAM(model=image_model, target_layers=[image_model.blocks[6][-1]], use_cuda=True)

dataset_train = COCODataset(captions_train, images_train, 'Data/COCO17/train2017/', cam=cam, device=device)
dataset_val = COCODataset(captions_val, images_val, 'Data/COCO17/val2017/', cam=cam, device=device)

dataloaders = {
    'train': DataLoader(dataset_train, batch_size=BATCH_SIZE, num_workers=6, collate_fn=dataset_train.collate_fn, shuffle=True),
    'val': DataLoader(dataset_val, batch_size=BATCH_SIZE, num_workers=6, collate_fn=dataset_val.collate_fn, shuffle=False)
}

class PolicyNetwork(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        self.fc = nn.Linear(1280, transformer_hidden_size)
    
    def forward(self, cap_in, attn, features):
        features = self.fc(features)
        return self.backbone(input_ids=cap_in, attention_mask=attn, encoder_hidden_states=features)

    def generate(self, features):
        features = self.fc(features)
        return self.backbone.generate(encoder_hidden_states=features)

config = GPT2Config(vocab_size=VOCAB_SIZE, n_positions=32, n_embd=transformer_hidden_size, n_layer=8, n_head=4, eos_token_id=2, bos_token_id=0, add_cross_attention=True)
policy_backbone = GPT2LMHeadModel(config).to(device)
value_backbone = GPT2LMHeadModel(config).to(device)

policy_network = PolicyNetwork(backbone=policy_backbone).to(device)
criterion = nn.CrossEntropyLoss(reduction='none').to(device)
optimizer = torch.optim.Adam(policy_network.parameters(), lr=max_lr / 10, weight_decay=weight_decay)
lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=max_lr, epochs=n_epochs, steps_per_epoch=len(dataloaders['train']))

bestLoss = 5
for epoch in range(n_epochs):
    print(f'#### Epoch {epoch + 1}/{n_epochs}:')
    for mode in ['train', 'val']:
        if mode == 'train':
            policy_network.train()
        elif mode == 'val':
            policy_network.eval()
        total_loss = 0
        total_accuracy = 0
        count = 0
        print(f"------------------{mode}----------------")
        for item in tqdm(dataloaders[mode]):
            data_batch = dataset_train.patch_batch(*item)
            img = data_batch['patched_images'].to(device)
            cap_in = data_batch['captions_in'].to(device)
            cap_out = data_batch['captions_out'].to(device)
            attn = data_batch['attention_mask'].to(device)
            N, T, C, H, W = img.shape
            with torch.no_grad():
                features = torch.mean(image_model.forward_features(einops.rearrange(img, 'N T C H W -> (N T) C H W')), dim=(-1, -2))
            features = einops.rearrange(features, '(N T) F -> N T F', T = T)
            with torch.set_grad_enabled(mode == 'train'):
                output = policy_network(cap_in, attn, features)
                out = einops.rearrange(output.logits, 'N T P -> N P T')
                loss = torch.mean(torch.sum(criterion(out, cap_out) * attn, dim=-1) / torch.sum(attn, dim=-1))
                total_loss += loss.item() * N
                count += N
                total_accuracy += torch.mean(torch.sum((torch.argmax(out, dim=1) == cap_out) * attn, dim=-1) / torch.sum(attn, dim=-1)).item() * N    
                
                if mode == 'train':
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    lr_scheduler.step()

        total_loss /= count
        total_accuracy /= count
        print("Loss:", total_loss, "Acc:", total_accuracy * 100)
        if mode == 'val' and total_loss < bestLoss:
            bestLoss = total_loss
            torch.save(policy_network.state_dict(), "Models/policy_network.pt")
            print('----Best model saved.')

    print(f'####################################')

        
