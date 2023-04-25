import json
from pytorch_grad_cam import GradCAM
import os
import timm
import numpy as np
import einops
import torch
import torch.nn as nn
from transformers import GPT2Config, GPT2LMHeadModel
from torch.utils.data import DataLoader
from tqdm import tqdm
from time import time
from dataloader import *
os.environ['TOKENIZERS_PARALLELISM'] = "false"
from reward import CaptionReward, id2string


# hypter-parameters
MAX_SEQ_LENGTH = 32
BATCH_SIZE = 128
transformer_hidden_size = 256
EOS_ID = 2
BOS_ID = 0
device = 'cuda:0'

max_lr = 0.001
weight_decay = 5e-4
n_epochs = 10


# load captions and image lists
captions_train = json.load(open('Data/COCO17/cap_dict_train.json', 'r'))
captions_train = {int(k):v for k, v in captions_train.items()}
captions_val = json.load(open('Data/COCO17/cap_dict_val.json', 'r'))
captions_val = {int(k):v for k, v in captions_val.items()}

orig = json.load(open('Data/COCO17/annotations/captions_train2017.json', 'r'))
images_train = sorted([item for item in orig['images'] if int(item['id']) in captions_train], key=lambda x: x['id'])

orig = json.load(open('Data/COCO17/annotations/captions_val2017.json', 'r'))
images_val = sorted([item for item in orig['images'] if int(item['id']) in captions_val], key=lambda x: x['id'])

dictionary = json.load(open('Data/COCO17/dictionary.json', 'r'))
dictionary = {int(k):v for k, v in dictionary.items()}
cap_reward = CaptionReward(dictionary, len(captions_train))

# load image model
image_model = timm.create_model('efficientnet_b0', pretrained=False).to(device)
image_model.load_state_dict(torch.load('Models/eff-b0.pth'))
_ = image_model.eval()

# load GradCAM
cam = GradCAM(model=image_model, target_layers=[image_model.blocks[6][-1]], use_cuda=True)

# Data loaders
dataset_train = COCODataset(captions_train, images_train, 'Data/COCO17/train2017/', cam=cam, device=device)
dataset_val = COCODataset(captions_val, images_val, 'Data/COCO17/val2017/', cam=cam, device=device)

dataloaders = {
    'train': DataLoader(dataset_train, batch_size=BATCH_SIZE, num_workers=6, collate_fn=dataset_train.collate_fn, shuffle=True),
    'val': DataLoader(dataset_val, batch_size=BATCH_SIZE, num_workers=6, collate_fn=dataset_val.collate_fn, shuffle=False)
}

# models
class PolicyNetwork(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        self.fc = nn.Linear(1280, 256)
    
    def forward(self, cap_in, attn, features):
        features = self.fc(features)
        return self.backbone(input_ids=cap_in, attention_mask=attn, encoder_hidden_states=features)

    def generate(self, features):
        features = self.fc(features)
        return self.backbone.generate(encoder_hidden_states=features)

class ValueNetwork(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        self.fc = nn.Linear(1280, 256)
        self.head = nn.Linear(256, 1)
    
    def forward(self, cap_in, attn, features):
        features = self.fc(features)
        hidden_states = self.backbone(input_ids=cap_in, attention_mask=attn, 
                                      encoder_hidden_states=features, output_hidden_states=True).hidden_states[-1]
        final_features = torch.sum(hidden_states * attn.unsqueeze(-1), dim=1) / torch.sum(attn, dim=-1, keepdim=True)
        return self.head(final_features)

# utility functions
def get_random_level(attn):
    rand_ids = (torch.rand(len(attn), device=device) * torch.sum(attn, dim=-1) + 0.99).long()
    new_attn = torch.ones(attn.shape, dtype=torch.int64)
    for i in range(len(rand_ids)):
        new_attn[i][:rand_ids[i]] = 1
    return new_attn.to(device)


def GenerateCaptionsWithBeamSearch(model, cap_in, features, beamSize=2, eos_id=EOS_ID):
    gen_caps = cap_in[:, 0:1].clone().cpu().long()
    N = len(cap_in)
    bz = len(gen_caps)
    candidates = [(gen_caps, torch.ones(N, 1, dtype=torch.int64).to('cpu'), 0)]
    for t in range(MAX_SEQ_LENGTH // 2 - 1):
        next_candidates = []
        for c in range(len(candidates)):
            with torch.no_grad():
                output = model(candidates[c][0].to(device), candidates[c][1].to(device), features).logits.cpu()
            probs, words = torch.topk(output[:, -1:, :], beamSize)
            for i in range(beamSize):
                cap_already_ended = torch.sum(eos_id == candidates[c][1], dim=-1).bool().float()
                attn = torch.cat((candidates[c][1], words[:, :, i] == eos_id), axis=1) 
                cap = torch.cat((candidates[c][0], words[:, :, i]), axis=1)
                score = candidates[c][2] - torch.log(probs[:, 0, i]) * (1 - cap_already_ended)
                next_candidates.append((cap, attn, score))
        #print(next_candidates)
        tmp = []
        for j in range(bz):
            best_beam = sorted([(item[0][j], item[1][j], item[2][j]) for item in next_candidates], key=lambda tup:tup[2].item())[:beamSize]
            tmp.append(best_beam)
        tmp = list(zip(*tmp))
        tmp = [list(zip(*item)) for item in tmp]
        # ordered_candidates = sorted(next_candidates, key=lambda tup:tup[2])
        candidates = [(torch.stack(list(item[0])), torch.stack(list(item[1])), torch.tensor(list(item[2])).float()) for item in tmp]#ordered_candidates[:beamSize]
    captions = [item[0] for item in candidates]
    attentions = [item[1] for item in candidates]
    return captions, attentions

# load policy and value networks
config = GPT2Config(vocab_size=VOCAB_SIZE, n_positions=32, n_embd=transformer_hidden_size, 
                    n_layer=8, n_head=4, eos_token_id=2, bos_token_id=0, add_cross_attention=True)
policy_backbone = GPT2LMHeadModel(config).to(device)
value_backbone = GPT2LMHeadModel(config).to(device)

policy_network = PolicyNetwork(backbone=policy_backbone).to(device)
policy_network.load_state_dict(torch.load('Models/policy_network.pt'))
_ = policy_network.train()

value_network = ValueNetwork(backbone=value_backbone).to(device)
value_network.load_state_dict(torch.load('Models/value_network.pt'))
_ = value_network.train()

# optimization
criterion = nn.MSELoss().to(device)
optimizer = torch.optim.Adam(value_network.parameters(), lr=max_lr / 10, weight_decay=weight_decay)
lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=max_lr, epochs=n_epochs, steps_per_epoch=len(dataloaders['train']))


curriculum = [2 * i for i in range(1, MAX_SEQ_LENGTH // 2)]


for level in curriculum:
    print(f"**************************** Curriculum learning, for level {level}:")
    print(f"********************************************************************")
    print(f"********************************************************************")
    bestLoss = 5.0
    for epoch in range(n_epochs):
        print(f'#### Epoch {epoch + 1}/{n_epochs}:')
        for mode in ['train', 'val']:
            total_loss = 0
            count = 0
            print(f"------------------{mode}----------------")
            batch_index = 0
            for item in tqdm(dataloaders[mode]):
                batch_index += 1
                data_batch = dataset_train.patch_batch(*item)
                img = data_batch['patched_images'].to(device)
                cap_in = data_batch['captions_in'].to(device)
                # cap_out = data_batch['captions_out'].to(device)
                attn = data_batch['attention_mask'].to(device)
                all_captions = data_batch['captions']
                print(len(all_captions))
                caplens = torch.sum(attn, dim=1)
                useful_samples = caplens >= level + 1
                if torch.sum(useful_samples) == 0:
                    continue
                cap_in = cap_in[useful_samples]
                attn = attn[useful_samples]
                # cap_out = cap_out[useful_samples]
                img = img[useful_samples]
                caplens = caplens[useful_samples]
                N, T, C, H, W = img.shape

                with torch.no_grad():
                    features = torch.mean(image_model.forward_features(einops.rearrange(img, 'N T C H W -> (N T) C H W')), dim=(-1, -2))
                features = einops.rearrange(features, '(N T) F -> N T F', T = T)
                t = np.random.randint(1, T + 1)
                features = features[:, :t]
                print(features.shape)
                episodicAvgLoss = 0
                for b in range(N):
                    log_probs = []
                    values = []
                    rewards = []
                    current_captions = cap_in[b:b+1, :caplens[b] - level]
                    current_attn = attn[b:b+1, :caplens[b] - level]
                    print('current_caption: ', current_captions)
                    for step in range(level):
                        with torch.set_grad_enabled(mode == 'train'):
                            logits = policy_network(current_captions, current_attn, features).logits
                            value = value_network(current_captions, current_attn, features)
                            probs = torch.softmax(logits, dim=-1)
                            dist = probs.cpu().detach().numpy()[0, 0]
                            action = np.random.choice(probs.shape[-1], p=dist)

                            gen_word = torch.from_numpy(np.array([action])).unqueeze(0).to(device)
                            current_captions = torch.cat([current_captions, gen_word], dim = 1)

                            log_prob = torch.log(probs[0, 0, action])
                            generated_caption = id2string(current_captions[0].cpu().numpy())
                            reward = cap_reward.cider(all_captions[b], generated_caption)
                            print("True Captions: ", all_captions[b])
                            print("Generated Caption: ", generated_caption)
                            print('CIDER loss = ', reward)
                            exit()

                            rewards.append(reward)
                            values.append(value)
                            log_probs.append(log_prob)

                    values = torch.FloatTensor(values).to(device)
                    rewards = torch.FloatTensor(rewards).to(device)
                    log_probs = torch.stack(log_probs).to(device)

                    advantage = values - rewards 
                    actorLoss = (-log_probs * advantage).mean()
                    criticLoss = 0.5 * (advantage ** 2).mean()
                    
                    loss = actorLoss + criticLoss
                    episodicAvgLoss += loss.item()/N
                    if mode == 'train':
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                print(epoch, ":", batch_index, ":", episodicAvgLoss)
    print(f"********************************************************************\n\n\n")