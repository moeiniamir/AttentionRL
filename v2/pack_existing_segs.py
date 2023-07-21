#!/usr/bin/env python
# coding: utf-8

# In[1]:


# parallelize the above code over 8 processes
from multiprocessing import Pool
from pathlib import Path
import numpy as np
import pickle
import os
import datetime


# In[4]:


def pack_seg_out(seg_out):
    seg = np.stack(seg_out['masks'], axis=0)
    new_seg_out = {
        'packed_masks': np.packbits(seg, axis=1),
        'original_shape': seg.shape,
        'scores': seg_out['scores'],
    }
    return new_seg_out


# In[15]:


def unpack_new_seg_out(new_seg_out):
    original_shape = new_seg_out['original_shape']
    seg = np.unpackbits(new_seg_out['packed_masks'], axis=1)[:, :original_shape[1], :original_shape[2]]
    return seg


# In[ ]:


def pack_and_replace(file):
    try:
        with open(file, 'rb') as f:
            seg_out = pickle.load(f)
    except Exception as e:
        print('error while loading', file)
        raise e
    if 'packed_masks' in seg_out:
        return
    try:
        new_seg_out = pack_seg_out(seg_out)
    except Exception as e:
        print('error while packing', file)
        raise e
    try:
        with open(file, 'wb') as f:
            pickle.dump(new_seg_out, f)
    except Exception as e:
        print('error while saving', file)
        raise e


# In[5]:


if __name__ == '__main__':
    # iterate over all files in ../Data/COCO17/train2017seg/
    files = [file for file in Path('../Data/COCO17/train2017seg/').iterdir()]
    # separate files that are older than 15:00:00 may 22 2023
    files = [file for file in files if os.path.getmtime(file) < datetime.datetime(2023, 5, 22, 15, 0, 0).timestamp()]


# In[ ]:


if __name__ == '__main__':
    with Pool(18) as p:
        p.map(pack_and_replace, files)

