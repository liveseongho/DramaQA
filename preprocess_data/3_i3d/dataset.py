import glob
import os
import os.path
import json
import pickle
from tqdm import tqdm

import cv2
import numpy as np
import torch
import torch.utils.data as data_utl

config = json.load(open('../preprocess_config.json', 'r', encoding='utf-8'))
max_img_number = config['max_img_number']

def video_to_tensor(pic):
    """Convert a ``numpy.ndarray`` to tensor.
    Converts a numpy.ndarray (T x H x W x C)
    to a torch.FloatTensor of shape (C x T x H x W)

    Args:
         pic (numpy.ndarray): Video to be converted to tensor.
    Returns:
         Tensor: Converted video.
    """
    return torch.from_numpy(pic.transpose([3,0,1,2]))


def load_rgb_frames(root_dir,     # '/data/dataset/AnotherMissOh/AnotherMissOh_images',
                    img_dir,      # ex) AnotherMissOh01/001/0078/
                    inp_img_ids,  # ex) [76925, 76933, 76941, 76949, 76957]
                    start,        # 0
                    num):         # len(inp_img_ids)   ex) 5

    img_dir = img_dir.split('|')[0]

    frames = []

    for i in range(start, start+num):
        img_id = inp_img_ids[i]

        image_path = os.path.join(root_dir, img_dir, 'IMAGE_' + img_id)

        if i % 10 == 0:
            print('in progress:', image_path, end='\r')


        img = cv2.imread(image_path)[:, :, [2, 1, 0]]
        w,h,c = img.shape
        if w < 226 or h < 226:
            d = 226.-min(w,h)
            sc = 1+d/min(w,h)
            img = cv2.resize(img,dsize=(0,0),fx=sc,fy=sc)
        img = (img/255.)*2 - 1
        frames.append(img)
    if num < 8:
        w,h,c = frames[0].shape
        pad = np.zeros((w, h, c))

        first_img_id = inp_img_ids[0]
        pad_frames = []
        if num <= 4:
            for j in range(8-num):
                pad_frames.append(pad)
            pad_frames.extend(frames)
            frames = pad_frames
        else:
            for j in range(8-num):
                frames.append(pad)

        assert len(frames) == 8

    ret = np.asarray(frames, dtype=np.float32)
    # print('shape:', ret.shape)
    return ret


def load_i3dclip_ids(root_dir):
    """
    root_dir:  /data/dataset/AnotherMissOh/AnotherMissOh_images

    returns: i3d_inputs dict
    - key  : path       str
    - value: img_list   list(int)
    """

    i3d_inputs = {}
    all_ids = []

    for idx, path in tqdm(enumerate(sorted(glob.glob(root_dir + '/*/*/*/', recursive=True)))):
        
        
        # if idx < 2664:
        #     continue
        # if idx > 2667:
        #     break
        
        # if idx>=1:
        #     break

        key = '/'.join(path.split('/')[5:])
        # if int(key[13:15]) <= 8:
        #     continue

        all_ids.append(key[:-1])

        # image_ids = list(map(lambda x: key + x[-14:], list(glob.glob(path + '*.jpg'))))
        img_list = list(glob.glob(path + '*.jpg'))
        '''if len(img_list) > 8:
            import numpy as np
            idx_list = list(map(lambda x: int(x), np.linspace(0, len(img_list)-1, 8)))

            img_list = np.array(img_list)
            img_list = img_list[idx_list]
            img_list = list(img_list)
            if idx<2:
                print('idx_list:', idx_list)
                print('img_list:', len(img_list))
                from pprint import pprint
                pprint(img_list)
                print(type(img_list))
                print(type(img_list[0]))
                print(type(idx_list))
                print(type(idx_list[0]))'''
        image_ids = list(map(lambda x: str(x)[-14:], img_list))
        image_ids.sort()
        
        
        if len(image_ids) > max_img_number:
            for i in range((len(image_ids)-1) // max_img_number + 1):
                # print('exceeded: ', i*max_img_number, min((i+1) * max_img_number, len(image_ids)))
                i3d_inputs[key + '|' + str(i)] = image_ids[
                                                    i*max_img_number :
                                                    min((i+1) * max_img_number, len(image_ids))
                                                 ]
        else:
            i3d_inputs[key] = image_ids
        '''if idx<3:
            print(path)
            print(key)
            print(image_ids)'''


    return i3d_inputs



def gosgos():
    if not os.path.exists('./extract_features/i3d_perclip_ids.pkl'):
        i3d_inputs = {}
        all_ids = pickle.load(open('/data/TGIFQA/feat_r6_vsync2/tgif_sample_imgid2idx.pkl', 'rb'))

        for gif_id in all_ids.keys():
            imgs = os.listdir('/data/TGIFQA/frames_r6_vsync2/%s'%(gif_id+'.gi'))
            imgs = [int(img.split('.')[0]) for img in imgs]
            imgs = sorted(imgs)
            img_len = len(imgs)

            anchor_imgs = all_ids[gif_id].keys()

            for anchor_img in anchor_imgs:
                anchor_idx = int(anchor_img)-1
                input_imgs = imgs[max(0, anchor_idx-4):min(img_len, anchor_idx+4)] # front 4, back 3

                clip_imgs = []
                for img_id in input_imgs:
                    clip_imgs.append(img_id)

                dict_key = str(gif_id) + '/' + str(anchor_img)
                i3d_inputs[dict_key] = clip_imgs

        with open('./extract_features/i3d_perclip_ids.pkl', 'wb') as f:
            pickle.dump(i3d_inputs, f)
    else:
        i3d_inputs = pickle.load(open('./extract_features/i3d_perclip_ids.pkl', 'rb'))

    return i3d_inputs


class DramaQADataset(data_utl.Dataset):

    def __init__(self, base_dir, mode, transforms=None, save_dir=''):


        self.mode = mode
        self.root = base_dir  # /data/dataset/AnotherMissOh/AnotherMissOh_images
        self.save_dir = save_dir

        self.transforms = transforms

        self.data = load_i3dclip_ids(self.root)
        self.data_keys = list(self.data.keys())
        self.v_max_len = 80
        
        '''for i, (k,v) in enumerate(self.data.items()):
            if i>=3:
                break
            print(k, v)
        print(self.data_keys[:3])'''

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        img_dir = self.data_keys[index]
        frame_num = len(self.data[img_dir])
        gif_name = img_dir.split('/')[-1]

        print('index:', index)
        print('input_imgs_key:', img_dir)
        print('len(self.data[input_imgs_key]):', len(self.data[img_dir]))
        
        '''print('self.data[input_imgs_key]:')
        from pprint import pprint
        pprint(self.data[input_imgs_key])'''
        
        imgs = load_rgb_frames(self.root, img_dir, self.data[img_dir], 0, frame_num)
        print('img load done')
        imgs = self.transforms(imgs)
        print('img transform done')

        return img_dir, video_to_tensor(imgs)

    def __len__(self):
        return len(self.data_keys)