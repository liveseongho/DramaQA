from collections import defaultdict

from PIL import Image
from tqdm import tqdm

import torch
from torch import nn, utils
from torchvision import models, transforms
from utils import *

from .modules_vision import VisionDataset

image_types = ['full_image', 'person_full']
image_size = [224, 224]
delimiter = '/'

def dict_for_each_episode():
    return [dict() for i in range(18 + 1)]  # episode index: from 1 to 18

def get_model(args):
    print('Loading extractor model: using resnet18')

    model = models.resnet18(pretrained=True)
    extractor = nn.Sequential(*list(model.children())[:-2])
    extractor.to(args['device'])

    return extractor

def preprocess_images(args):
    #print('Loading visual')
    visuals = load_visual(args)

    image_path = Path(args['image_path'])
    cache_dir = Path(image_path) / 'cache'
    if not cache_dir.is_dir():
        cache_dir.mkdir()

    cached = {}
    not_cached = {}
    ext = '.pickle'
    for key in image_types:
        cache_path = cache_dir / (key + ext)
        if cache_path.is_file():
            cached[key] = cache_path
        else:
            not_cached[key] = cache_path

    features = {key: defaultdict(dict) for key in image_types}

    for key, path in cached.items():
        print("Loading %s feature cache" % key)
        features[key] = load_pickle(path)

    if not_cached: # not_cached not empty: some image types are not cached
        not_cached_types = ', '.join(not_cached)
        print('%s feature cache missing' % not_cached_types)
        print('Loading image files and extracting %s features' % not_cached_types)

        transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        model = get_model(args)
        episode_paths = list(image_path.glob('*'))
        for e in tqdm(episode_paths, desc='Episode'):
            shot_paths = list(e.glob('*/*'))  # episode/scene/shot

            # Load image and flatten
            images = load_images(shot_paths)
            images = {"{}{}{}".format(vid, delimiter, name): image for vid, shots in images.items() for name, image in shots.items()}
            dataset = ObjectDataset(args, images, visuals, not_cached, transform=transform)

            chunk = extract_features(args, dataset, model)

            for key in image_types:
                features[key].update(chunk[key])
                #for episode_total, episode_part in zip(features[key], chunk[key]):
                #    episode_total.update(episode_part)

            del images, dataset # delete data to retrieve memory
        del model # delete extractor model to retrieve memory

        if args['save_cache']:
            for key, path in not_cached.items():
                print("Saving %s feature cache as %s" % (key, path))
                save_pickle(features[key], path)

    return features, visuals

def load_images(shot_paths):
    """
    images = {
        shot1: {
            frame_id1: PIL image1,
            ...
        },
        ...
    }
    """

    images = list(tqdm(map(load_image, shot_paths), total=len(shot_paths), desc='loading images'))
    images = {k: v for k, v in images}

    return images

def load_image(shot_path):
    """
    res = {
        frame_id1: PIL image1,
        ...
    }
    """

    image_paths = shot_path.glob('*')
    vid = '_'.join(shot_path.parts[-3:])
    res = {}
    image_paths = sorted(list(image_paths))
    for image_path in image_paths:
        name = image_path.parts[-1] # name ex) IMAGE_0000046147.jpg
        image = Image.open(image_path)
        res[name] = image.copy()
        image.close()

    return (vid, res)

def load_visual(args):
    visual = read_json(args['visual_path'])
    visual_by_episode = dict_for_each_episode()

    for shot, frames in visual.items():
        episode = get_episode_id(shot)
        episode_dict = visual_by_episode[episode]

        for frame in frames:
            frame_id = get_frame_id(frame['frame_id'])
            episode_dict[frame_id] = frame

    return visual_by_episode

class ObjectDataset(VisionDataset):
    def __init__(self, args, images, visuals, not_cached, **kwargs):
        super(ObjectDataset, self).__init__('~/', **kwargs)

        self.args = args
        self.images = list([(k, v) for k, v in images.items()])
        self.visuals = visuals
        self.not_cached = not_cached

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        key, pil_full_image = self.images[idx]

        episode = get_episode_id(key)
        frame = get_frame_id(key)
        visual = self.visuals[episode].get(frame, None)
        data = {'key': (key[:19], key[:24], episode, frame)}
        assert get_shot_id(key) != 0, key

        if self.transform is not None:
            full_image = self.transform(pil_full_image)

        if 'full_image' in self.not_cached:
            data['full_image'] = full_image

        if 'person_full' in self.not_cached:
            data['person_full'] = self.get_person_full(pil_full_image, visual, full_image) # use full image for padding

        return data

    def collate_fn(self, batch):
        collected = defaultdict(list)
        for data in batch:
            for key, value in data.items():
                collected[key].append(value)

        if 'full_image' in self.not_cached:
            collected['full_image'] = torch.stack(collected['full_image'])

        return collected

    def get_person_full(self, pil_full_image, visual, padding):
        person_fulls = []
        if visual is not None:
            persons = visual["persons"]
            for p in persons:
                full_rect = p["person_info"]["full_rect"]

                if full_rect["max_x"] != '':
                    person_full = transforms.functional.crop(pil_full_image, *self.bbox_transform(full_rect))
                    if self.transform is not None:
                        person_full = self.transform(person_full)
                else: # no bounding box
                    person_full = padding

                person_fulls.append(person_full)

        if not person_fulls: # empty (no visual data or no person)
            person_fulls.append(padding)

        person_fulls = torch.stack(person_fulls)

        return person_fulls

    def bbox_transform(self, rect):
        """min_x, min_y, max_x, max_y -> top left corner coordinates, height, width"""

        top_left_v = rect["min_y"]
        top_left_h = rect["min_x"]
        height = rect["max_y"] - top_left_v
        width = rect["max_x"] - top_left_h

        return top_left_v, top_left_h, height, width


def mean_pool(tensor, dim):
    return torch.mean(tensor, dim=dim, keepdim=False)

def extract_and_pool(tensor, model, device):
    tensor = tensor.to(device)
    tensor = model(tensor)          # N x C x H x W (N: extractor_batch_size / number of person fulls in a frame, C: 512)
    tensor = mean_pool(tensor, -1)  # N x C x H
    tensor = mean_pool(tensor, -1)  # N x C
    tensor = tensor.cpu().numpy()

    return tensor

def extract_features(args, dataset, model):
    """
    full_images_by_episode = {
        scene_id: {
            vid: {
                frame_id: vector, # shape: (C,)
                ...
            }
            ...
        }
        ...
    }

    person_fulls_by_episode = {
        scene_id: {
            vid: {
                frame_id: matrix, # shape: (N, C) N: number of person
                ...
            }
            ...
        }
        ...
    }
    """

    device = args['device']
    not_cached = dataset.not_cached

    dataloader = utils.data.DataLoader(
        dataset,
        batch_size=args['extractor_batch_size'],
        shuffle=False,
        num_workers=args['extractor_workers'],
        collate_fn=dataset.collate_fn
    )

    model.eval()

    features = {key: defaultdict(dict) for key in image_types}
    with torch.no_grad():
        for data in tqdm(dataloader, desc='extracting features'):
            keys = data['key']

            if 'full_image' in not_cached:
                full_images = extract_and_pool(data['full_image'], model, device)
                for (scene_id, vid, e, f), fi, in zip(keys, full_images):
                    if scene_id in features['full_image']:
                        if vid in features['full_image'][scene_id]:
                            features['full_image'][scene_id][vid][f] = fi
                        else:
                            features['full_image'][scene_id][vid] = defaultdict(dict)
                            features['full_image'][scene_id][vid][f] = fi
                    else:
                        features['full_image'][scene_id] = defaultdict(dict)
                        features['full_image'][scene_id][vid] = defaultdict(dict)
                        features['full_image'][scene_id][vid][f] = fi


            if 'person_full' in not_cached:
                person_fulls = [extract_and_pool(pfu, model, device) for pfu in data['person_full']]
                for (scene_id, vid, e, f), pfu in zip(keys, person_fulls):
                    if scene_id in features['full_image']:
                        if vid in features['person_full'][scene_id]:
                            features['person_full'][scene_id][vid][f] = pfu
                        else:
                            features['person_full'][scene_id][vid] = defaultdict(dict)
                            features['person_full'][scene_id][vid][f] = pfu
                    else:
                        features['person_full'][scene_id] = defaultdict(dict)
                        features['person_full'][scene_id][vid] = defaultdict(dict)
                        features['person_full'][scene_id][vid][f] = pfu

    del dataloader

    return features

