from collections import defaultdict

from PIL import Image
from tqdm import tqdm
import gc

import torch
from torch import nn, utils
from torchvision import models, transforms
from utils import *
from os.path import isfile

from .modules_vision import VisionDataset

image_types = ['full_image', 'person_full']
image_size = [224, 224]
delimiter = '/'

modes = ['train', 'val', 'test']
sos_token = '<sos>'
eos_token = '<eos>'
pad_token = '<pad>'
unk_token = '<unk>'

def dict_for_each_episode():
    return [dict() for i in range(18 + 1)]  # episode index: from 1 to 18

def get_model(args):
    print('Loading extractor model: using resnet18')

    model = models.resnet18(pretrained=True)
    extractor = nn.Sequential(*list(model.children())[:-2])
    extractor.to(args['device'])

    return extractor

def preprocess_images(args, visuals):
    #print('Loading visual')

    image_path = Path(args['image_path'])
    #cache_dir = Path(image_path) / 'cache'
    cache_dir = Path(args['image_cache_path'])
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
            gc.collect()
        del model # delete extractor model to retrieve memory
        gc.collect()

        if args['save_cache']:
            for key, path in not_cached.items():
                print("Saving %s feature cache as %s" % (key, path))
                save_pickle(features[key], path)

    return features

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
    gc.collect()

    return features


def process_video(args, save_path, feature_path, speaker_index, vocab):
    pad_index = vocab.get_index(pad_token)
    none_index = speaker_index['None']

    visuals = load_visual(args)
    if isfile(feature_path):
        features = load_pickle(feature_path)
    else:
        print("saving processed_video ...")
        features = preprocess_images(args, visuals)
        save_pickle(features, feature_path)

    """
    {
        full_image:   full_image (tensor of shape (512,)),
        person_list:      [[person1_id_idx, behavior1_idx, emotion1_idx], ...],
        person_fulls: [person_full1 (tensor of shape (512,)), ... ]
    }
    """
    full_images = features['full_image']
    person_fulls = features['person_full']

    new_visuals = defaultdict(dict)
    for i in range(1, 19):
        for key, value in visuals[i].items():
            frame_id = value['frame_id']
            scene_id = frame_id[:19]
            vid = frame_id[:24]
            f = get_frame_id(frame_id)


            if scene_id in new_visuals:
                if vid in new_visuals[scene_id]:
                    new_visuals[scene_id][vid][f] = value
                else:
                    new_visuals[scene_id][vid] = defaultdict(dict)
                    new_visuals[scene_id][vid][f] = value
            else:
                new_visuals[scene_id] = defaultdict(dict)
                new_visuals[scene_id][vid] = defaultdict(dict)
                new_visuals[scene_id][vid][f] = value
    visuals = new_visuals

    for scene_vid, vid_dict in full_images.items():
        for vid, frames in vid_dict.items():
            for key, value in frames.items():
                frames[key] = {
                    'full_image': value,
                    'person_list': [],
                    'person_fulls': [],
                    'object_list': [],
                    'place' : ''
                }
                if vid not in visuals[scene_vid] or key not in visuals[scene_vid][vid]:
                    continue

                visual = visuals[scene_vid][vid][key]
                processed_p = frames[key]['person_list']

                for person in visual["persons"]:
                    person_id = person['person_id'].title()
                    person_id_idx = none_index if person_id == '' else speaker_index[person_id]  # none -> None

                    person_info = person['person_info']

                    behavior = person_info['behavior'].lower()
                    behavior_idx = pad_index if behavior == '' else vocab.get_index(behavior.split()[0])

                    emotion = person_info['emotion'].lower()
                    emotion_idx = pad_index if emotion == '' else vocab.get_index(emotion)

                    if len(person['related_objects']) > 0:
                        related_obj_id = person['related_objects'][0]['related_object_id'].lower()
                        related_obj_id_idx = vocab.get_index(related_obj_id)
                        relation = person['person_info']['predicate'].lower()
                        relation_idx = vocab.get_index(relation)
                    else:
                        related_obj_id_idx = pad_index
                        relation_idx = pad_index

                    processed = [person_id_idx, behavior_idx, emotion_idx, related_obj_id_idx, relation_idx] # Don't convert visual to a tensor yet
                    processed_p.append(processed)

                if processed_p:
                    frames[key]['person_fulls'] = list(person_fulls[scene_vid][vid][key])

                obj_list = list()
                for obj in visual["objects"]:
                    obj_id = obj['object_id']
                    obj_id_idx = vocab.get_index(obj_id)
                    obj_list.append(obj_id_idx)

                frames[key]['object_list'] = obj_list
                frames[key]['place'] = pad_index if visual['place'] == '' else vocab.get_index(visual['place'])

    vids_list = list()
    qa_paths = {m: Path(args['qa_path']) / 'AnotherMissOhQA_{}_set.json'.format(m) for m in modes}
    qa_set = list()

    for mode in modes:
        qa_set.append(read_json(qa_paths[mode]))

    for i, set_og in enumerate(qa_set):
        vids_l = set()
        for d in set_og:
            vids_l.add(d['vid'][:-5])
        vids_list.append(vids_l)

    train_vids, val_vids, test_vids = vids_list
    scene_vids = {'train': vids_list[0], 'val': vids_list[1], 'test': vids_list[2]}

    full_images_by_modes = {mode: {k: full_images[k] for k in scene_vids[mode]} for mode in modes}
    for mode in modes:
        save_pickle(full_images_by_modes[mode], save_path[mode])


def process_video_gpt(args, save_path, speaker_index, pad_index, tokenizer):
    none_index = speaker_index['None']

    print("saving processed_video ...")
    features, visuals = preprocess_images(args)
    #print('features : ', features)
    """
    {
        full_image:   full_image (tensor of shape (512,)),
        persons:      [[person1_id_idx, behavior1_idx, emotion1_idx], ...],
        person_fulls: [person_full1 (tensor of shape (512,)), ... ]
    }
    """
    full_images = features['full_image']
    person_fulls = features['person_full']

    new_visuals = defaultdict(dict)
    for i in range(1, 19):
        for key, value in visuals[i].items():
            frame_id = value['frame_id']
            scene_id = frame_id[:19]
            vid = frame_id[:24]
            f = get_frame_id(frame_id)


            if scene_id in new_visuals:
                if vid in new_visuals[scene_id]:
                    new_visuals[scene_id][vid][f] = value
                else:
                    new_visuals[scene_id][vid] = defaultdict(dict)
                    new_visuals[scene_id][vid][f] = value
            else:
                new_visuals[scene_id] = defaultdict(dict)
                new_visuals[scene_id][vid] = defaultdict(dict)
                new_visuals[scene_id][vid][f] = value
    visuals = new_visuals

    for scene_vid, vid_dict in full_images.items():
        for vid, frames in vid_dict.items():
            for key, value in frames.items():
                frames[key] = {
                    'full_image': value,
                    'persons': [],
                    'person_fulls': []
                }
                if vid not in visuals[scene_vid] or key not in visuals[scene_vid][vid]:
                    continue

                visual = visuals[scene_vid][vid][key]
                processed_p = frames[key]['persons']

                for person in visual["persons"]:
                    person_id = person['person_id'].title()
                    person_id_idx = none_index if person_id == '' else tokenizer.convert_tokens_to_ids(person_id)  # none -> None

                    person_info = person['person_info']

                    behavior = person_info['behavior'].lower()
                    behavior_idx = pad_index if behavior == '' else tokenizer.convert_tokens_to_ids(behavior.split()[0])

                    emotion = person_info['emotion'].lower()
                    emotion_idx = pad_index if emotion == '' else tokenizer.convert_tokens_to_ids(emotion)

                    processed = [person_id_idx, behavior_idx, emotion_idx] # Don't convert visual to a tensor yet
                    processed_p.append(processed)

                if processed_p:
                    frames[key]['person_fulls'] = list(person_fulls[scene_vid][vid][key])

    vids_list = list()
    qa_paths = {m: Path(args['qa_path']) / 'AnotherMissOhQA_{}_set.json'.format(m) for m in modes}
    qa_set = list()

    for mode in modes:
        qa_set.append(read_json(qa_paths[mode]))

    for i, set_og in enumerate(qa_set):
        vids_l = set()
        for d in set_og:
            vids_l.add(d['vid'][:-5])
        vids_list.append(vids_l)

    train_vids, val_vids, test_vids = vids_list
    scene_vids = {'train': vids_list[0], 'val': vids_list[1], 'test': vids_list[2]}
#    train_vids, val_vids = vids_list
#    scene_vids = {'train': vids_list[0], 'val': vids_list[1]}

    # for Vip
#    full_images_by_modes = {mode: {k: full_images[k] for k in scene_vids[mode]} for mode in modes}
#    full_images = {}
#    for mode in modes:
#        full_images.update(full_images_by_modes[mode])
#    save_pickle(full_images, save_path)

    # for Open-ended
    full_images_by_modes = {mode: {k: full_images[k] for k in scene_vids[mode]} for mode in modes}
    for mode in modes:
        save_pickle(full_images_by_modes[mode], save_path[mode])


