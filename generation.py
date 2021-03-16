import argparse
import torch
from tqdm import tqdm
import data_loader.data_loaders as module_data
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_model
from parse_config import ConfigParser
from utils.util import beam_search, sample_sequence
import os, json

from transformers import *
from model.model_gpt import VideoGPT2LMHeadModel

SPECIAL_TOKENS = ["<bos>", "<eos>", "<que>", "<ans>", "<speaker>", 
                  "<subtitle>", "<video>", "<pad>"]
SPECIAL_TOKENS_DICT = {'bos_token': "<bos>", 'eos_token': "<eos>", 'additional_special_tokens': ["<que>", "<ans>", "<speaker>", "<subtitle>", "<video>"], 'pad_token': "<pad>"}


torch.multiprocessing.set_sharing_strategy('file_system')

def main(config):
    logger = config.get_logger('test')

    # setup data_loader instances
    data_loader = config.init_obj('data_loader', module_data, 'val')

    # build model architecture
    video = config['data_loader']['args']['inputs']

    checkpoint = '/data/results/models/AAAI2021/0311_183723/'
    tokenizer_class = GPT2Tokenizer
    tokenizer = tokenizer_class.from_pretrained(checkpoint)
    tokenizer.add_special_tokens(SPECIAL_TOKENS_DICT)

    checkpoint = '/data/results/models/AAAI2021/0311_183723/checkpoint-epoch10.pth'
    checkpoint = torch.load(checkpoint)
    model_class = VideoGPT2LMHeadModel

    model = model_class.from_pretrained('gpt2')
    model.resize_token_embeddings(len(tokenizer))

    state_dict = checkpoint['state_dict']
    model.load_state_dict(state_dict)
    model.resize_token_embeddings(len(tokenizer))

    # prepare model for testing
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    with open('/data/dataset/AnotherMissOh/AnotherMissOh_QA/AnotherMissOhQA_val_set_script.json', 'r') as f:
        data_set = json.load(f)

    text_data = dict()
    for d in data_set:
        text_data[d["qid"]] = d

    data = {}
    data['level1'] = []
    data['level2'] = []
    data['level3'] = []
    data['level4'] = []
    with torch.no_grad():
        tqdm_bar = tqdm(data_loader, desc='Test Epoch')
        iteration = 0
        for batch_idx, batch in enumerate(tqdm_bar):
            input_ids = batch[0].to(device)
            token_type_ids = batch[1].to(device)
            i3d = None
            level = 0
            if video:
                i3d = batch[5].to(device)            
                que = batch[8][0][0]
                level = batch[9][0]
            else: 
                que = batch[5][0][0]
                level = batch[9][0]
            output = {}
            target = batch[3][0]
            pred = sample_sequence(model, input_ids, token_type_ids, tokenizer, device, video=i3d)
            output['Question'] = tokenizer.decode(que, skip_special_tokens=True)
            output['Target'] = tokenizer.decode(target, skip_special_tokens=True)
            output['Prediction'] = tokenizer.decode(pred, skip_special_tokens=True)

            if level == 1 and len(data['level1']) < 25:
                data['level1'].append(output)
            elif level == 2 and len(data['level2']) < 25:
                data['level2'].append(output)               
            elif level == 3 and len(data['level3']) < 25:
                data['level3'].append(output)
            elif level == 4 and len(data['level4']) < 25:
                data['level4'].append(output)

            if iteration == 200:
                path = './level_output_gpt2.json'
                with open(path, 'w') as outfile:
                    json.dump(data, outfile, indent = 4)
                break
            iteration = iteration + 1

if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')

    config = ConfigParser.from_args(args)
    main(config)
