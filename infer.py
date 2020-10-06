import argparse
import torch
import os, json
from collections import defaultdict
from tqdm import tqdm
import data_loader.data_loaders as module_data
#import model.loss as module_loss
import model.metric as module_metric
import model.model as module_model
#import model.mdam as module_model
#import model_tvqa.tvqa_abc as module_model
#import model.baseline as module_model
from parse_config import ConfigParser
from utils.util import batch_to_device
import model.baseline as module_baseline

torch.multiprocessing.set_sharing_strategy('file_system')

def main(config):
    print(config['model'])
    logger = config.get_logger('test')

    # setup data_loader instances
    data_loader = config.init_obj('data_loader', module_data, 'test')

    if True: #config.test is None:
        # build model architecture
        model = config.init_obj('model', module_model, pt_emb=data_loader.vocab)
        logger.info(model)

        # get function handles of loss and metrics
        #loss_fn = getattr(module_loss, config['loss'])
        #criterion = config.init_obj('loss', module_loss)
        metric_fns = [getattr(module_metric, met) for met in config['metrics']]

        logger.info('Loading checkpoint: {} ...'.format(config.resume))
        checkpoint = torch.load(config.resume)
        state_dict = checkpoint['state_dict']
        if config['n_gpu'] > 1:
            model = torch.nn.DataParallel(model)
        model.load_state_dict(state_dict)
    else:
        model = config.init_obj('model', module_baseline, pt_emb=data_loader.vocab)
        logger.info(model)

        metric_fns = [getattr(module_metric, met) for met in config['metrics']]

        logger.info('Loading baseline: {}'.format('LongestAnswer'))

    # prepare model for testing
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    total_loss = 0.0
    total_metrics = torch.zeros(len(metric_fns))
    answers = {}

    with torch.no_grad():
        tqdm_bar = tqdm(data_loader, desc='Test Epoch')

        for batch_idx, batch in enumerate(tqdm_bar):
            data, _ = batch_to_device(config['data_loader']['args']['inputs'], batch, device)
            output = model(data)

            _, preds = output.max(dim=1)


            for qid, pred_idx in zip(batch['qid'], preds):
                answers[qid] = pred_idx.item()
            ans_path = './answers.json'#config.resume.parent / 'answers.json'

    with open(ans_path, 'w') as f:
        json.dump(answers, f, indent=4)
    print("Saved answers at {}".format(ans_path))

    hypo = answers
    gt = open_data("data/AnotherMissOh/AnotherMissOh_QA/AnotherMissOhQA_test_with_gt.json")

    hypo_keys = set(hypo.keys())
    gt_keys = set(gt.keys())
    assert not (gt_keys - hypo_keys), print("Keys missing: {}".format(gt_keys - hypo_keys))

    gt_dicts = divide_with_key(gt, 'q_level_logic')
    accs = {str(k): get_acc(hypo, v, k) for k, v in gt_dicts.items()}
    accs['total'] = [sum(v[0] for v in accs.values()), sum(v[1] for v in accs.values())]
    keys = sorted(list(accs.keys()))

    for k in keys:
        v = accs[k]
        if k == 'total':
            print("test_accuracy: {}".format(v[0]/v[1]))
        else:
            print("test_accuracy_diff{}: {}".format(k, v[0] / v[1]))


def open_data(path):
    path = os.path.expanduser(path)
    assert os.path.isfile(path), print("file does not exist: {}".format(path))
    with open(path, 'r') as f:
        data = json.load(f)
        if isinstance(data, list):
            data = {row['qid']: row for row in data}
    return data

def divide_with_key(dt, key):
    res = defaultdict(dict)
    for k, v in dt.items():
        res[v[key]][k] = v
    return res

def get_acc(hypo, gt, k):
    gt_keys = list(gt.keys())
    N = len(gt_keys)
    acc = [float(hypo[k] == gt[k]['correct_idx']) for k in gt_keys]
    return sum(acc), N




if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')
    args.add_argument('-t', '--test', default=None, type=str,
                      help='test model name (default: None)')
    config = ConfigParser.from_args(args)
    main(config)
