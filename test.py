import argparse
import torch
from tqdm import tqdm
import data_loader.data_loaders as module_data
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_model
from parse_config import ConfigParser
from utils.util import batch_to_device
import os, json 

torch.multiprocessing.set_sharing_strategy('file_system')

def main(config):
    logger = config.get_logger('test')

    # setup data_loader instances
    data_loader = config.init_obj('data_loader', module_data, 'val')

    # build model architecture
    model = config.init_obj('model', module_model, pt_emb=data_loader.vocab)
    logger.info(model)

    # get function handles of loss and metrics
    criterion = config.init_obj('loss', module_loss)
    metric_fns = [getattr(module_metric, met) for met in config['metrics']]

    logger.info('Loading checkpoint: {} ...'.format(config.resume))
    checkpoint = torch.load(config.resume)
    state_dict = checkpoint['state_dict']
    if config['n_gpu'] > 1:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(state_dict)

    # prepare model for testing
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    total_loss = 0.0
    total_metrics = torch.zeros(len(metric_fns))


    with open('./data/AnotherMissOh/AnotherMissOh_QA/AnotherMissOhQA_val_set_script.json', 'r') as f:
        data_set = json.load(f)

    text_data = dict()
    for d in data_set:
        text_data[d["qid"]] = d


    with torch.no_grad():
        tqdm_bar = tqdm(data_loader, desc='Test Epoch')

        for batch_idx, batch in enumerate(tqdm_bar):
            if 7879 not in batch['qid']:
                continue
            data, target = batch_to_device(config['data_loader']['args']['inputs'], batch, device)
            output, module_output = model(data)
            _, pred_idx = output.max(dim=1)
            # sub_low_batch, sub_high_batch, vis_low_batch, vis_high_batch = \
            #     module_output[:,:,0], module_output[:,:,1], module_output[:,:,2], module_output[:,:,3]

            sub_high_batch, vis_high_batch = \
                module_output[:,:,0], module_output[:,:,1]

            # q_level = data['q_level_logic']
            behemo = data['vgraphs']
            qids = batch['qid']

            beh_not_to_find = ['none', '<unk>', '<pad>']

            for i in range(len(batch['qid'])):
                # sub_low = sub_low_batch[i,:]
                sub_high = sub_high_batch[i,:]
                # vis_low = vis_low_batch[i,:]
                vis_high = vis_high_batch[i,:]

                targ = target[i].item()
                pred = pred_idx[i].item()

                # q_l = q_level[i]
                qid = qids[i]
                q_l = text_data[qid]["q_level_logic"]
                be = behemo[i]  # 30, 10, 3 # [char, beh, emo]
                be = be.view(-1, 3)
                behavior = []
                for j in range(be.size(0)):
                    behavior.append(be[j, 1])

                flag = 0
                sum_ = 0


                if qid != 7879:
                    continue
                # for beh in beh_not_to_find:
                #     sum_ += behavior.count(data_loader.vocab.get_index(beh))
                #     if behavior.count(data_loader.vocab.get_index(beh)) > 30:
                #         print(text_data[qid])
                #         flag = 1
                #         continue

                used_beh = set()
                for beh in behavior:
                    beh_word = data_loader.vocab.get_index(beh)
                    if beh_word not in beh_not_to_find:
                        used_beh.add(beh_word)

                #if len(used_beh) < 1:
                #    continue

                #if behavior.count(data_loader.vocab.get_index('<pad>')) == len(behavior):
                #    continue
                '''
                if targ != pred or q_l < 3 or flag == 1: # or \
                        # np.argmax(np.array(sub_high.cpu())) != targ or np.argmax(np.array(vis_high.cpu())) != targ:
                        # np.argmax(np.array(sub_low.cpu())) != targ or np.argmax(np.array(vis_low.cpu())) != targ or \
                    # max(b_c) < max(s_c) or max(s_c) > 0.8 or np.argmax(np.array(m_c.cpu())) != pred or np.argmax(np.array(b_c.cpu())) != pred:
                    # or q_l == 1
                    # or np.argmax(np.array(s_c.cpu())) == np.argmax(np.array(b_c.cpu())):
                    continue
                '''
                # if max(sub_low) >= 0.95 or max(sub_high) >= 0.95 or max(vis_low) >= 0.95 or max(vis_high) >= 0.95:
                #     print(text_data[qid])
                #     continue
                # if np.argmax(np.array(sub_high.cpu())) == np.argmax(np.array(sub_low.cpu())) and max(sub_low) >= max(sub_high):
                #     continue
                # if np.argmax(np.array(vis_high.cpu())) == np.argmax(np.array(vis_low.cpu())) and max(vis_low) >= max(vis_high):
                #     continue

                if text_data[qid]["videoType"] != "scene":
                    continue

                logfile = open(os.path.join('./qualitative', '4-high_val.txt'), 'a+')

                logfile.write('\tq_level : %d\n' % q_l)
                logfile.write('\tvid : %s\n' % text_data[qid]["vid"])
                s,e = text_data[qid]["shot_contained"]
                logfile.write('\tshot contained: %d %d\n' %(int(s), int(e)))
                logfile.write('\tque: %s\n' % text_data[qid]["que"])
                logfile.write('\tcorrect_idx: %d\n' % int(targ))
                logfile.write('\tprediction: %d\n' % int(pred))
                for k in range(5):
                    logfile.write('\tans%d: %s\n'% (k, text_data[qid]["answers"][k]))
                logfile.write('\tanswer score:\n')
                # logfile.write('\t sub_low_score: %.4f %.4f %.4f %.4f %.4f\n' % (sub_low[0].item(), sub_low[1].item(), sub_low[2].item(), sub_low[3].item(), sub_low[4].item()))
                logfile.write('\t sub_high_score: %.4f %.4f %.4f %.4f %.4f\n' % (sub_high[0].item(), sub_high[1].item(), sub_high[2].item(), sub_high[3].item(), sub_high[4].item()))
                # logfile.write('\t vis_low_score: %.4f %.4f %.4f %.4f %.4f\n' % (vis_low[0].item(), vis_low[1].item(), vis_low[2].item(), vis_low[3].item(), vis_low[4].item()))
                logfile.write('\t vis_high_score: %.4f %.4f %.4f %.4f %.4f\n' % (vis_high[0].item(), vis_high[1].item(), vis_high[2].item(), vis_high[3].item(), vis_high[4].item()))

                beh_str = ''
                for beh in behavior:
                    temp = str(data_loader.vocab.get_word(int(beh.item()))) + ', '
                    beh_str += temp
                logfile.write(beh_str)
                logfile.write('\n')

                if type(text_data[qid]['subtitle']) == str:
                    logfile.write(text_data[qid]['subtitle'])
                    logfile.write('\n')
                else:
                    for sub in text_data[qid]['subtitle']['contained_subs']:
                        logfile.write('%s: %s\n'%(sub['speaker'], sub['utter']))
                # if type(sub_text_data[qid]['subtitle']) == str:
                #     logfile.write(sub_text_data[qid]['subtitle'])
                #     logfile.write('\n')
                # else:
                #     for sub in sub_text_data[qid]['subtitle']['contained_subs']:
                #         logfile.write('%s: %s\n'%(sub['speaker'], sub['utter']))

                logfile.write('\n')

                logfile.close()



            #
            # save sample images, or do something with output here
            #
            # computing loss, metrics on test set
            '''
            loss = criterion(output, target)
            batch_size = data.shape[0]
            total_loss += loss.item() * batch_size
            for i, metric in enumerate(metric_fns):
                if 'accuracy_diff' not in metric.__name__:
                    total_metrics[i] += metric(output, target) * batch_size
                else:
                    total_metrics[i] += metric(output, target, data['q_level_logic']) * batch_size
            '''
                

    n_samples = len(data_loader.sampler)
    log = {'loss': total_loss / n_samples}
    log.update({
        met.__name__: total_metrics[i].item() / n_samples for i, met in enumerate(metric_fns)
    })
    logger.info(log)


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
