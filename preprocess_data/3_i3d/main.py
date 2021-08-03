import argparse
import datetime
import os
import json

import numpy as np
import nvidia_smi
import torch
import torch.nn as nn
from torchvision import transforms

import videotransforms
from dataset import DramaQADataset
from models import resnet

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
nvidia_smi.nvmlInit()
handle = nvidia_smi.nvmlDeviceGetHandleByIndex(0)
# card id 0 hardcoded here, there is also a call to get all available card ids, so we could iterate

def print_gpu_usage():
    res = nvidia_smi.nvmlDeviceGetUtilizationRates(handle)
    print('gpu: {}%, gpu-mem: {}%'.format(res.gpu, res.memory))

print_gpu_usage()
# from pynvml.smi import nvidia_smi
# nvsmi = nvidia_smi.getInstance()
# def getMemoryUsage():
#     usage = nvsmi.DeviceQuery("memory.used")["gpu"][0]["fb_memory_usage"]
#     return "%d %s" % (usage["used"], usage["unit"])
# print("Before GPU Memory: %s" % getMemoryUsage())
#
# def getMemoryUsage():
#     usage = nvsmi.DeviceQuery


parser = argparse.ArgumentParser()
parser.add_argument('-mode', type=str, default='rgb', choices=['rgb', 'flow'], help='rgb or flow')
parser.add_argument('-load_model', type=str, default='pretrained/i3d_r50_nl_kinetics.pth')
# parser.add_argument('-root', type=str, default='/data/dataset/AnotherMissOh/AnotherMissOh_images')
# parser.add_argument('-root', type=str, default='/data/dataset/AnotherMissOh/frame_v0.2')
parser.add_argument('-batch_size', type=int, default=1)
parser.add_argument('--gpu', type=str, default='2')

GPU_NUM = [0, 1] # 원하는 GPU 번호 입력
# device = torch.device('cuda:{}'.format(GPU_NUM) if torch.cuda.is_available() else 'cpu')
# torch.cuda.set_device(device) # change allocation of current GPU

# args = parser.parse_args()
# parser.add_argument('-save_dir', type=str, default='/data/dataset/AnotherMissOh/i3d_' + args.mode + '_v0.2')
args = parser.parse_args()
# os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu



config = json.load(open('preprocess_config.json', 'r', encoding='utf-8'))

max_clip_number = config['max_clip_number']

root_dir = config['root_dir']
frame_dir = config['frame_dir']
save_dir = config['i3d_rgb_dir'] if args.mode == 'rgb' else config['i3d_flow_dir']
os.makedirs(save_dir, exist_ok=True)
print('save_dir:', save_dir)

def run(mode, root, batch_size=1, parallel=True):

    # setup dataset
    test_transforms = transforms.Compose([videotransforms.CenterCrop(224)])

    dataset = DramaQADataset(root, mode, test_transforms, save_dir=save_dir)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

    dataloaders = {'all': dataloader}

    all_data_len = dataset.__len__()

    '''file_name = 'acqa_i3d_roi_hw7_perclip.hdf5'
    h = h5py.File(save_dir + '/' + file_name, 'w')
    i3d_roi_feat = h.create_dataset(
        'i3d_roi_feat', (all_data_len, 7, 7, 2048), 'f')

    file_name = 'acqa_i3d_avg_hw7_perclip.hdf5'
    h = h5py.File(save_dir + '/' + file_name, 'w')
    i3d_avg_feat = h.create_dataset(
        'i3d_avg_feat', (all_data_len, 400), 'f')'''

    # setup the model
    # net = resnet.i3_res50(400) # vanilla I3D ResNet50
    net = resnet.i3_res50_nl(400) # Nonlocal

    net.cuda()

    if parallel:
        net = nn.DataParallel(net, device_ids=GPU_NUM)

    start_t = datetime.datetime.today()
    print("start time:", str(start_t))

    for phase in ['all']:
        net.eval()  # Set model to evaluate mode

        # tot_loss = 0.0
        # tot_loc_loss = 0.0
        # tot_cls_loss = 0.0

        all_len = len(dataloaders[phase])
        print('all_len:', all_len)
        # indices = {}

        # Iterate over data.
        for i, data in enumerate(dataloaders[phase]):
    
    
            img_dir, inputs = data
            
            temp = img_dir[0].split('|')
            if len(temp) > 1:
                splited = True
            img_dir = temp[0]       # ex) AnotherMissOh03/045/0957/

            if i%1 == 0:
                print(img_dir, '\t', inputs.shape) # [1, 3, 8T, 224, 224]
    
            
            repeat_num = (inputs.shape[2]-1) // max_clip_number + 1


            save_path = os.path.join(save_dir, img_dir[:-1]
                                     .replace('/', '_')) + '.npy'
            stored_data = np.load(save_path) if os.path.isfile(save_path) \
                else np.zeros((0, 2048))
            
            with torch.no_grad():
                for r in range(repeat_num):
                    
                    try:
                        start_idx = r * max_clip_number
                        end_idx = min((r+1) * max_clip_number, inputs.shape[2])
                        # print('start_idx, end_idx:', start_idx, end_idx)
                        feat = net(inputs[
                                   :, :,
                                   start_idx : end_idx,
                                   :, :
                                   ])

                        # print('stored_data.shape:', stored_data.shape) # T, 2048
                        stored_data = np.concatenate((stored_data, feat.squeeze(1).data.cpu().numpy()), axis=0)
                        # torch.cuda.empty_cache()
                        
                    except RuntimeError as e:
                        print(e)
                        print('\nkey:', img_dir)
                        print('inputs.shape:', inputs.shape)
                        exit()
                    
                    # print(len(outputs))
                    #print(outputs[0].shape)
                    #print(outputs[1])
                    # (roi_feat, avg_feat, feat), loss_dict = outputs
                    # roi_feat, avg_feat, loss_dict = net(batch)       # net(batch): [1, 400], {}
                    # print('roi_feat.shape:', roi_feat.shape) # [1, 2048, T, 7, 7]
                    # print('avg_feat.shape:', avg_feat.shape) # [9, 400]
                    # roi_feat = roi_feat.squeeze(0).permute(1,2,3,0).data.cpu().numpy() # T, 7, 7, 2048
                    # avg_feat = avg_feat.squeeze().unsqueeze(0).data.cpu().numpy() # 1, 400
                    # print(roi_feat.shape, avg_feat.shape)
                    # print('roi_feat.shape:', roi_feat.shape) #
                    # print('avg_feat.shape:', avg_feat.shape) #
    
            np.save(save_path, stored_data)
            
            if i%1 == 0:
                print('stored_data.shape:', stored_data.shape) # T, 2048

                print("[{}/{}] {} saved".format(i+1, all_len, save_path), end='\t')
                print("time elapsed:", str(datetime.datetime.today() - start_t))
                print_gpu_usage()
                print('-' * 80)

            

            # indices[img_dir[0]] = i
            # feat_len = roi_feat.shape[0]
            # assert feat_len == 1
            # print(features.shape, feature_len)
            # i3d_roi_feat[i, :, :, :] = roi_feat
            # i3d_avg_feat[i, :] = avg_feat





            # with open(os.path.join(save_dir, data[1][0][:-1].replace('/', '_')), 'wb') as f:
            #     pickle.dump(indices, f)

        else:
            print("[%d/%d] clips done" % (i+1, all_len))


    # pkl_name = 'acqa_i3d_hw7_perclip_id2idx.pkl'
    # with open(os.path.join(save_dir, pkl_name), 'wb') as f:
    #     pickle.dump(indices, f)


if __name__ == '__main__':

    print('args:', args, sep='\n')

    run(mode=args.mode, root=frame_dir, batch_size=args.batch_size, parallel=True)