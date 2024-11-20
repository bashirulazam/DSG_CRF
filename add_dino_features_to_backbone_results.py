import pickle
import torch
import numpy as np
import os
import scipy.io as sio
import random
from tqdm import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau
from dataloader.action_genome import AG, cuda_collate_fn
from pytictoc import TicToc
from lib.config import Config




conf = Config()
for i in conf.args:
    print(i,':', conf.args[i])



gpu_device = torch.device('cuda:0')


# some parameters
tr = []



def add_frame_keys(pred, frame_names):

    pred['frame_list'] = []
    final_frame_ind = int(pred['im_idx'][-1] + 1)
    for i in range(0,final_frame_ind):
        pred['frame_list'].append(frame_names[i])

    return pred



def add_dino_features(in_folder,out_folder,dataloader,dino_feat_length = 1536):

    for b, data in enumerate(tqdm(dataloader)):
        frame_names = data[5]
        filename = frame_names[0].split('/')[0]
        filename = filename.split('.')[0] + '.pt'
        results = torch.load(in_folder + filename , map_location=torch.device('cpu'))
        pred_all = results[1]
        print(pred_all.keys())
        pred_all = add_frame_keys(pred_all, frame_names)
        gt_annotation = results[0]
        print(pred_all['rel_feat'].shape)
        final_frame_ind = int(pred_all['im_idx'][-1] + 1)
        dino_catted_pred_rel_feat_all = torch.zeros([pred_all['rel_feat'].shape[0], dino_feat_length + pred_all['rel_feat'].shape[1]])
        for frame_ind in range(0, final_frame_ind):
            pred_frame_ind = pred_all['im_idx'] == frame_ind
            pred_pairs = pred_all['pair_idx'][pred_frame_ind].cpu().clone().numpy()
            no_rels = pred_pairs.shape[0]
            vid_name = pred_all['frame_list'][frame_ind].split('/')[0].split('.')[0]
            frame_name = pred_all['frame_list'][frame_ind].split('/')[1].split('.')[0]
            dino_feat_path = os.path.join('results/dino_features', vid_name, frame_name)
            dino_feat = torch.tensor(np.load(dino_feat_path))
            pred_rel_feat = pred_all['rel_feat'][pred_frame_ind]
            dino_catted_pred_rel_feat = torch.zeros([pred_rel_feat.shape[0],dino_feat.shape[1]+pred_rel_feat.shape[1]])
            for l in range(0, no_rels):
                curr_feat = pred_rel_feat[l].flatten().reshape(1, -1)
                curr_feat = torch.cat((curr_feat, dino_feat), 1)
                dino_catted_pred_rel_feat[l, :] = curr_feat

            dino_catted_pred_rel_feat_all[pred_frame_ind] = dino_catted_pred_rel_feat


        pred_all['rel_feat'] = dino_catted_pred_rel_feat_all

        print(pred_all['rel_feat'].shape)

        result = [gt_annotation, pred_all]
        torch.save(result, out_folder + filename)

if __name__ == "__main__":

    train_data_folder = 'results/' + conf.mode + '_backbone_training/'
    train_data_folder_with_dino = 'results/' + conf.mode + '_backbone_training_with_dino/'
    if not os.path.exists(train_data_folder_with_dino):
        os.makedirs(train_data_folder_with_dino)
    
    AG_dataset = AG(mode="train", datasize=conf.datasize, data_path=conf.data_path, filter_nonperson_box_frame=True,
                     filter_small_box=False if conf.mode == 'predcls' else True)
    dataloader = torch.utils.data.DataLoader(AG_dataset, shuffle=False, num_workers=0, collate_fn=cuda_collate_fn)
    add_dino_features(in_folder=train_data_folder, out_folder=train_data_folder_with_dino, dataloader=dataloader)

    test_data_folder = 'results/' + conf.mode + '_backbone_testing/'
    test_data_folder_with_dino = 'results/' + conf.mode + '_backbone_testing_with_dino/'
    if not os.path.exists(test_data_folder_with_dino):
        os.makedirs(test_data_folder_with_dino)

    AG_dataset = AG(mode="test", datasize=conf.datasize, data_path=conf.data_path, filter_nonperson_box_frame=True,
                    filter_small_box=False if conf.mode == 'predcls' else True)
    dataloader = torch.utils.data.DataLoader(AG_dataset, shuffle=False, num_workers=0, collate_fn=cuda_collate_fn)
    add_dino_features(in_folder=test_data_folder, out_folder=test_data_folder_with_dino, dataloader=dataloader)
