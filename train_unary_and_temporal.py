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
from lib.Temporal_Model import MyTempTransformer
#from lib.Temporal_Model_Encoder_Only import MyTempTransformer
#from lib.Temporal_Model_Decoder_Only import MyTempTransformer
from lib.Unary_Model_Combined import MyUnaryTransformer
#from lib.Unary_Model_Combined_Encoder_Only import MyUnaryTransformer
#from lib.Unary_Model_Combined_Decoder_Only import MyUnaryTransformer
from lib.forward_pass_utils import  compute_unary_loss, compute_temporal_loss
#from lib.forward_pass_utils_v5_only_encoder import  compute_unary_loss, compute_temporal_loss
#from lib.forward_pass_utils_v5_only_encoder_mlm_loss import  compute_unary_loss, compute_temporal_loss
#from lib.forward_pass_utils_v5_only_decoder import compute_unary_loss, compute_temporal_loss
#from lib.forward_pass_utils_v5_only_decoder_no_obj import compute_unary_loss, compute_temporal_loss
#from lib.forward_pass_utils_v5_no_obj import compute_unary_loss, compute_temporal_loss




conf = Config()
for i in conf.args:
    print(i,':', conf.args[i])



gpu_device = torch.device('cuda:0')

unary_prior_model = MyUnaryTransformer(num_encoder_layers=1,
                                               num_decoder_layers=1,
                                               emb_size=3472,
                                               nhead=8,
                                               tgt_vocab_size=64 # (37 + 3 + 6 + 17 )
                                               ).to(device=gpu_device)



temporal_prior_model = MyTempTransformer(num_encoder_layers=1,
                                               num_decoder_layers=1,
                                               emb_size=3472,
                                               nhead=8,
                                               att_tgt_vocab_size=64,
                                               spa_tgt_vocab_size=64,
                                               con_tgt_vocab_size=64).to(device=gpu_device)


loss_fn = torch.nn.CrossEntropyLoss()

temporal_optimizer = torch.optim.Adam(
        temporal_prior_model.parameters(),
        lr=0.00001, betas=(0.9, 0.98), eps=1e-9)
unary_optimizer = torch.optim.Adam(
        unary_prior_model.parameters(),
        lr=0.00001, betas=(0.9, 0.98), eps=1e-9)
temporal_scheduler = ReduceLROnPlateau(temporal_optimizer, "max", patience=1, factor=0.5, verbose=True, threshold=1e-4,
                              threshold_mode="abs", min_lr=1e-7)
unary_scheduler = ReduceLROnPlateau(unary_optimizer, "max", patience=1, factor=0.5, verbose=True, threshold=1e-4,
                              threshold_mode="abs", min_lr=1e-7)



checkpointdir = 'unary_and_temporal_prior_checkpoints/'
temporal_checkpointdir = checkpointdir + '/temporal/'  
unary_checkpointdir = checkpointdir + '/unary/'
if not os.path.exists(unary_checkpointdir):
    os.makedirs(unary_checkpointdir)
if not os.path.exists(temporal_checkpointdir):
    os.makedirs(temporal_checkpointdir)
# some parameters
tr = []
train_data_folder = 'results/' + conf.mode + '_backbone_training_with_dino/'
filelist = os.listdir(train_data_folder)
print('Loading results from ' + train_data_folder)


valList = []

with open('valFiles.txt', 'r') as fp:
    for line in fp:
        valfile = line[:-1]
        valList.append(valfile)


print("Validation files loaded")
t = TicToc()

def add_frame_keys(pred, frame_names):

    pred['frame_list'] = []
    for i in range(0,len(pred['human_idx'])):
        pred['frame_list'].append(frame_names[i])


    return pred


def train_unary():

    for epoch in range(0, 50):
        t.tic()
        random.shuffle(filelist)
        if epoch != 0:
            unary_model_path_epoch = unary_checkpointdir + '/unary_prior_model_' + str(epoch - 1) + '.pt'
            unary_model_ckpt = torch.load(unary_model_path_epoch, map_location=gpu_device)
            unary_prior_model.load_state_dict(unary_model_ckpt)
            print('Unary models loaded from epoch no: ' + str(epoch - 1) + 'for unary  training')


        unary_prior_model.train()

        unary_loss_att_iter = []
        unary_loss_spa_iter = []
        unary_loss_con_iter = []
        for b, filename in enumerate(tqdm(filelist)):
            if filename in valList:
                #print(filename)
                continue
            results = torch.load(train_data_folder + filename , map_location=torch.device('cpu'))
            pred_all = results[1]
            #print(pred_all.keys())
            unary_losses = compute_unary_loss(pred_all=pred_all, gpu_device=gpu_device, model=unary_prior_model, loss_fn=loss_fn)
            unary_optimizer.zero_grad()
            unary_loss = sum(unary_losses.values())
            unary_loss.backward()
            unary_optimizer.step()

            unary_loss_att_iter.append(unary_losses["attention_relation_loss"].item())
            unary_loss_spa_iter.append(unary_losses["spatial_relation_loss"].item())
            unary_loss_con_iter.append(unary_losses["contact_relation_loss"].item())

        un_loss_file_att_iter = unary_checkpointdir + '/att_unary_prior_loss_' + str(epoch)
        un_loss_file_spa_iter = unary_checkpointdir + '/spa_unary_prior_loss_' + str(epoch)
        un_loss_file_con_iter = unary_checkpointdir + '/con_unary_prior_loss_' + str(epoch)
        unary_model_path_epoch = unary_checkpointdir + '/unary_prior_model_' + str(epoch) + '.pt'
        un_att_file_iter = open(un_loss_file_att_iter, "wb")
        un_spa_file_iter = open(un_loss_file_spa_iter, "wb")
        un_con_file_iter = open(un_loss_file_con_iter, "wb")
        np.save(un_att_file_iter, unary_loss_att_iter)
        np.save(un_spa_file_iter, unary_loss_spa_iter)
        np.save(un_con_file_iter, unary_loss_con_iter)
        un_att_file_iter.close
        un_spa_file_iter.close
        un_con_file_iter.close
        torch.save(unary_prior_model.state_dict(), unary_model_path_epoch)
        t.toc()
    print('All Unary Epochs Done')

def train_temporal():
    for epoch in range(0, 50):
        t.tic()
        random.shuffle(filelist)
        if epoch != 0:
            temporal_model_path_epoch = temporal_checkpointdir + '/temp_prior_model_' + str(epoch - 1) + '.pt'
            temporal_model_ckpt = torch.load(temporal_model_path_epoch, map_location=gpu_device)
            temporal_prior_model.load_state_dict(temporal_model_ckpt)
            print('Temporal Models loaded from epoch no: ' + str(epoch - 1) )

        temporal_prior_model.train()
        temporal_loss_att_iter = []
        temporal_loss_spa_iter = []
        temporal_loss_con_iter = []

        for b, filename in enumerate(tqdm(filelist)):
            if filename in valList:
                #print(filename)
                continue
            results = torch.load(train_data_folder + filename, map_location=torch.device('cpu'))
            pred_all = results[1]


            temporal_losses = compute_temporal_loss(pred_all=pred_all, gpu_device=gpu_device, model=temporal_prior_model,
                                              loss_fn=loss_fn)
            temporal_optimizer.zero_grad()
            temporal_loss = sum(temporal_losses.values())
            temporal_loss.backward()
            temporal_optimizer.step()

            temporal_loss_att_iter.append(temporal_losses["attention_relation_loss"].item())
            temporal_loss_spa_iter.append(temporal_losses["spatial_relation_loss"].item())
            temporal_loss_con_iter.append(temporal_losses["contact_relation_loss"].item())




        temporal_model_path_epoch = temporal_checkpointdir + '/temp_prior_model_' + str(epoch) + '.pt'
        temp_att_file_iter = open(temporal_checkpointdir + '/att_temp_prior_loss_' + str(epoch), "wb")
        temp_spa_file_iter = open(temporal_checkpointdir + '/spa_temp_prior_loss_' + str(epoch), "wb")
        temp_con_file_iter = open(temporal_checkpointdir + '/con_temp_prior_loss_' + str(epoch), "wb")
        np.save(temp_att_file_iter, temporal_loss_att_iter)
        np.save(temp_spa_file_iter, temporal_loss_spa_iter)
        np.save(temp_con_file_iter, temporal_loss_con_iter)
        temp_att_file_iter.close
        temp_spa_file_iter.close
        temp_con_file_iter.close
        torch.save(temporal_prior_model.state_dict(), temporal_model_path_epoch)
        t.toc()
    print('All Temporal Epochs Done')

if __name__ == "__main__":

    train_unary()
    train_temporal()
