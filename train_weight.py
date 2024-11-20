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
#from lib.Unary_Model_Att import MyUnaryAttTransformer
#from lib.Unary_Model_Att_Encoder_Only import MyUnaryAttTransformer
#from lib.Unary_Model_Att_Decoder_Only import MyUnaryAttTransformer
#from lib.forward_pass_utils_v4_with_dino import  compute_unary_loss, compute_temporal_loss
from lib.forward_pass_utils import perform_unary_inference, perform_temporal_inference, perform_temporal_inference_with_unary, compute_combined_loss, combine_predictions_with_weights, compute_weight
#from lib.forward_pass_utils_v5_only_encoder import  compute_unary_loss, compute_temporal_loss
#from lib.forward_pass_utils_v5_only_encoder_mlm_loss import  compute_unary_loss, compute_temporal_loss
#from lib.forward_pass_utils_v5_only_decoder import compute_unary_loss, compute_temporal_loss
#from lib.forward_pass_utils_v5_only_decoder_no_obj import compute_unary_loss, compute_temporal_loss
#from lib.forward_pass_utils_v5_no_obj import compute_unary_loss, compute_temporal_loss

from lib.weight_model import OurWeight


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
#
#
#
temporal_prior_model = MyTempTransformer(num_encoder_layers=1,
                                               num_decoder_layers=1,
                                               emb_size=3472,
                                               nhead=8,
                                               att_tgt_vocab_size=64,
                                               spa_tgt_vocab_size=64,
                                               con_tgt_vocab_size=64).to(device=gpu_device)
dino_feat_length = 3472
weight_model = OurWeight(feat_length=dino_feat_length).to(device=gpu_device)
weight_model.train()
unary_prior_model.eval()
temporal_prior_model.eval()
# checkpointdir = 'best_models_' + conf.mode + '/'
# unary_model_path = checkpointdir + '/best_model_unary_only.pt'
unary_model_ckpt = torch.load(conf.unary_model_path, map_location=gpu_device)
unary_prior_model.load_state_dict(unary_model_ckpt)
print('Unary models loaded from path :' + conf.unary_model_path)
#
#
#temporal_model_path = checkpointdir + '/best_model_temporal_only_first_order.pt'
temporal_model_ckpt = torch.load(conf.temporal_model_path, map_location=gpu_device)
temporal_prior_model.load_state_dict(temporal_model_ckpt)
print('Temporal Models loaded from path :' + conf.temporal_model_path)

loss_fn = torch.nn.CrossEntropyLoss()
#loss_fn = torch.nn.NLLLoss()
model_optimizer = torch.optim.Adam(weight_model.parameters(), lr=0.0001, betas=(0.9, 0.98))

model_scheduler = ReduceLROnPlateau(model_optimizer, "max", patience=1, factor=0.5, verbose=True, threshold=1e-4,
                              threshold_mode="abs", min_lr=1e-7)


weigth_checkpointdir = 'unary_and_temporal_prior_checkpoints/weight/'

tr = []
train_data_folder = 'results/' + conf.mode + '_backbone_training_with_dino/'
# unary_train_data_folder = 'results/' + conf.mode + '_unary_only_training/'
# temporal_train_data_folder = 'results/' + conf.mode + '_temporal_only_training/'
filelist = os.listdir(train_data_folder)
print('Loading results from ' + train_data_folder)

valList = []

with open('valFiles.txt', 'r') as fp:
    for line in fp:
        valfile = line[:-1]
        valList.append(valfile)


print("Validation files loaded")
t = TicToc()
def del_keys(pred):
    del pred['features']
    del pred['union_feat']
    del pred['union_box']
    del pred['spatial_masks']
    if conf.mode=='sgcls' or conf.mode=='sgdet':
        del pred['fmaps']
    return pred

def add_frame_keys(pred, frame_names):

    pred['frame_list'] = []
    for i in range(0, len(pred['human_idx'])):
        pred['frame_list'].append(frame_names[i])


    return pred


def train_weight():
    for epoch in range(0, 50):
        random.shuffle(filelist)


        if epoch != 0:
            weight_model_path_epoch = weigth_checkpointdir + '/weight_model_' + str(epoch - 1) + '.pt'
            weight_model_ckpt = torch.load(weight_model_path_epoch, map_location=gpu_device)
            weight_model.load_state_dict(weight_model_ckpt)
            print('Weights models for Attention/Spatial/Contacting loaded from epoch no: ' + str(
                epoch - 1) + 'for weight  training')
        else:
            print('Epoch 0 is running')

        weight_loss_att_iter = []
        weight_loss_spa_iter = []
        weight_loss_con_iter = []
        for b, filename in enumerate(tqdm(filelist[0:1000])):
            if filename in valList:
                # print(filename)
                continue

            results = torch.load(train_data_folder + filename, map_location=torch.device(gpu_device))
            pred_all = results[1]
            # unary_results = torch.load(unary_train_data_folder + filename, map_location=torch.device(gpu_device))
            # unary_pred_all = unary_results[1]
            # temporal_results = torch.load(temporal_train_data_folder + filename, map_location=torch.device(gpu_device))
            # temporal_pred_all = temporal_results[1]
            #
            unary_pred_all = perform_unary_inference(pred_all=pred_all, gpu_device=gpu_device, model=unary_prior_model)
            # results[1] = pred_all
            # torch.save(results, unary_train_data_folder+filename)
            # continue

            temporal_pred_all = perform_temporal_inference(pred_all=pred_all, gpu_device=gpu_device, model=temporal_prior_model)
            # results[1] = pred_all
            # torch.save(results, temporal_train_data_folder+filename)
            # continue
            #pred_all = perform_temporal_inference_with_unary(pred_all=pred_all, gpu_device=gpu_device, model=temporal_prior_model)
            temporal_pred_all = compute_weight(pred_all=temporal_pred_all, gpu_device=gpu_device, model=weight_model, feat_length=1936)
            temporal_pred_all = combine_predictions_with_weights(unary_pred_all, temporal_pred_all, gpu_device)


            # continue
            # print(b)

            weight_losses = compute_combined_loss(pred_all=temporal_pred_all, gpu_device=gpu_device,
                                                    loss_fn=loss_fn)

            model_optimizer.zero_grad()
            weight_loss = sum(weight_losses.values())
            weight_loss.backward()
            model_optimizer.step()
            #print(weight_loss)
            # for p in weight_model.parameters():
            #     print(p.name)
            #     print(p.data)

            weight_loss_att_iter.append(weight_losses["attention_relation_loss"].item())
            weight_loss_spa_iter.append(weight_losses["spatial_relation_loss"].item())
            weight_loss_con_iter.append(weight_losses["contact_relation_loss"].item())
        weight_loss_file_att_iter = weigth_checkpointdir + '/att_weight_prior_loss_' + str(epoch)
        weight_loss_file_spa_iter = weigth_checkpointdir + '/spa_weight_prior_loss_' + str(epoch)
        weight_loss_file_con_iter = weigth_checkpointdir + '/con_weight_prior_loss_' + str(epoch)
        weight_model_path_epoch = weigth_checkpointdir + '/weight_model_' + str(epoch) + '.pt'
        weight_att_file_iter = open(weight_loss_file_att_iter, "wb")
        weight_spa_file_iter = open(weight_loss_file_spa_iter, "wb")
        weight_con_file_iter = open(weight_loss_file_con_iter, "wb")
        np.save(weight_att_file_iter, weight_loss_att_iter)
        np.save(weight_spa_file_iter, weight_loss_spa_iter)
        np.save(weight_con_file_iter, weight_loss_con_iter)
        weight_att_file_iter.close
        weight_spa_file_iter.close
        weight_con_file_iter.close
        torch.save(weight_model.state_dict(), weight_model_path_epoch)


    print('All Weight Training Epochs Are Done')



if __name__ == "__main__":

    train_weight()
