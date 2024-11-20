import torch
import numpy as np
import random
from torch.nn.utils.rnn import pad_sequence

def compute_temporal_logits_for_training(pred_all, gpu_device, model, joint=False):
    if joint == False:
        pred_all['att_logits'] = torch.zeros((pred_all['attention_distribution'].shape[0], 64))
        pred_all['spa_logits'] = torch.zeros((pred_all['spatial_distribution'].shape[0], 64))
        pred_all['con_logits'] = torch.zeros((pred_all['contacting_distribution'].shape[0], 64))




    ####List of every important variables
    pred_rel_feat_list = []
    gt_atts_list = []
    gt_cons_list = []
    gt_spas_list = []
    pred_obj_labels_list = []

    batch_feat = []
    batch_att = []
    batch_spa = []
    batch_con = []
    batch_indicator = []
    batch_att_target_relevant = []
    batch_spa_target_relevant = []
    batch_con_target_relevant = []
    final_frame_ind = int(pred_all['im_idx'][-1] + 1)
    for frame_ind in range(0, final_frame_ind):

        pred_frame_ind = pred_all['im_idx'] == frame_ind
        pred_pairs = pred_all['pair_idx'][pred_frame_ind].cpu().clone().numpy()

        no_rels = pred_pairs.shape[0]

        pred_obj_labels = pred_all['labels'][pred_pairs[:, 1]].cpu().numpy()
        pred_rel_feat = pred_all['rel_feat'][pred_frame_ind]
        gt_atts = []
        gt_spas = []
        gt_cons = []
        for ind in np.where(pred_frame_ind.cpu().numpy())[0]:
            gt_atts.append(pred_all['attention_gt'][ind])
            gt_spas.append(pred_all['spatial_gt'][ind])
            gt_cons.append(pred_all['contacting_gt'][ind])

        pred_rel_feat_list.append(pred_rel_feat)
        gt_atts_list.append(gt_atts)
        gt_cons_list.append(gt_cons)
        gt_spas_list.append(gt_spas)
        pred_obj_labels_list.append(pred_obj_labels)
        current_objs = pred_obj_labels

        for l in range(0, no_rels):
            prev_frame_inds = np.asarray([frame_ind - 1])
            curr_obj = pred_obj_labels[l]
            curr_gt_att = torch.tensor([gt_atts[l][0] + 37])
            curr_gt_spa = torch.tensor([random.choice(gt_spas[l]) + 37 + 3])
            curr_gt_con = torch.tensor([random.choice(gt_cons[l]) + 37 + 3 + 6])
            batch_att_target_relevant.append(curr_gt_att)
            batch_spa_target_relevant.append(curr_gt_spa)
            batch_con_target_relevant.append(curr_gt_con)
            curr_feat = pred_rel_feat[l]
            sequence_feat = curr_feat.flatten().reshape(1, -1)
            sequence_gt_att = curr_gt_att
            sequence_gt_spa = curr_gt_spa
            sequence_gt_con = curr_gt_con
            sequence_indicator = torch.tensor([1])
            for prev_frame_ind in prev_frame_inds:

                if prev_frame_ind > -1: #Cannot search if frame ind is less than 0
                    prev_obj_labels = pred_obj_labels_list[prev_frame_ind]
                    prev_gt_atts = gt_atts_list[prev_frame_ind]
                    prev_gt_spas = gt_spas_list[prev_frame_ind]
                    prev_gt_cons = gt_cons_list[prev_frame_ind]
                    prev_rel_feats = pred_rel_feat_list[prev_frame_ind]
                    prev_obj_ind = np.where(prev_obj_labels == curr_obj)
                    if prev_obj_ind[0].size > 0:
                        prev_gt_att = torch.tensor([prev_gt_atts[prev_obj_ind[0][0]][0]]).reshape(1) + 37
                        prev_gt_spa = torch.tensor([random.choice(prev_gt_spas[prev_obj_ind[0][0]]) + 37 + 3])
                        prev_gt_con = torch.tensor([random.choice(prev_gt_cons[prev_obj_ind[0][0]]) + 37 + 3 + 6])
                        prev_feat = prev_rel_feats[prev_obj_ind[0][0]]
                        sequence_feat = torch.cat((prev_feat.flatten().reshape(1, -1), sequence_feat), 0)
                        sequence_gt_att = torch.cat((prev_gt_att, sequence_gt_att), 0)
                        sequence_gt_spa = torch.cat((prev_gt_spa, sequence_gt_spa), 0)
                        sequence_gt_con = torch.cat((prev_gt_con, sequence_gt_con), 0)
                        sequence_indicator = torch.cat((torch.tensor(0).reshape(1), sequence_indicator), 0)


                    else:
                        # Adding padded values
                        sequence_feat = torch.cat(
                            (sequence_feat, torch.zeros_like(curr_feat.flatten().reshape(1, -1))), 0)
                        sequence_gt_att = torch.cat((sequence_gt_att, torch.tensor(63).reshape(1)), 0)
                        sequence_gt_spa = torch.cat((sequence_gt_spa, torch.tensor(63).reshape(1)), 0)
                        sequence_gt_con = torch.cat((sequence_gt_con, torch.tensor(63).reshape(1)), 0)
                        sequence_indicator = torch.cat((sequence_indicator, torch.tensor(0).reshape(1)), 0)
                else:
                    # Adding padded values
                    sequence_feat = torch.cat((sequence_feat, torch.zeros_like(curr_feat.flatten().reshape(1, -1))),
                                              0)
                    sequence_gt_att = torch.cat((sequence_gt_att, torch.tensor(63).reshape(1)), 0)
                    sequence_gt_spa = torch.cat((sequence_gt_spa, torch.tensor(63).reshape(1)), 0)
                    sequence_gt_con = torch.cat((sequence_gt_con, torch.tensor(63).reshape(1),), 0)
                    sequence_indicator = torch.cat((sequence_indicator, torch.tensor(0).reshape(1)), 0)

            # Adding Object Token as Start token
            sequence_gt_att = torch.cat((torch.tensor(curr_obj).reshape(1), sequence_gt_att), 0)
            sequence_gt_spa = torch.cat((torch.tensor(curr_obj).reshape(1), sequence_gt_spa), 0)
            sequence_gt_con = torch.cat((torch.tensor(curr_obj).reshape(1), sequence_gt_con), 0)

            batch_feat.append(sequence_feat)
            # Adding start token
            batch_att.append(sequence_gt_att)
            batch_spa.append(sequence_gt_spa)
            batch_con.append(sequence_gt_con)
            batch_indicator.append(sequence_indicator)

    if len(batch_feat) > 0:

        batch_feat = torch.stack(batch_feat).to(device=gpu_device)
        batch_att = torch.stack(batch_att).to(device=gpu_device)
        batch_spa = torch.stack(batch_spa).to(device=gpu_device)
        batch_con = torch.stack(batch_con).to(device=gpu_device)
        batch_indicator = torch.stack(batch_indicator).to(device=gpu_device)
        batch_att_target_relevant = torch.stack(batch_att_target_relevant).to(device=gpu_device)
        batch_spa_target_relevant = torch.stack(batch_spa_target_relevant).to(device=gpu_device)
        batch_con_target_relevant = torch.stack(batch_con_target_relevant).to(device=gpu_device)



        #print(batch_att)
        batch_att_in = batch_att.transpose(0, 1)[:-1]
        batch_att_target = batch_att.transpose(0, 1)[1:]

        batch_spa_in = batch_spa.transpose(0, 1)[:-1]
        batch_spa_target = batch_spa.transpose(0, 1)[1:]

        batch_con_in = batch_con.transpose(0, 1)[:-1]
        batch_con_target = batch_con.transpose(0, 1)[1:]

        batch_att_logits, batch_spa_logits, batch_con_logits = model(batch_feat.transpose(0, 1),
                                                                     batch_att_in, batch_spa_in,
                                                                     batch_con_in)
        #print(batch_att_logits.shape)
        batch_att_logits_relevant = []
        batch_spa_logits_relevant = []
        batch_con_logits_relevant = []
        check_batch_att_target_relevant = []
        check_batch_spa_target_relevant = []
        check_batch_con_target_relevant = []
        true_indices = (batch_indicator == 1).nonzero()


        for i in range(true_indices.shape[0]):
            batch_att_logits_relevant.append(batch_att_logits[true_indices[i, 1], i, :])
            batch_spa_logits_relevant.append(batch_spa_logits[true_indices[i, 1], i, :])
            batch_con_logits_relevant.append(batch_con_logits[true_indices[i, 1], i, :])
            check_batch_att_target_relevant.append(batch_att_target[true_indices[i, 1], i])
            check_batch_spa_target_relevant.append(batch_spa_target[true_indices[i, 1], i])
            check_batch_con_target_relevant.append(batch_con_target[true_indices[i, 1], i])


        batch_att_logits_relevant = torch.stack(batch_att_logits_relevant)
        batch_spa_logits_relevant = torch.stack(batch_spa_logits_relevant)
        batch_con_logits_relevant = torch.stack(batch_con_logits_relevant)
        check_batch_att_target_relevant = torch.stack(check_batch_att_target_relevant)
        check_batch_spa_target_relevant = torch.stack(check_batch_spa_target_relevant)
        check_batch_con_target_relevant = torch.stack(check_batch_con_target_relevant)





        # print(pred_all['rel_feat'].shape)
        # print(batch_att_logits.shape)
        # print(batch_spa_logits.shape)
        # print(batch_con_logits.shape)

        if joint == False:
            pred_all['att_logits'] = batch_att_logits_relevant
            pred_all['spa_logits'] = batch_spa_logits_relevant
            pred_all['con_logits'] = batch_con_logits_relevant
        else:
            pred_all['att_logits'] = pred_all['att_logits'] +  batch_att_logits_relevant
            pred_all['spa_logits'] = pred_all['spa_logits'] +  batch_spa_logits_relevant
            pred_all['con_logits'] = pred_all['con_logits'] + batch_con_logits_relevant

        pred_all['att_gt_combined'] = batch_att_target_relevant
        pred_all['spa_gt_combined'] = batch_spa_target_relevant
        pred_all['con_gt_combined'] = batch_con_target_relevant

        return pred_all



def compute_temporal_loss(pred_all, gpu_device, model, loss_fn):
    ####List of every important variables
    pred_rel_feat_list = []
    gt_atts_list = []
    gt_cons_list = []
    gt_spas_list = []
    pred_obj_labels_list = []

    batch_feat = []
    batch_att = []
    batch_spa = []
    batch_con = []
    final_frame_ind = int(pred_all['im_idx'][-1] + 1)
    for frame_ind in range(0, final_frame_ind):

        pred_frame_ind = pred_all['im_idx'] == frame_ind
        pred_pairs = pred_all['pair_idx'][pred_frame_ind].cpu().clone().numpy()

        no_rels = pred_pairs.shape[0]

        pred_obj_labels = pred_all['labels'][pred_pairs[:, 1]].cpu().numpy()
        pred_rel_feat = pred_all['rel_feat'][pred_frame_ind]
        gt_atts = []
        gt_spas = []
        gt_cons = []
        for ind in np.where(pred_frame_ind.cpu().numpy())[0]:
            gt_atts.append(pred_all['attention_gt'][ind])
            gt_spas.append(pred_all['spatial_gt'][ind])
            gt_cons.append(pred_all['contacting_gt'][ind])

        pred_rel_feat_list.append(pred_rel_feat)
        gt_atts_list.append(gt_atts)
        gt_cons_list.append(gt_cons)
        gt_spas_list.append(gt_spas)
        pred_obj_labels_list.append(pred_obj_labels)
        current_objs = pred_obj_labels

        for l in range(0, no_rels):
            prev_frame_inds = np.asarray([frame_ind - 1])
            curr_obj = pred_obj_labels[l]
            curr_gt_att = torch.tensor([gt_atts[l][0] + 37])
            curr_gt_spa = torch.tensor([random.choice(gt_spas[l]) + 37 + 3])
            curr_gt_con = torch.tensor([random.choice(gt_cons[l]) + 37 + 3 + 6])
            curr_feat = pred_rel_feat[l]
            sequence_feat = curr_feat.flatten().reshape(1, -1)
            sequence_gt_att = curr_gt_att
            sequence_gt_spa = curr_gt_spa
            sequence_gt_con = curr_gt_con
            for prev_frame_ind in prev_frame_inds:

                if prev_frame_ind > -1:
                    prev_obj_labels = pred_obj_labels_list[prev_frame_ind]
                    prev_gt_atts = gt_atts_list[prev_frame_ind]
                    prev_gt_spas = gt_spas_list[prev_frame_ind]
                    prev_gt_cons = gt_cons_list[prev_frame_ind]
                    prev_rel_feats = pred_rel_feat_list[prev_frame_ind]
                    prev_obj_ind = np.where(prev_obj_labels == curr_obj)
                    if prev_obj_ind[0].size > 0:
                        prev_gt_att = torch.tensor([prev_gt_atts[prev_obj_ind[0][0]][0]]).reshape(1) + 37
                        prev_gt_spa = torch.tensor([random.choice(prev_gt_spas[prev_obj_ind[0][0]]) + 37 + 3])
                        prev_gt_con = torch.tensor([random.choice(prev_gt_cons[prev_obj_ind[0][0]]) + 37 + 3 + 6])
                        prev_feat = prev_rel_feats[prev_obj_ind[0][0]]
                        sequence_feat = torch.cat((prev_feat.flatten().reshape(1, -1), sequence_feat), 0)
                        sequence_gt_att = torch.cat((prev_gt_att, sequence_gt_att), 0)
                        sequence_gt_spa = torch.cat((prev_gt_spa, sequence_gt_spa), 0)
                        sequence_gt_con = torch.cat((prev_gt_con, sequence_gt_con), 0)


                    else:
                        # Adding padded values
                        sequence_feat = torch.cat(
                            (sequence_feat, torch.zeros_like(curr_feat.flatten().reshape(1, -1))), 0)
                        sequence_gt_att = torch.cat((sequence_gt_att, torch.tensor(63).reshape(1)), 0)
                        sequence_gt_spa = torch.cat((sequence_gt_spa, torch.tensor(63).reshape(1)), 0)
                        sequence_gt_con = torch.cat((sequence_gt_con, torch.tensor(63).reshape(1)), 0)
                else:
                    # Adding padded values
                    sequence_feat = torch.cat((sequence_feat, torch.zeros_like(curr_feat.flatten().reshape(1, -1))),
                                              0)
                    sequence_gt_att = torch.cat((sequence_gt_att, torch.tensor(63).reshape(1)), 0)
                    sequence_gt_spa = torch.cat((sequence_gt_spa, torch.tensor(63).reshape(1)), 0)
                    sequence_gt_con = torch.cat((sequence_gt_con, torch.tensor(63).reshape(1),), 0)

            # Adding Object Token as Start token
            sequence_gt_att = torch.cat((torch.tensor(curr_obj).reshape(1), sequence_gt_att), 0)
            sequence_gt_spa = torch.cat((torch.tensor(curr_obj).reshape(1), sequence_gt_spa), 0)
            sequence_gt_con = torch.cat((torch.tensor(curr_obj).reshape(1), sequence_gt_con), 0)

            batch_feat.append(sequence_feat)
            # Adding start token
            batch_att.append(sequence_gt_att)
            batch_spa.append(sequence_gt_spa)
            batch_con.append(sequence_gt_con)

    if len(batch_feat) > 0:
        batch_feat = torch.stack(batch_feat).to(device=gpu_device)
        batch_att = torch.stack(batch_att).to(device=gpu_device)
        batch_spa = torch.stack(batch_spa).to(device=gpu_device)
        batch_con = torch.stack(batch_con).to(device=gpu_device)

        #print(batch_att)
        batch_att_in = batch_att.transpose(0, 1)[:-1]
        batch_att_target = batch_att.transpose(0, 1)[1:]

        batch_spa_in = batch_spa.transpose(0, 1)[:-1]
        batch_spa_target = batch_spa.transpose(0, 1)[1:]

        batch_con_in = batch_con.transpose(0, 1)[:-1]
        batch_con_target = batch_con.transpose(0, 1)[1:]

        batch_att_logits, batch_spa_logits, batch_con_logits = model(batch_feat.transpose(0, 1),
                                                                     batch_att_in, batch_spa_in,
                                                                     batch_con_in)

        losses = {}
        losses["attention_relation_loss"] = loss_fn(batch_att_logits.reshape(-1, batch_att_logits.shape[-1]),
                                                    batch_att_target.reshape(-1))
        losses["spatial_relation_loss"] = loss_fn(batch_spa_logits.reshape(-1, batch_spa_logits.shape[-1]),
                                                  batch_spa_target.reshape(-1))
        losses["contact_relation_loss"] = loss_fn(batch_con_logits.reshape(-1, batch_con_logits.shape[-1]),
                                                  batch_con_target.reshape(-1))

        return losses

def compute_weight(pred_all, gpu_device, model,feat_length):
    ####List of every important variables
    pred_rel_feat_list = []
    gt_atts_list = []
    gt_cons_list = []
    gt_spas_list = []
    pred_obj_labels_list = []

    batch_feat = []
    batch_att = []
    batch_spa = []
    batch_con = []
    final_frame_ind = int(pred_all['im_idx'][-1] + 1)
    for frame_ind in range(0, final_frame_ind):

        pred_frame_ind = pred_all['im_idx'] == frame_ind
        pred_pairs = pred_all['pair_idx'][pred_frame_ind].cpu().clone().numpy()

        no_rels = pred_pairs.shape[0]

        pred_obj_labels = pred_all['labels'][pred_pairs[:, 1]].cpu().numpy()
        pred_rel_feat = pred_all['rel_feat'][pred_frame_ind]
        gt_atts = []
        gt_spas = []
        gt_cons = []
        for ind in np.where(pred_frame_ind.cpu().numpy())[0]:
            gt_atts.append(pred_all['attention_gt'][ind])
            gt_spas.append(pred_all['spatial_gt'][ind])
            gt_cons.append(pred_all['contacting_gt'][ind])

        pred_rel_feat_list.append(pred_rel_feat)
        gt_atts_list.append(gt_atts)
        gt_cons_list.append(gt_cons)
        gt_spas_list.append(gt_spas)
        pred_obj_labels_list.append(pred_obj_labels)
        current_objs = pred_obj_labels

        for l in range(0, no_rels):
            prev_frame_inds = np.asarray([frame_ind - 1])
            curr_obj = pred_obj_labels[l]
            curr_gt_att = torch.tensor([gt_atts[l][0] + 37])
            curr_gt_spa = torch.tensor([random.choice(gt_spas[l]) + 37 + 3])
            curr_gt_con = torch.tensor([random.choice(gt_cons[l]) + 37 + 3 + 6])
            curr_feat = pred_rel_feat[l]
            sequence_feat = curr_feat.flatten().reshape(1, -1)
            sequence_gt_att = curr_gt_att
            sequence_gt_spa = curr_gt_spa
            sequence_gt_con = curr_gt_con
            for prev_frame_ind in prev_frame_inds:

                if prev_frame_ind > -1:
                    prev_obj_labels = pred_obj_labels_list[prev_frame_ind]
                    prev_gt_atts = gt_atts_list[prev_frame_ind]
                    prev_gt_spas = gt_spas_list[prev_frame_ind]
                    prev_gt_cons = gt_cons_list[prev_frame_ind]
                    prev_rel_feats = pred_rel_feat_list[prev_frame_ind]
                    prev_obj_ind = np.where(prev_obj_labels == curr_obj)
                    if prev_obj_ind[0].size > 0:
                        prev_gt_att = torch.tensor([prev_gt_atts[prev_obj_ind[0][0]][0]]).reshape(1) + 37
                        prev_gt_spa = torch.tensor([random.choice(prev_gt_spas[prev_obj_ind[0][0]]) + 37 + 3])
                        prev_gt_con = torch.tensor([random.choice(prev_gt_cons[prev_obj_ind[0][0]]) + 37 + 3 + 6])
                        prev_feat = prev_rel_feats[prev_obj_ind[0][0]]
                        sequence_feat = torch.cat((prev_feat.flatten().reshape(1, -1), sequence_feat), 0)
                        sequence_gt_att = torch.cat((prev_gt_att, sequence_gt_att), 0)
                        sequence_gt_spa = torch.cat((prev_gt_spa, sequence_gt_spa), 0)
                        sequence_gt_con = torch.cat((prev_gt_con, sequence_gt_con), 0)


                    else:
                        # Adding padded values
                        sequence_feat = torch.cat(
                            (sequence_feat, torch.zeros_like(curr_feat.flatten().reshape(1, -1))), 0)
                        sequence_gt_att = torch.cat((sequence_gt_att, torch.tensor(63).reshape(1)), 0)
                        sequence_gt_spa = torch.cat((sequence_gt_spa, torch.tensor(63).reshape(1)), 0)
                        sequence_gt_con = torch.cat((sequence_gt_con, torch.tensor(63).reshape(1)), 0)
                else:
                    # Adding padded values
                    sequence_feat = torch.cat((sequence_feat, torch.zeros_like(curr_feat.flatten().reshape(1, -1))),
                                              0)
                    sequence_gt_att = torch.cat((sequence_gt_att, torch.tensor(63).reshape(1)), 0)
                    sequence_gt_spa = torch.cat((sequence_gt_spa, torch.tensor(63).reshape(1)), 0)
                    sequence_gt_con = torch.cat((sequence_gt_con, torch.tensor(63).reshape(1),), 0)

            # Adding Object Token as Start token
            sequence_gt_att = torch.cat((torch.tensor(curr_obj).reshape(1), sequence_gt_att), 0)
            sequence_gt_spa = torch.cat((torch.tensor(curr_obj).reshape(1), sequence_gt_spa), 0)
            sequence_gt_con = torch.cat((torch.tensor(curr_obj).reshape(1), sequence_gt_con), 0)

            batch_feat.append(sequence_feat)
            # Adding start token
            batch_att.append(sequence_gt_att)
            batch_spa.append(sequence_gt_spa)
            batch_con.append(sequence_gt_con)

    if len(batch_feat) > 0:
        batch_feat = torch.stack(batch_feat).to(device=gpu_device)
        #print(batch_feat.shape)
        pred_all['CRF_weights'] = model(batch_feat[:, 0, :].squeeze(1), batch_feat[:,1,:].squeeze(1))
        return pred_all

def compute_unary_loss(pred_all, gpu_device, model, loss_fn):
    ####List of every important variables
    pred_rel_feat_list = []
    gt_atts_list = []
    gt_cons_list = []
    gt_spas_list = []
    pred_obj_labels_list = []

    batch_feat = []
    batch_att = []
    batch_spa = []
    batch_con = []
    batch_frame_name = []
    batch_obj = []
    final_frame_ind = int(pred_all['im_idx'][-1] + 1)
    for frame_ind in range(0, final_frame_ind):

        pred_frame_ind = pred_all['im_idx'] == frame_ind
        pred_pairs = pred_all['pair_idx'][pred_frame_ind].cpu().clone().numpy()

        no_rels = pred_pairs.shape[0]

        pred_obj_labels = pred_all['labels'][pred_pairs[:, 1]].cpu().numpy()
        pred_rel_feat = pred_all['rel_feat'][pred_frame_ind]
        gt_atts = []
        gt_spas = []
        gt_cons = []
        for ind in np.where(pred_frame_ind.cpu().numpy())[0]:
            gt_atts.append(pred_all['attention_gt'][ind])
            gt_spas.append(pred_all['spatial_gt'][ind])
            gt_cons.append(pred_all['contacting_gt'][ind])

        pred_rel_feat_list.append(pred_rel_feat)
        gt_atts_list.append(gt_atts)
        gt_cons_list.append(gt_cons)
        gt_spas_list.append(gt_spas)
        pred_obj_labels_list.append(pred_obj_labels)
        current_objs = pred_obj_labels

        for l in range(0, no_rels):

            curr_obj = torch.tensor([pred_obj_labels[l]])
            curr_gt_att = torch.tensor([gt_atts[l][0]]) + 37  # Obj
            curr_gt_spa = torch.tensor([random.choice(gt_spas[l])]) + 37 + 3  # Obj + Att
            curr_gt_con = torch.tensor([random.choice(gt_cons[l])]) + 37 + 3 + 6  # Obj + Att + Spa
            curr_feat = pred_rel_feat[l].flatten().reshape(1, -1)
            if curr_obj.shape[0] > 0:
                batch_feat.append(curr_feat)
                batch_att.append(curr_gt_att)
                batch_spa.append(curr_gt_spa)
                batch_con.append(curr_gt_con)
                batch_obj.append(curr_obj)

    if len(batch_feat) > 0:
        batch_att = torch.stack(batch_att).to(device=gpu_device)
        batch_spa = torch.stack(batch_spa).to(device=gpu_device)
        batch_con = torch.stack(batch_con).to(device=gpu_device)
        batch_obj = torch.stack(batch_obj).to(device=gpu_device)
        batch_feat = torch.stack(batch_feat).to(device=gpu_device)

        batch_att_target = batch_att.transpose(0, 1)
        batch_spa_target = batch_spa.transpose(0, 1)
        batch_con_target = batch_con.transpose(0, 1)
        batch_att_logits, batch_spa_logits, batch_con_logits, _ = model(batch_feat.transpose(0, 1), batch_obj.transpose(0, 1))

        losses = {}

        losses["attention_relation_loss"] = loss_fn(batch_att_logits.reshape(-1, batch_att_logits.shape[-1]),
                                                    batch_att_target.reshape(-1))
        losses["spatial_relation_loss"] = loss_fn(batch_spa_logits.reshape(-1, batch_spa_logits.shape[-1]),
                                                  batch_spa_target.reshape(-1))
        losses["contact_relation_loss"] = loss_fn(batch_con_logits.reshape(-1, batch_con_logits.shape[-1]),
                                                  batch_con_target.reshape(-1))

        return losses





def perform_temporal_inference(pred_all, gpu_device, model):
    pred_all['att_infer'] = torch.zeros_like(pred_all['attention_distribution'])
    pred_all['con_infer'] = torch.zeros_like(pred_all['contacting_distribution'])
    pred_all['spa_infer'] = torch.zeros_like(pred_all['spatial_distribution'])

    ####List of every important variables
    pred_rel_feat_list = []
    pred_obj_labels_list = []
    pred_att_labels_list = []
    pred_spa_labels_list = []
    pred_con_labels_list = []
    final_frame_ind = int(pred_all['im_idx'][-1] + 1)
    for frame_ind in range(0, final_frame_ind):

        pred_frame_ind = pred_all['im_idx'] == frame_ind
        pred_pairs = pred_all['pair_idx'][pred_frame_ind]
        no_rels = pred_pairs.shape[0]

        pred_obj_labels = pred_all['pred_labels'][pred_pairs[:, 1]]
        pred_att_dist = pred_all['attention_distribution'][pred_frame_ind]
        pred_spa_dist = pred_all['spatial_distribution'][pred_frame_ind]
        pred_con_dist = pred_all['contacting_distribution'][pred_frame_ind]
        pred_rel_feat = pred_all['rel_feat'][pred_frame_ind]
        pred_all['att_infer'][pred_frame_ind] = pred_att_dist
        pred_all['con_infer'][pred_frame_ind] = pred_con_dist
        pred_all['spa_infer'][pred_frame_ind] = pred_spa_dist

        # Sub_IoU = IoUBoxList(subj_bbox_in_video[1], subj_bbox_in_video[0])
        att_probs = pred_all['att_infer'][pred_frame_ind]
        spa_probs = pred_all['spa_infer'][pred_frame_ind]
        con_probs = pred_all['con_infer'][pred_frame_ind]
        for l in range(0, no_rels):
            prev_frame_inds = np.asarray([frame_ind - 1])
            pr_ra_m = pred_att_dist[l]
            pr_rc_m = pred_con_dist[l]
            pr_rs_m = pred_spa_dist[l]
            curr_obj = pred_obj_labels[l]
            curr_feat = pred_rel_feat[l]
            sequence_feat = curr_feat.flatten().reshape(1, -1)
            sequence_att_label = torch.tensor(curr_obj).reshape(1).to(device=gpu_device)
            sequence_spa_label = torch.tensor(curr_obj).reshape(1).to(device=gpu_device)
            sequence_con_label = torch.tensor(curr_obj).reshape(1).to(device=gpu_device)

            for prev_frame_ind in prev_frame_inds:
                if prev_frame_ind > -1:
                    prev_obj_labels = pred_obj_labels_list[prev_frame_ind]
                    prev_rel_feats = pred_rel_feat_list[prev_frame_ind]
                    prev_att_labels = pred_att_labels_list[prev_frame_ind]
                    prev_spa_labels = pred_spa_labels_list[prev_frame_ind]
                    prev_con_labels = pred_con_labels_list[prev_frame_ind]
                    prev_obj_ind = np.where(prev_obj_labels.cpu().numpy() == curr_obj.cpu().numpy())
                    # Check if the current object exist in previous frame or not
                    if prev_obj_ind[0].size > 0:
                        prev_att = prev_att_labels[prev_obj_ind[0][0]].reshape(1) + 37
                        prev_spa = prev_spa_labels[prev_obj_ind[0][0]].reshape(1) + 37 + 3
                        prev_con = prev_con_labels[prev_obj_ind[0][0]].reshape(1) + 37 + 3 + 6
                        prev_feat = prev_rel_feats[prev_obj_ind[0][0]]

                        #### Do something here
                        sequence_feat = torch.cat((prev_feat.flatten().reshape(1, -1), sequence_feat), 0)
                        sequence_att_label = torch.cat((sequence_att_label, prev_att), 0)
                        sequence_spa_label = torch.cat((sequence_spa_label, prev_spa), 0)
                        sequence_con_label = torch.cat((sequence_con_label, prev_con), 0)

            seq_length = sequence_feat.shape[0]

            if seq_length > 0:  # Check if there is any memory or not
                input_feat = sequence_feat.reshape(seq_length, 1, -1).to(device=gpu_device)
                input_att_label = sequence_att_label.reshape(-1, 1).to(device=gpu_device)
                input_spa_label = sequence_spa_label.reshape(-1, 1).to(device=gpu_device)
                input_con_label = sequence_con_label.reshape(-1, 1).to(device=gpu_device)

                out_att_logits, out_spa_logits, out_con_logits = model(input_feat, input_att_label,
                                                                       input_spa_label, input_con_label)

                pr_ra_m = torch.softmax(out_att_logits[-1, 0, 37:40], 0)
                pr_rs_m = torch.softmax(out_spa_logits[-1, 0, 40:46], 0)
                pr_rc_m = torch.softmax(out_con_logits[-1, 0, 46:63], 0)

            att_probs[l, :] = pr_ra_m
            spa_probs[l, :] = pr_rs_m
            con_probs[l, :] = pr_rc_m

        pred_all['att_infer'][pred_frame_ind] = att_probs
        pred_all['spa_infer'][pred_frame_ind] = spa_probs
        pred_all['con_infer'][pred_frame_ind] = con_probs
        pred_att_labels = torch.tensor(np.argmax(att_probs.detach().cpu().numpy(), 1)).to(device=gpu_device)
        pred_spa_labels = torch.tensor(np.argmax(spa_probs.detach().cpu().numpy(), 1)).to(device=gpu_device)
        pred_con_labels = torch.tensor(np.argmax(con_probs.detach().cpu().numpy(), 1)).to(device=gpu_device)
        pred_rel_feat_list.append(pred_rel_feat)
        pred_obj_labels_list.append(pred_obj_labels)
        pred_att_labels_list.append(pred_att_labels)
        pred_spa_labels_list.append(pred_spa_labels)
        pred_con_labels_list.append(pred_con_labels)

    return pred_all





def normalize_probs(array):

    return array / array.sum(dim=1, keepdim=True)


def combine_predictions_with_weights(unary_pred_all, temporal_pred_all, gpu_device):
    pr_ra_temp = temporal_pred_all['att_infer']
    pr_ra_unary = unary_pred_all['att_infer']
    #print(pr_ra_temp)
    #print(pr_ra_unary)
    pr_rs_temp = temporal_pred_all['spa_infer']
    pr_rs_unary = unary_pred_all['spa_infer']
    pr_rc_temp =  temporal_pred_all['con_infer']
    pr_rc_unary = unary_pred_all['con_infer']
    w = temporal_pred_all['CRF_weights']
    #print(w)

    #print(w.shape)
    #print(w[1, :].shape)
    #print(w[:,1])
    #print(pr_ra_temp.shape)
    #print(w[:, 0, 0:3].squeeze(1).shape)
    #exit(0)

    pr_ra_comb = torch.mul(torch.pow(pr_ra_temp, w), torch.pow(pr_ra_unary, 1- w))
    pr_rs_comb = torch.mul(torch.pow(pr_rs_temp, w), torch.pow(pr_rs_unary, 1- w))
    pr_rc_comb = torch.mul(torch.pow(pr_rc_temp, w), torch.pow(pr_rc_unary, 1- w))

    # pr_ra_comb = torch.mul(torch.pow(pr_ra_temp, w[:, 0, 0:3].squeeze(1)), torch.pow(pr_ra_unary, w[:, 1, 0:3].squeeze(1)))
    # pr_rs_comb = torch.mul(torch.pow(pr_rs_temp, w[:, 0, 3:9].squeeze(1)), torch.pow(pr_rs_unary, w[:, 1, 3:9].squeeze(1)))
    # pr_rc_comb = torch.mul(torch.pow(pr_rc_temp, w[:, 0, 9:26].squeeze(1)), torch.pow(pr_rc_unary, w[:, 1, 9:26].squeeze(1)))

    pr_ra_comb = normalize_probs(pr_ra_comb)
    pr_rs_comb = normalize_probs(pr_rs_comb)
    pr_rc_comb = normalize_probs(pr_rc_comb)

    temporal_pred_all['att_infer']  = pr_ra_comb
    temporal_pred_all['spa_infer']  = pr_rs_comb
    temporal_pred_all['con_infer'] =  pr_rc_comb

    return temporal_pred_all


def perform_temporal_inference_with_unary(pred_all, gpu_device, model):
    pred_all['att_infer'] = pred_all['att_infer'].detach()
    pred_all['spa_infer'] = pred_all['spa_infer'].detach()
    pred_all['con_infer'] = pred_all['con_infer'].detach()

    ####List of every important variables
    pred_rel_feat_list = []
    pred_obj_labels_list = []
    pred_att_labels_list = []
    pred_spa_labels_list = []
    pred_con_labels_list = []
    final_frame_ind = int(pred_all['im_idx'][-1] + 1)

    for frame_ind in range(0, final_frame_ind):


        pred_frame_ind = pred_all['im_idx'] == frame_ind
        weights = pred_all['CRF_weights'][pred_frame_ind].to(device=gpu_device)
        pred_pairs = pred_all['pair_idx'][pred_frame_ind]
        no_rels = pred_pairs.shape[0]

        pred_obj_labels = pred_all['pred_labels'][pred_pairs[:, 1]]
        pred_att_dist = pred_all['att_infer'][pred_frame_ind].detach()
        pred_spa_dist = pred_all['spa_infer'][pred_frame_ind].detach()
        pred_con_dist = pred_all['con_infer'][pred_frame_ind].detach()
        pred_rel_feat = pred_all['rel_feat'][pred_frame_ind]

        # Sub_IoU = IoUBoxList(subj_bbox_in_video[1], subj_bbox_in_video[0])
        att_probs = pred_all['att_infer'][pred_frame_ind]
        spa_probs = pred_all['spa_infer'][pred_frame_ind]
        con_probs = pred_all['con_infer'][pred_frame_ind]
        for l in range(0, no_rels):
            prev_frame_inds = np.asarray([frame_ind - 1 ])
            pr_ra_m = pred_att_dist[l]
            pr_rs_m = pred_spa_dist[l]
            pr_rc_m = pred_con_dist[l]
            curr_obj = pred_obj_labels[l]
            curr_feat = pred_rel_feat[l]
            w = weights[l]
            #print(w.shape)
            sequence_feat = curr_feat.flatten().reshape(1, -1)
            sequence_att_label = torch.tensor(curr_obj).reshape(1).to(device=gpu_device)
            sequence_spa_label = torch.tensor(curr_obj).reshape(1).to(device=gpu_device)
            sequence_con_label = torch.tensor(curr_obj).reshape(1).to(device=gpu_device)

            for prev_frame_ind in prev_frame_inds:
                if prev_frame_ind > -1:
                    prev_obj_labels = pred_obj_labels_list[prev_frame_ind]
                    prev_rel_feats = pred_rel_feat_list[prev_frame_ind]
                    prev_att_labels = pred_att_labels_list[prev_frame_ind]
                    prev_spa_labels = pred_spa_labels_list[prev_frame_ind]
                    prev_con_labels = pred_con_labels_list[prev_frame_ind]
                    prev_obj_ind = torch.nonzero(prev_obj_labels == curr_obj)
                    #print(prev_obj_ind)
                    # Check if the current object exist in previous frame or not
                    if prev_obj_ind.nelement() !=0 :
                        prev_att = prev_att_labels[prev_obj_ind[0][0]].reshape(1) + 37
                        prev_spa = prev_spa_labels[prev_obj_ind[0][0]].reshape(1) + 37 + 3
                        prev_con = prev_con_labels[prev_obj_ind[0][0]].reshape(1) + 37 + 3 + 6
                        prev_feat = prev_rel_feats[prev_obj_ind[0][0]]

                        #### Do something here
                        sequence_feat = torch.cat((prev_feat.flatten().reshape(1, -1), sequence_feat), 0)
                        sequence_att_label = torch.cat((sequence_att_label, prev_att), 0)
                        sequence_spa_label = torch.cat((sequence_spa_label, prev_spa), 0)
                        sequence_con_label = torch.cat((sequence_con_label, prev_con), 0)

            seq_length = sequence_feat.shape[0]

            if seq_length > 0:  # Check if there is any memory or not
                input_feat = sequence_feat.reshape(seq_length, 1, -1).to(device=gpu_device)
                input_att_label = sequence_att_label.reshape(-1, 1).to(device=gpu_device)
                input_spa_label = sequence_spa_label.reshape(-1, 1).to(device=gpu_device)
                input_con_label = sequence_con_label.reshape(-1, 1).to(device=gpu_device)

                out_att_logits, out_spa_logits, out_con_logits = model(input_feat, input_att_label,
                                                                       input_spa_label, input_con_label)



                pr_ra_temp = torch.softmax(out_att_logits[-1, 0, 37:40], 0)
                pr_rs_temp = torch.softmax(out_spa_logits[-1, 0, 40:46], 0)
                pr_rc_temp = torch.softmax(out_con_logits[-1, 0, 46:63], 0)

                pr_ra_m = torch.mul(torch.pow(pr_ra_temp, w), torch.pow(pr_ra_m, 1- w))
                pr_ra_m = pr_ra_m / torch.sum(pr_ra_m, 0)
                pr_rs_m = torch.mul(torch.pow(pr_rs_temp, w), torch.pow(pr_rs_m, 1 - w))
                pr_rs_m = pr_rs_m / torch.sum(pr_rs_m, 0)
                pr_rc_m = torch.mul(torch.pow(pr_rc_temp, w), torch.pow(pr_rc_m, 1 - w))
                pr_rc_m = pr_rc_m / torch.sum(pr_rc_m, 0)

                # pr_ra_m = torch.mul(torch.pow(pr_ra_temp, w[0, 0:3].flatten()), torch.pow(pr_ra_m, w[1, 0:3].flatten()))
                # pr_ra_m = pr_ra_m / torch.sum(pr_ra_m, 0)
                # pr_rs_m = torch.mul(torch.pow(pr_rs_temp,w[0, 3:9]), torch.pow(pr_rs_m, w[1, 3:9]))
                # pr_rs_m = pr_rs_m / torch.sum(pr_rs_m, 0)
                # pr_rc_m = torch.mul(torch.pow(pr_rc_temp, w[0, 9:26]), torch.pow(pr_rc_m, w[1, 9:26]))
                # pr_rc_m = pr_rc_m / torch.sum(pr_rc_m, 0)



            att_probs[l, :] = pr_ra_m
            spa_probs[l, :] = pr_rs_m
            con_probs[l, :] = pr_rc_m

        pred_all['att_infer'][pred_frame_ind] = att_probs
        pred_all['spa_infer'][pred_frame_ind] = spa_probs
        pred_all['con_infer'][pred_frame_ind] = con_probs
        pred_att_labels = torch.tensor(np.argmax(att_probs.detach().cpu().numpy(), 1)).to(device=gpu_device)
        pred_spa_labels = torch.tensor(np.argmax(spa_probs.detach().cpu().numpy(), 1)).to(device=gpu_device)
        pred_con_labels = torch.tensor(np.argmax(con_probs.detach().cpu().numpy(), 1)).to(device=gpu_device)
        pred_rel_feat_list.append(pred_rel_feat)
        pred_obj_labels_list.append(pred_obj_labels)
        pred_att_labels_list.append(pred_att_labels)
        pred_spa_labels_list.append(pred_spa_labels)
        pred_con_labels_list.append(pred_con_labels)

    return pred_all


def perform_unary_inference(pred_all, gpu_device, model):

    batch_feat = pred_all['rel_feat'].unsqueeze(0).to(device=gpu_device)
    batch_obj = pred_all['pred_labels'][pred_all['pair_idx'][:, 1]].unsqueeze(0).to(device=gpu_device)
    out_att_logits, out_spa_logits, out_con_logits, _ = model(batch_feat, batch_obj)
    pred_all['att_logits'] = out_att_logits.squeeze(0)
    pred_all['spa_logits'] = out_spa_logits.squeeze(0)
    pred_all['con_logits'] = out_con_logits.squeeze(0)
    pred_all['att_infer'] = torch.softmax(out_att_logits[0, :, 37:40],1)
    pred_all['spa_infer'] = torch.softmax(out_spa_logits[0, :, 40:46],1)
    pred_all['con_infer'] = torch.softmax(out_con_logits[0, :, 46:63],1)

    return pred_all


def compute_combined_loss_old(pred_all, gpu_device, loss_fn):
    batch_att_logits = pred_all['att_logits']
    batch_spa_logits = pred_all['spa_logits']
    batch_con_logits = pred_all['con_logits']

    attention_label = pred_all['att_gt_combined']
    spatial_label = pred_all['spa_gt_combined']
    contact_label = pred_all['con_gt_combined']

    # print(batch_att_logits.shape)
    # print(attention_label.shape)

    losses = {}

    losses["attention_relation_loss"] = loss_fn(batch_att_logits, attention_label.flatten())
    losses["spatial_relation_loss"] = loss_fn(batch_spa_logits, spatial_label.flatten())
    losses["contact_relation_loss"] = loss_fn(batch_con_logits, contact_label.flatten())
    #print(losses)
    #exit()
    return losses


def print_predicted_and_gt_triplet(pred_all, attention_classes, spa_classes, con_classes, obj_classes, infer_status):

    if infer_status:
        pred_all['attention_distribution'] = pred_all['att_infer'].detach()
        pred_all['spatial_distribution'] = pred_all['spa_infer'].detach()
        pred_all['contacting_distribution'] = pred_all['con_infer'].detach()
    final_frame_ind = int(pred_all['im_idx'][-1] + 1)

    for frame_ind in range(0, final_frame_ind):
        pred_frame_ind = pred_all['im_idx'] == frame_ind
        pred_pairs = pred_all['pair_idx'][pred_frame_ind].cpu().clone().numpy()
        print(pred_all['frame_list'][frame_ind])

        no_rels = pred_pairs.shape[0]

        pred_obj_labels = pred_all['labels'][pred_pairs[:, 1]].cpu().numpy()

        pred_att_dist = pred_all['attention_distribution'][pred_frame_ind]
        pred_spa_dist = pred_all['spatial_distribution'][pred_frame_ind]
        pred_con_dist = pred_all['contacting_distribution'][pred_frame_ind]
        gt_atts = []
        gt_spas = []
        gt_cons = []
        for ind in np.where(pred_frame_ind.cpu().numpy())[0]:
            gt_atts.append(pred_all['attention_gt'][ind])
            gt_spas.append(pred_all['spatial_gt'][ind])
            gt_cons.append(pred_all['contacting_gt'][ind])



        for l in range(0, no_rels):

            curr_obj = obj_classes[pred_obj_labels[l]]
            curr_gt_att = [attention_classes[x] for x in gt_atts[l]]
            curr_gt_spa = [spa_classes[x] for x in gt_spas[l]]
            curr_gt_con = [con_classes[x] for x in gt_cons[l]]

            pred_att = attention_classes[np.argmax(pred_att_dist[l])]
            pred_spa = spa_classes[np.argmax(pred_spa_dist[l])]
            pred_con = con_classes[np.argmax(pred_con_dist[l])]


            print(curr_obj)
            print('Ground Truth Triplet')
            print(curr_gt_att)
            print(curr_gt_spa)
            print(curr_gt_con)

            print('Predicted Triplet')
            print(pred_att)
            print(pred_spa)
            print(pred_con)


def compute_combined_loss(pred_all, gpu_device, loss_fn):
    ####List of every important variables
    pred_rel_feat_list = []
    gt_atts_list = []
    gt_cons_list = []
    gt_spas_list = []
    pred_obj_labels_list = []
    batch_feat = []
    batch_att = []
    batch_spa = []
    batch_con = []
    batch_frame_name = []
    batch_obj = []
    final_frame_ind = int(pred_all['im_idx'][-1] + 1)
    for frame_ind in range(0, final_frame_ind):

        pred_frame_ind = pred_all['im_idx'] == frame_ind
        pred_pairs = pred_all['pair_idx'][pred_frame_ind].cpu().clone().numpy()

        no_rels = pred_pairs.shape[0]
        gt_atts = []
        gt_spas = []
        gt_cons = []
        for ind in np.where(pred_frame_ind.cpu().numpy())[0]:
            gt_atts.append(pred_all['attention_gt'][ind])
            gt_spas.append(pred_all['spatial_gt'][ind])
            gt_cons.append(pred_all['contacting_gt'][ind])

        gt_atts_list.append(gt_atts)
        gt_cons_list.append(gt_cons)
        gt_spas_list.append(gt_spas)

        for l in range(0, no_rels):
            curr_gt_att = torch.tensor([gt_atts[l][0]]) # Obj
            curr_gt_spa = torch.tensor([random.choice(gt_spas[l])])   # Obj + Att
            curr_gt_con = torch.tensor([random.choice(gt_cons[l])])  # Obj + Att + Spa
            batch_att.append(curr_gt_att)
            batch_spa.append(curr_gt_spa)
            batch_con.append(curr_gt_con)

    batch_att_target = torch.stack(batch_att).to(device=gpu_device)
    batch_spa_target = torch.stack(batch_spa).to(device=gpu_device)
    batch_con_target = torch.stack(batch_con).to(device=gpu_device)

    batch_att_probs = pred_all['att_infer'].to(device=gpu_device)
    batch_spa_probs = pred_all['spa_infer'].to(device=gpu_device)
    batch_con_probs = pred_all['con_infer'].to(device=gpu_device)

    losses = {}

    losses["attention_relation_loss"] = loss_fn(torch.log(batch_att_probs),
                                                batch_att_target.flatten())
    losses["spatial_relation_loss"] = loss_fn(torch.log(batch_spa_probs),
                                              batch_spa_target.flatten())
    losses["contact_relation_loss"] = loss_fn(torch.log(batch_con_probs),
                                              batch_con_target.flatten())

    # losses["attention_relation_loss"] = loss_fn(batch_att_probs,
    #                                             batch_att_target.flatten())
    # losses["spatial_relation_loss"] = loss_fn(batch_spa_probs,
    #                                           batch_spa_target.flatten())
    # losses["contact_relation_loss"] = loss_fn(batch_con_probs,
    #                                           batch_con_target.flatten())

    return losses

