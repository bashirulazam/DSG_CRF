import pickle
import torch
import numpy as np
import os
import scipy.io as sio
import random
from tqdm import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau
from dataloader.action_genome import AG, cuda_collate_fn
from lib.config import Config
from lib.Temporal_Model import MyTempTransformer
#from lib.Temporal_Model_Encoder_Only import MyTempTransformer
#from lib.Temporal_Model_Decoder_Only import MyTempTransformer
from lib.Unary_Model_Combined import MyUnaryTransformer
#from lib.Unary_Model_Combined_Encoder_Only import MyUnaryTransformer
#from lib.Unary_Model_Combined_Decoder_Only import MyUnaryTransformer
from lib.evaluation_recall_decomposed import BasicSceneGraphEvaluator
from lib.forward_pass_utils import perform_unary_inference, perform_temporal_inference
#from lib.forward_pass_utils_v5_only_encoder import perform_unary_inference, perform_temporal_inference
#from lib.forward_pass_utils_v5_only_encoder_mlm_loss import perform_unary_inference, perform_temporal_inference
#from lib.forward_pass_utils_v5_only_decoder import perform_unary_inference, perform_temporal_inference
#from lib.forward_pass_utils_v5_only_decoder_no_obj import perform_unary_inference, perform_temporal_inference
#from lib.forward_pass_utils_v5_no_obj import perform_unary_inference, perform_temporal_inference

conf = Config()
for i in conf.args:
    print(i,':', conf.args[i])

AG_dataset = AG(mode="test", datasize=conf.datasize, data_path=conf.data_path, filter_nonperson_box_frame=True,
                filter_small_box=False if conf.mode == 'predcls' else True)

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




checkpointdir = 'unary_and_temporal_prior_checkpoints/'
temporal_checkpointdir = checkpointdir + '/temporal/' 
unary_checkpointdir = checkpointdir + '/unary/'
# some parameters
tr = []
train_data_folder = 'results/' + conf.mode + '_backbone_training_with_dino/'
unary_val_folder = 'results/' + conf.mode + '_unary_val/'
temporal_val_folder = 'results/' + conf.mode + '_temporal_val/'
filelist = os.listdir(train_data_folder)

unary_prior_model.eval()
temporal_prior_model.eval()
valList = []

with open('valFiles.txt', 'r') as fp:
    for line in fp:
        valfile = line[:-1]
        valList.append(valfile)


print("Validation files loaded")



def val_unary():
    for epoch in range(0, 50):
        evaluator_unary_cons_att = BasicSceneGraphEvaluator(
            mode=conf.mode,
            AG_object_classes=AG_dataset.object_classes,
            AG_all_predicates=AG_dataset.relationship_classes,
            AG_attention_predicates=AG_dataset.attention_relationships,
            AG_spatial_predicates=AG_dataset.spatial_relationships,
            AG_contacting_predicates=AG_dataset.contacting_relationships,
            iou_threshold=0.5,
            constraint='with', rel_type='att', topkList = {3: [], 5: [],  10: []})


        evaluator_unary_cons_all = BasicSceneGraphEvaluator(
            mode=conf.mode,
            AG_object_classes=AG_dataset.object_classes,
            AG_all_predicates=AG_dataset.relationship_classes,
            AG_attention_predicates=AG_dataset.attention_relationships,
            AG_spatial_predicates=AG_dataset.spatial_relationships,
            AG_contacting_predicates=AG_dataset.contacting_relationships,
            iou_threshold=0.5,
            constraint='with', rel_type='all', topkList={10: [], 20: [], 50: [], 100: []})

        evaluator_unary_cons_spa = BasicSceneGraphEvaluator(
            mode=conf.mode,
            AG_object_classes=AG_dataset.object_classes,
            AG_all_predicates=AG_dataset.relationship_classes,
            AG_attention_predicates=AG_dataset.attention_relationships,
            AG_spatial_predicates=AG_dataset.spatial_relationships,
            AG_contacting_predicates=AG_dataset.contacting_relationships,
            iou_threshold=0.5,
            constraint='with', rel_type='spa', topkList={3: [], 5: [], 10: []})

        evaluator_unary_cons_con = BasicSceneGraphEvaluator(
            mode=conf.mode,
            AG_object_classes=AG_dataset.object_classes,
            AG_all_predicates=AG_dataset.relationship_classes,
            AG_attention_predicates=AG_dataset.attention_relationships,
            AG_spatial_predicates=AG_dataset.spatial_relationships,
            AG_contacting_predicates=AG_dataset.contacting_relationships,
            iou_threshold=0.5,
            constraint='with', rel_type='con', topkList={3: [], 5: [], 10: []})

        evaluator_unary_no_cons_spa = BasicSceneGraphEvaluator(
            mode=conf.mode,
            AG_object_classes=AG_dataset.object_classes,
            AG_all_predicates=AG_dataset.relationship_classes,
            AG_attention_predicates=AG_dataset.attention_relationships,
            AG_spatial_predicates=AG_dataset.spatial_relationships,
            AG_contacting_predicates=AG_dataset.contacting_relationships,
            iou_threshold=0.5,
            constraint='no', rel_type='spa', topkList={3: [], 5: [], 10: []})

        evaluator_unary_no_cons_con = BasicSceneGraphEvaluator(
            mode=conf.mode,
            AG_object_classes=AG_dataset.object_classes,
            AG_all_predicates=AG_dataset.relationship_classes,
            AG_attention_predicates=AG_dataset.attention_relationships,
            AG_spatial_predicates=AG_dataset.spatial_relationships,
            AG_contacting_predicates=AG_dataset.contacting_relationships,
            iou_threshold=0.5,
            constraint='no', rel_type='con', topkList={3: [], 5: [], 10: []})


        unary_model_path_epoch = unary_checkpointdir + '/unary_prior_model_' + str(epoch) + '.pt'
        unary_model_ckpt = torch.load(unary_model_path_epoch, map_location=gpu_device)
        unary_prior_model.load_state_dict(unary_model_ckpt)
        print('Unary models loaded from epoch no: ' + str(epoch) + 'for unary  training')


        for b, filename in enumerate(tqdm(filelist)):
            if filename not in valList:
                #print(filename)
                continue

            results = torch.load(train_data_folder + filename, map_location=torch.device('cpu'))
            gt_annotation = results[0]
            results = torch.load(train_data_folder + filename, map_location=torch.device(gpu_device))
            pred_all = results[1]
            pred_all = perform_unary_inference(pred_all=pred_all, gpu_device=gpu_device, model=unary_prior_model)
            results = [results[0], pred_all]
            torch.save(results, unary_val_folder + filename)
            evaluator_unary_cons_att.evaluate_scene_graph(gt_annotation, dict(pred_all), infer_status=1)
            evaluator_unary_cons_spa.evaluate_scene_graph(gt_annotation, dict(pred_all), infer_status=1)
            evaluator_unary_cons_con.evaluate_scene_graph(gt_annotation, dict(pred_all), infer_status=1)
            evaluator_unary_no_cons_spa.evaluate_scene_graph(gt_annotation, dict(pred_all), infer_status=1)
            evaluator_unary_no_cons_con.evaluate_scene_graph(gt_annotation, dict(pred_all), infer_status=1)
            evaluator_unary_cons_all.evaluate_scene_graph(gt_annotation, dict(pred_all), infer_status=1)


        print("Unary Inference Done for Attention , Spatial and Contacting")
        print("Results for provided unary checkpoint")
        print('-------------------------with constraint-------------------------------')
        print('..................Attention Type.........................................')
        cons_att_per_rel = evaluator_unary_cons_att.print_stats()
        print('..................Spatial Type.........................................')
        cons_spa_per_rel = evaluator_unary_cons_spa.print_stats()
        print('..................Contacting Type.........................................')
        cons_con_per_rel = evaluator_unary_cons_con.print_stats()
        print('..................All Type.......................................')
        cons_all_per_rel = evaluator_unary_cons_all.print_stats()
        print('done')

        print('-------------------------no constraint-------------------------------')
        print('..................Spatial Type.........................................')
        no_cons_spa_per_rel = evaluator_unary_no_cons_spa.print_stats()
        print('..................Contacting Type.........................................')
        no_cons_con_per_rel = evaluator_unary_no_cons_con.print_stats()
        print("Unary Inference Done for Spatial/Contacting")

    print('All Epochs Done for Unary')

def val_temporal():
    for epoch in range(0, 50):

        evaluator_temporal_cons_all = BasicSceneGraphEvaluator(
            mode=conf.mode,
            AG_object_classes=AG_dataset.object_classes,
            AG_all_predicates=AG_dataset.relationship_classes,
            AG_attention_predicates=AG_dataset.attention_relationships,
            AG_spatial_predicates=AG_dataset.spatial_relationships,
            AG_contacting_predicates=AG_dataset.contacting_relationships,
            iou_threshold=0.5,
            constraint='with', rel_type='all', topkList={10: [], 20: [], 50: [], 100: []})

        evaluator_temporal_cons_att = BasicSceneGraphEvaluator(
            mode=conf.mode,
            AG_object_classes=AG_dataset.object_classes,
            AG_all_predicates=AG_dataset.relationship_classes,
            AG_attention_predicates=AG_dataset.attention_relationships,
            AG_spatial_predicates=AG_dataset.spatial_relationships,
            AG_contacting_predicates=AG_dataset.contacting_relationships,
            iou_threshold=0.5,
            constraint='with', rel_type='att', topkList={3: [], 5: [], 10: []})

        evaluator_temporal_cons_spa = BasicSceneGraphEvaluator(
            mode=conf.mode,
            AG_object_classes=AG_dataset.object_classes,
            AG_all_predicates=AG_dataset.relationship_classes,
            AG_attention_predicates=AG_dataset.attention_relationships,
            AG_spatial_predicates=AG_dataset.spatial_relationships,
            AG_contacting_predicates=AG_dataset.contacting_relationships,
            iou_threshold=0.5,
            constraint='with', rel_type='spa', topkList={3: [], 5: [], 10: []})

        evaluator_temporal_cons_con = BasicSceneGraphEvaluator(
            mode=conf.mode,
            AG_object_classes=AG_dataset.object_classes,
            AG_all_predicates=AG_dataset.relationship_classes,
            AG_attention_predicates=AG_dataset.attention_relationships,
            AG_spatial_predicates=AG_dataset.spatial_relationships,
            AG_contacting_predicates=AG_dataset.contacting_relationships,
            iou_threshold=0.5,
            constraint='with', rel_type='con', topkList={3: [], 5: [], 10: []})

        evaluator_temporal_no_cons_spa = BasicSceneGraphEvaluator(
            mode=conf.mode,
            AG_object_classes=AG_dataset.object_classes,
            AG_all_predicates=AG_dataset.relationship_classes,
            AG_attention_predicates=AG_dataset.attention_relationships,
            AG_spatial_predicates=AG_dataset.spatial_relationships,
            AG_contacting_predicates=AG_dataset.contacting_relationships,
            iou_threshold=0.5,
            constraint='no', rel_type='spa', topkList={3: [], 5: [], 10: []})

        evaluator_temporal_no_cons_con = BasicSceneGraphEvaluator(
            mode=conf.mode,
            AG_object_classes=AG_dataset.object_classes,
            AG_all_predicates=AG_dataset.relationship_classes,
            AG_attention_predicates=AG_dataset.attention_relationships,
            AG_spatial_predicates=AG_dataset.spatial_relationships,
            AG_contacting_predicates=AG_dataset.contacting_relationships,
            iou_threshold=0.5,
            constraint='no', rel_type='con', topkList={3: [], 5: [], 10: []})

        temporal_model_path_epoch = temporal_checkpointdir + '/temp_prior_model_' + str(epoch) + '.pt'
        temporal_model_ckpt = torch.load(temporal_model_path_epoch, map_location=gpu_device)
        temporal_prior_model.load_state_dict(temporal_model_ckpt)
        print('Temporal Models loaded from epoch no: ' + str(epoch)) 

        for b, filename in enumerate(tqdm(filelist)):
            if filename not in valList:
                # print(filename)
                continue

            results = torch.load(train_data_folder + filename, map_location=torch.device('cpu'))
            gt_annotation = results[0]

            results = torch.load(train_data_folder + filename, map_location=torch.device(gpu_device))
            pred_all = results[1]
            pred_all = perform_temporal_inference(pred_all=pred_all, gpu_device=gpu_device,
                                                  model=temporal_prior_model)
            results = [results[0], pred_all]
            torch.save(results, temporal_val_folder + filename)
            evaluator_temporal_cons_att.evaluate_scene_graph(gt_annotation, dict(pred_all), infer_status=1)
            evaluator_temporal_cons_spa.evaluate_scene_graph(gt_annotation, dict(pred_all), infer_status=1)
            evaluator_temporal_cons_con.evaluate_scene_graph(gt_annotation, dict(pred_all), infer_status=1)
            evaluator_temporal_no_cons_spa.evaluate_scene_graph(gt_annotation, dict(pred_all), infer_status=1)
            evaluator_temporal_no_cons_con.evaluate_scene_graph(gt_annotation, dict(pred_all), infer_status=1)
            evaluator_temporal_cons_all.evaluate_scene_graph(gt_annotation, dict(pred_all), infer_status=1)



        print("Temporal Inference Done")
        print("Results for provided temporal checkpoint")
        print('-------------------------with constraint-------------------------------')
        print('..................Attention Type.......................................')
        cons_att_per_rel = evaluator_temporal_cons_att.print_stats()
        print('..................Spatial Type.........................................')
        cons_spa_per_rel = evaluator_temporal_cons_spa.print_stats()
        print('..................Contacting Type.........................................')
        cons_con_per_rel = evaluator_temporal_cons_con.print_stats()
        print('..................All Type.......................................')
        cons_all_per_rel = evaluator_temporal_cons_all.print_stats()
        print('done')

        print('-------------------------no constraint-------------------------------')
        print('..................Spatial Type.........................................')
        no_cons_spa_per_rel = evaluator_temporal_no_cons_spa.print_stats()
        print('..................Contacting Type.........................................')
        no_cons_con_per_rel = evaluator_temporal_no_cons_con.print_stats()
        print('done')

    print('All Epochs Done for temporal')
if __name__ == "__main__":
    val_unary()
    val_temporal()
