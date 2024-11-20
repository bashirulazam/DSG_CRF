import numpy as np
np.set_printoptions(precision=4)
import copy
import torch
import pickle

from dataloader.action_genome import AG, cuda_collate_fn

from lib.config import Config
from lib.evaluation_recall_decomposed import BasicSceneGraphEvaluator
from lib.object_detector import detector
from lib.backbone import STTran
from lib.Temporal_Model import MyTempTransformer
#from lib.Temporal_Model_Encoder_Only import MyTempTransformer
#from lib.Temporal_Model_Decoder_Only import MyTempTransformer
from lib.Unary_Model_Combined import MyUnaryTransformer
#from lib.Unary_Model_Combined_Encoder_Only import MyUnaryTransformer
#from lib.Unary_Model_Combined_Decoder_Only import MyUnaryTransformer
#from lib.Unary_Model_Att_Decoder_Only import MyUnaryAttTransformer
#from lib.forward_pass_utils_v5_weight_comb import perform_unary_inference,  perform_temporal_inference, perform_temporal_inference_with_unary
from lib.forward_pass_utils import perform_unary_inference,  perform_temporal_inference, perform_temporal_inference_with_unary, compute_weight
#from lib.forward_pass_utils_v5 import perform_unary_inference,  perform_temporal_inference, perform_temporal_inference_with_unary
#from lib.forward_pass_utils_v5_no_obj import perform_unary_inference,  perform_temporal_inference, perform_temporal_inference_with_unary
#from lib.forward_pass_utils_v5_only_encoder import perform_unary_inference, perform_temporal_inference, perform_temporal_inference_with_unary
#from lib.forward_pass_utils_v5_only_encoder_mlm_loss import perform_unary_inference, perform_temporal_inference, perform_temporal_inference_with_unary
#from lib.forward_pass_utils_v5_only_decoder import perform_unary_inference, perform_temporal_inference, perform_temporal_inference_with_unary
#from lib.forward_pass_utils_v5_only_decoder_no_obj import perform_unary_inference,  perform_temporal_inference, perform_temporal_inference_with_unary
from lib.weight_model import OurWeight

conf = Config()
for i in conf.args:
    print(i,':', conf.args[i])

AG_dataset = AG(mode="test", datasize=conf.datasize, data_path=conf.data_path, filter_nonperson_box_frame=True,
                filter_small_box=False if conf.mode == 'predcls' else True)
dataloader = torch.utils.data.DataLoader(AG_dataset, shuffle=False, num_workers=0, collate_fn=cuda_collate_fn)

gpu_device = torch.device('cuda:0')
object_detector = detector(train=False, object_classes=AG_dataset.object_classes, use_SUPPLY=True, mode=conf.mode).to(device=gpu_device)
object_detector.eval()


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


dino_feat_length = 3472
weight_model = OurWeight(feat_length=dino_feat_length).to(device=gpu_device)
unary_prior_model.eval()
temporal_prior_model.eval()
weight_model.eval()


unary_ckpt = torch.load(conf.unary_model_path, map_location=gpu_device)
unary_prior_model.load_state_dict(unary_ckpt, strict=False)
print('*'*50)
print('CKPT {} is loaded'.format(conf.unary_model_path))



temporal_ckpt = torch.load(conf.temporal_model_path, map_location=gpu_device)
temporal_prior_model.load_state_dict(temporal_ckpt, strict=False)
print('*'*50)
print('CKPT {} is loaded'.format(conf.temporal_model_path))

weight_ckpt = torch.load(conf.weight_model_path, map_location=gpu_device)
weight_model.load_state_dict(weight_ckpt, strict=False)

evaluator_cons_all = BasicSceneGraphEvaluator(
        mode=conf.mode,
        AG_object_classes=AG_dataset.object_classes,
        AG_all_predicates=AG_dataset.relationship_classes,
        AG_attention_predicates=AG_dataset.attention_relationships,
        AG_spatial_predicates=AG_dataset.spatial_relationships,
        AG_contacting_predicates=AG_dataset.contacting_relationships,
        iou_threshold=0.5,
        constraint='with', rel_type='all', topkList = {10: [], 20: [], 50: [], 100: []})
evaluator_no_cons_all = BasicSceneGraphEvaluator(
        mode=conf.mode,
        AG_object_classes=AG_dataset.object_classes,
        AG_all_predicates=AG_dataset.relationship_classes,
        AG_attention_predicates=AG_dataset.attention_relationships,
        AG_spatial_predicates=AG_dataset.spatial_relationships,
        AG_contacting_predicates=AG_dataset.contacting_relationships,
        iou_threshold=0.5,
        constraint='no', rel_type='all',topkList = {10: [], 20: [], 50: [], 100: []})

evaluator_cons_att = BasicSceneGraphEvaluator(
        mode=conf.mode,
        AG_object_classes=AG_dataset.object_classes,
        AG_all_predicates=AG_dataset.relationship_classes,
        AG_attention_predicates=AG_dataset.attention_relationships,
        AG_spatial_predicates=AG_dataset.spatial_relationships,
        AG_contacting_predicates=AG_dataset.contacting_relationships,
        iou_threshold=0.5,
        constraint='with', rel_type='att', topkList = {3: [], 5: [],  10: []})

evaluator_cons_spa = BasicSceneGraphEvaluator(
        mode=conf.mode,
        AG_object_classes=AG_dataset.object_classes,
        AG_all_predicates=AG_dataset.relationship_classes,
        AG_attention_predicates=AG_dataset.attention_relationships,
        AG_spatial_predicates=AG_dataset.spatial_relationships,
        AG_contacting_predicates=AG_dataset.contacting_relationships,
        iou_threshold=0.5,
        constraint='with', rel_type='spa', topkList = {3:[],  5: [], 10: []})

evaluator_cons_con = BasicSceneGraphEvaluator(
        mode=conf.mode,
        AG_object_classes=AG_dataset.object_classes,
        AG_all_predicates=AG_dataset.relationship_classes,
        AG_attention_predicates=AG_dataset.attention_relationships,
        AG_spatial_predicates=AG_dataset.spatial_relationships,
        AG_contacting_predicates=AG_dataset.contacting_relationships,
        iou_threshold=0.5,
        constraint='with', rel_type='con', topkList = {3: [],  5: [], 10: []})

evaluator_no_cons_spa = BasicSceneGraphEvaluator(
        mode=conf.mode,
        AG_object_classes=AG_dataset.object_classes,
        AG_all_predicates=AG_dataset.relationship_classes,
        AG_attention_predicates=AG_dataset.attention_relationships,
        AG_spatial_predicates=AG_dataset.spatial_relationships,
        AG_contacting_predicates=AG_dataset.contacting_relationships,
        iou_threshold=0.5,
        constraint='no', rel_type='spa', topkList = {3: [],  5: [],  10: []})

evaluator_no_cons_con = BasicSceneGraphEvaluator(
        mode=conf.mode,
        AG_object_classes=AG_dataset.object_classes,
        AG_all_predicates=AG_dataset.relationship_classes,
        AG_attention_predicates=AG_dataset.attention_relationships,
        AG_spatial_predicates=AG_dataset.spatial_relationships,
        AG_contacting_predicates=AG_dataset.contacting_relationships,
        iou_threshold=0.5,
        constraint='no', rel_type='con', topkList = {3: [], 5: [], 10: []})

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
    for i in range(0,len(pred['human_idx'])):
        pred['frame_list'].append(frame_names[i])


    return pred

infer_status = 1
with torch.no_grad():
    for b, data in enumerate(dataloader):

        im_data = copy.deepcopy(data[0].cuda(0))
        im_info = copy.deepcopy(data[1].cuda(0))
        gt_boxes = copy.deepcopy(data[2].cuda(0))
        num_boxes = copy.deepcopy(data[3].cuda(0))
        gt_annotation = AG_dataset.gt_annotations[data[4]]
        frame_names = data[5]
        #print('I am priting frame names')
        #print(len(frame_names))
        #print(frame_names)
        vid_name = frame_names[0].split('/')[0].split('.')[0]
        print(vid_name)
        results = torch.load(conf.backbone_result_folder + vid_name + '.pt', map_location=torch.device(gpu_device))
        pred  = results[1]
        pred = compute_weight(pred_all=pred, gpu_device=gpu_device, model=weight_model, feat_length=1936)
        pred = perform_unary_inference(pred_all=pred, gpu_device=gpu_device, model=unary_prior_model)
        #pred = perform_temporal_inference(pred_all=pred, gpu_device=gpu_device, model=temporal_prior_model)
        pred = perform_temporal_inference_with_unary(pred_all=pred, gpu_device=gpu_device, model=temporal_prior_model)

        #pred = del_keys(pred)
        #print(pred.keys())
        pred = add_frame_keys(pred, frame_names)
        result = [gt_annotation, pred]
        #torch.save(result, 'results/' + conf.mode + '_combined/'+vid_name+'.pt')

        evaluator_cons_att.evaluate_scene_graph(gt_annotation, dict(pred), infer_status=infer_status)
        evaluator_cons_spa.evaluate_scene_graph(gt_annotation, dict(pred), infer_status=infer_status)
        evaluator_cons_con.evaluate_scene_graph(gt_annotation, dict(pred), infer_status=infer_status)
        evaluator_no_cons_spa.evaluate_scene_graph(gt_annotation, dict(pred), infer_status=infer_status)
        evaluator_no_cons_con.evaluate_scene_graph(gt_annotation, dict(pred), infer_status=infer_status)
        evaluator_cons_all.evaluate_scene_graph(gt_annotation, dict(pred), infer_status=infer_status)
        evaluator_no_cons_all.evaluate_scene_graph(gt_annotation, dict(pred), infer_status=infer_status)


print("Unary Inference Done")
print('-------------------------with constraint-------------------------------')
print('..................Attention Type.......................................')
cons_att_per_rel = evaluator_cons_att.print_stats()
print('..................Spatial Type.........................................')
cons_spa_per_rel = evaluator_cons_spa.print_stats()
print('..................Contacting Type.........................................')
cons_con_per_rel = evaluator_cons_con.print_stats()
print('..................All Type.......................................')
cons_all_per_rel = evaluator_cons_all.print_stats()
print('done')

print('-------------------------no constraint-------------------------------')
print('..................Spatial Type.........................................')
no_cons_spa_per_rel = evaluator_no_cons_spa.print_stats()
print('..................Contacting Type.........................................')
no_cons_con_per_rel = evaluator_no_cons_con.print_stats()
print('..................All Type.........................................')
no_cons_all_per_rel = evaluator_no_cons_all.print_stats()
print('done')




