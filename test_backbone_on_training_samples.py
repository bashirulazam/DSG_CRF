import numpy as np
np.set_printoptions(precision=4)
import copy
import torch
import pickle

from dataloader.action_genome import AG, cuda_collate_fn

from lib.config import Config
from lib.object_detector import detector
from lib.backbone import STTran


conf = Config()
for i in conf.args:
    print(i,':', conf.args[i])

AG_dataset = AG(mode="train", datasize=conf.datasize, data_path=conf.data_path, filter_nonperson_box_frame=True,
                filter_small_box=False if conf.mode == 'predcls' else True)
dataloader = torch.utils.data.DataLoader(AG_dataset, shuffle=False, num_workers=0, collate_fn=cuda_collate_fn)

gpu_device = torch.device('cuda:0')
object_detector = detector(train=False, object_classes=AG_dataset.object_classes, use_SUPPLY=True, mode=conf.mode).to(device=gpu_device)
object_detector.eval()


backbone_model = STTran(mode=conf.mode,
               attention_class_num=len(AG_dataset.attention_relationships),
               spatial_class_num=len(AG_dataset.spatial_relationships),
               contact_class_num=len(AG_dataset.contacting_relationships),
               attention_classes = AG_dataset.attention_relationships,
               spatial_classes = AG_dataset.spatial_relationships,
               contacting_classes = AG_dataset.contacting_relationships,
               obj_classes=AG_dataset.object_classes,
               enc_layer_num=conf.enc_layer,
               dec_layer_num=conf.dec_layer).to(device=gpu_device)


backbone_model.eval()


backbone_ckpt = torch.load(conf.backbone_model_path, map_location=gpu_device)
backbone_model.load_state_dict(backbone_ckpt['state_dict'], strict=False)
print('*'*50)
print('CKPT {} is loaded'.format(conf.backbone_model_path))


def del_keys(pred, val_status=None):
    if conf.mode=='predcls':
        del pred['features']
        del pred['union_feat']
        del pred['union_box']
        del pred['spatial_masks']

    if conf.mode=='sgcls':
        del pred['features']
        del pred['union_feat']
        del pred['union_box']
        del pred['spatial_masks']
        del pred['fmaps']

    if conf.mode=='sgdet':
        if val_status==1:
            del pred['features']
            del pred['union_feat']
            del pred['union_box']
            del pred['spatial_masks']
            del pred['fmaps']
        if val_status==0:
            del pred['features']
            del pred['union_feat']
            del pred['spatial_masks']

    return pred


def add_frame_keys(pred, frame_names):

    pred['frame_list'] = []
    for i in range(0,len(pred['human_idx'])):
        pred['frame_list'].append(frame_names[i])

    return pred

valList = []

with open('valFiles.txt', 'r') as fp:
    for line in fp:
        valfile = line[:-1].split('.')[0]
        valList.append(valfile)


print("Validation files loaded")


infer_status = 1
val_status = 0
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
        if conf.mode == 'sgdet':
            if vid_name not in valList:
                object_detector.is_train = True
                backbone_model.object_classifier.training = True
                val_status = 0
            if vid_name in valList:
                object_detector.is_train = False
                backbone_model.object_classifier.training = False
                val_status = 1
        entry = object_detector(im_data, im_info, gt_boxes, num_boxes, gt_annotation, im_all=None)
        pred = backbone_model(entry)
        pred = del_keys(pred,val_status)
        result = [gt_annotation, pred]
        torch.save(result, 'results/' + conf.mode + '_backbone_training/' + vid_name + '.pt')








