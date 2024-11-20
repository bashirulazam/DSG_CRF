import numpy as np
np.set_printoptions(precision=4)
import copy
import torch
import pickle
import os
from dataloader.action_genome import AG, cuda_collate_fn
from PIL import Image
from torchvision import models, transforms
from tqdm import tqdm
from lib.config import Config

conf = Config()
for i in conf.args:
    print(i,':', conf.args[i])
if __name__ == "__main__":
    gpu_device = torch.device('cuda:0')

    pretrained_model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitg14')
    pretrained_model.to(gpu_device)
    pretrained_model.eval()

    preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])




    #for mode in ["train", "test"]:
    # AG_dataset = AG(mode="train", datasize=conf.datasize, data_path=conf.data_path, filter_nonperson_box_frame=True,
    #                 filter_small_box=False if conf.mode == 'predcls' else True)
    # dataloader = torch.utils.data.DataLoader(AG_dataset, shuffle=False, num_workers=4,
    #                                                collate_fn=cuda_collate_fn)
    #with torch.no_grad():
    video_names = sorted(os.listdir(os.path.join(conf.data_path,'frames')))
    for v in video_names:
        print(v)
        frames = sorted(os.listdir(os.path.join(conf.data_path,'frames',v)))
        feat_dir = os.path.join('results/dino_features',v.split('.')[0])
        if not os.path.exists(feat_dir):
            os.makedirs(feat_dir)

        for f in frames:
            im_path  = os.path.join(conf.data_path, 'frames', v,f)
            print(im_path)
            im_name = f.split('.')[0]
            feat_file = os.path.join(feat_dir, im_name)
            print(feat_file)

            im = Image.open(im_path)
            im = im.convert('RGB')
            input_tensor = preprocess(im)
            input_batch = input_tensor.unsqueeze(0).to(gpu_device)
            feature_objects = pretrained_model(input_batch).cpu().detach().numpy()

            file = open(feat_file, "wb")
            np.save(file, feature_objects)
            #print(im_path)
            #print(feat_file)
            file.close

