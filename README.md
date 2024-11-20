# Within-Triplet CRF for Dynamic Scene Graph Generation
We propose a Within-Triplet Transformer-based CRF model **WT-CRF** to generate dynamic scene graphs of the given video. **WT-CRF** computes the unary and temporal potential of a relationship pair given the local-global within-triplet features and combines these potentials with predicted weights in a Conditional Random Field (CRF) framework. 


## Installation ## 

We followed the installation instructions from  [Cong's STTran](https://github.com/yrcong/STTran) repo. 

### Requirements
- python=3.6
- pytorch=1.1
- scipy=1.1.0
- torchvision=0.3
- cypthon
- dill
- easydict
- h5py
- opencv
- pandas
- tqdm
- yaml

We borrow some compiled code for bbox operations.
```
cd lib/draw_rectangles
python setup.py build_ext --inplace
cd ..
cd fpn/box_intersections_cpu
python setup.py build_ext --inplace
```
For the object detector part, please follow the compilation from https://github.com/jwyang/faster-rcnn.pytorch
We provide a pretrained FasterRCNN model for Action Genome. Please download [here](https://drive.google.com/file/d/1-u930Pk0JYz3ivS6V_HNTM1D5AxmN5Bs/view?usp=sharing) and put it in 
```
fasterRCNN/models/faster_rcnn_ag.pth
```

### Dataset
We use the dataset [Action Genome](https://www.actiongenome.org/#download) to train/evaluate our method. Please process the downloaded dataset with the [Toolkit](https://github.com/JingweiJ/ActionGenome). The directories of the dataset should look like:
```
|-- action_genome
    |-- annotations   #gt annotations
    |-- frames        #sampled frames
    |-- videos        #original videos
```
 In the experiments for SGCLS/SGDET, we only keep bounding boxes with short edges larger than 16 pixels. Please download the file [object_bbox_and_relationship_filtersmall.pkl](https://drive.google.com/file/d/19BkAwjCw5ByyGyZjFo174Oc3Ud56fkaT/view?usp=sharing) and put it in the ```dataloader```

## Preparing Local-Global features for train/test

### Generating local features with STTran backbone

+ For pretrained $BACKBONE_MODEL_PATH ([PredCls](https://github.com/bashirulazam/DSG_CRF/releases/download/Backbone_v1.0.0/frame_level_sttran_predcls.tar), [SGCls](https://github.com/bashirulazam/DSG_CRF/releases/download/Backbone_v1.0.0/frame_level_sttran_sgcls.tar), [SGDet](https://github.com/bashirulazam/DSG_CRF/releases/download/Backbone_v1.0.0/frame_level_sttran_sgdet.tar )) for $mode = predcls, sgcls, sgdet:
     + for Training samples
    ```
    CUDA_VISIBLE_DEVICES=0 python test_backbone_on_training_samples.py -mode $mode -datasize large -data_path dataset/ag/ -backbone_model_path $BACKBONE_MODEL_PATH
    ```
    For each training video $vid_name with $mode the precomputed results with features will be saved in ```'results/' + conf.mode + '_backbone_training/' + vid_name + '.pt'```    

    + for Testing samples
    ```
    CUDA_VISIBLE_DEVICES=0 python test_backbone_on_testing_samples.py -mode $mode -datasize large -data_path dataset/ag/ -backbone_model_path $BACKBONE_MODEL_PATH
    ```
    For each testing video $vid_name with $mode the precomputed results with features will be saved in ```'results/' + conf.mode + '_backbone_testing/' + vid_name + '.pt'```
    
### Generating global geatures with DINO_v2

The environment setting for local feature generation can not be applied for generating DINO based global frame features. Therefore, we precompute the dino features for each frame and dump them as numpy binary files. To install DINO_v2, please visit their github page [here](https://github.com/facebookresearch/dinov2). 

+ For all video samples, run the following script to precompute the DINO_v2 frame features for global context 
```
CUDA_VISIBLE_DEVICES=0 python extract_dino_features_from_frames.py
```

### Appending local and global features 

+ For both training and testing in all three settings (predcls, sgcls, sgdet), run the following script to append the local and global features 

```
CUDA_VISIBLE_DEVICES=0 python add_dino_features_to_backbone_results.py 
```
This script will load the local precomputed features for each relationships of each video frames and append the global DINO_v2 features to each of them and save them in another directory with the format  ```'results/' + conf.mode + '_backbone_training_with_dino/' ``` (for training) and  ```'results/' + conf.mode + '_backbone_testing_with_dino/' ``` (for testing). 
## Piecewise training of unary, temporal, and weight predicting model

### Training unary and temporal model
You can train the unary and temporal transformer with train_unary_and_temporal.py. We train for 50 epochs.  
+ For $mode = {predcls, sgcls, sgdet}: 
```
CUDA_VISIBLE_DEVICES=0 python train_unary_and_temporal.py -mode $mode  
```

### Validation study to choose the best uanry and temporal model
We have created a validation list of 1000 videos from training set which is not fed to the training procedure. You can run this validation script to report decomposed performance for each epoch and choose the best performing model for the final evaluation on the testing dataset. 
+ For $mode = {predcls, sgcls, sgdet}: 
```
CUDA_VISIBLE_DEVICES=0 python val_unary_and_temporal.py -mode $mode  
```


### Train weight predicintig model to combine unary and temporal 
With the best unary and temporal model, we train the weight model which predict the weights for unary and temporal clique for each relationship
```
CUDA_VISIBLE_DEVICES=0 python train_weight.py -unary_model_path $unary_model_path -temporal_model_path $temporal_model_path
```


## Evaluation
We can evaluate the **WT-CRF** with the following code
+ For PredCLS ([Unary](https://github.com/bashirulazam/DSG_CRF/releases/download/PredCls_v1.0.0/best_model_unary_only.pt), [Temporal](https://github.com/bashirulazam/DSG_CRF/releases/download/PredCls_v1.0.0/best_model_temporal_only_first_order.pt), [Weight](https://github.com/bashirulazam/DSG_CRF/releases/download/PredCls_v1.0.0/weight_model.pt)): 
```
CUDA_VISIBLE_DEVICES=0 python test_unary_and_temporal.py -mode predcls -datasize large -data_path dataset/ag/ -backbone_result_folder results/predcls_backbone_with_dino/ -unary_model_path $unary_model_path  -temporal_model_path $temporal_model_path -weight_model_path $weight_model_path
```
+ For SGCLS ([Unary](https://github.com/bashirulazam/DSG_CRF/releases/download/SGCls_v1.0.0/best_model_unary_only.pt), [Temporal](https://github.com/bashirulazam/DSG_CRF/releases/download/SGCls_v1.0.0/best_model_temporal_only_first_order.pt), [Weight](https://github.com/bashirulazam/DSG_CRF/releases/download/SGCls_v1.0.0/weight_model.pt)): : 
```
CUDA_VISIBLE_DEVICES=0 python test_unary_and_temporal.py -mode sgcls -datasize large -data_path dataset/ag/ -backbone_result_folder results/sgcls_backbone_with_dino/ -unary_model_path $unary_model_path  -temporal_model_path $temporal_model_path -weight_model_path $weight_model_path
```
+ For SGDET ([Unary](https://github.com/bashirulazam/DSG_CRF/releases/download/SGDet_v1.0.0/best_model_unary_only.pt), [Temporal](https://github.com/bashirulazam/DSG_CRF/releases/download/SGDet_v1.0.0/best_model_temporal_only_first_order.pt), [Weight](https://github.com/bashirulazam/DSG_CRF/releases/download/SGDet_v1.0.0/weight_model.pt)): : 
```
CUDA_VISIBLE_DEVICES=0 python test_unary_and_temporal.py -mode sgdet -datasize large -data_path dataset/ag/ -backbone_result_folder results/sgdet_backbone_with_dino/ -unary_model_path $unary_model_path  -temporal_model_path $temporal_model_path -weight_model_path $weight_model_path


