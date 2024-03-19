# GM-MFA


## Abstract

Bird's eye view(BEV) segmentation map is a recent development in autonomous driving that provide effective environmental information, such as drivable areas and lane dividers. Most of the existing methods use surround view cameras and LiDAR as inputs for segmentation and the fusion between modalities is done using only simple splicing, which lacks exploiting the correlation and complementarity between modalities. Furthermore, the models are not very scalable to a wider variety of modalities.
This paper presents GM-MFA(Group-mix attention Multimodal Feature Aggregator), an end-to-end learning framework that can adapt to multiple modal feature combinations for BEV segmentation. GM-MFA comprises the following components: (i) The camera has a dual-branch structure that strengthens the linkage between local and global features. (ii) Multi-head deformable cross-attention is applied as multimodal feature aggregators to aggregate image, LiDAR, and Radar feature maps in BEV for implicitly aligning multimodal BEV features. (iii) The Group-Mix attention is used to enrich the attention map feature space and enhance the correlation between features.
The method completes training and validation on the unScenes dataset, which is 5 mIoU points higher than the baseline method. Additionally, feature heatmaps and ablation studies are provided to demonstrate the method's effectiveness.
## Results

### BEV Map Segmentation (on nuScenes validation)

|   Model   | Modality |  mIoU  | 
|:---------:|:--------:|:------:|
| BEVFusion |  C+L+R   | 0.5499 |
|    MFA    |    C+L+R     | 57.09  |
|  GM-MFA   |    C+L+R     | 48.56  |

## Usage

### Prerequisites

The code is built with following libraries:

- Python >= 3.8, \<3.9
- OpenMPI = 4.0.4 and mpi4py = 3.0.3 (Needed for torchpack)
- Pillow = 8.4.0 (see [here](https://github.com/mit-han-lab/bevfusion/issues/63))
- [PyTorch](https://github.com/pytorch/pytorch) >= 1.9, \<= 1.10.2
- [tqdm](https://github.com/tqdm/tqdm)
- [torchpack](https://github.com/mit-han-lab/torchpack)
- [mmcv](https://github.com/open-mmlab/mmcv) = 1.4.0
- [mmdetection](http://github.com/open-mmlab/mmdetection) = 2.20.0
- [nuscenes-dev-kit](https://github.com/nutonomy/nuscenes-devkit)

After installing these dependencies, please run this command to install the codebase:

```bash
python setup.py develop
```

### Data Preparation

#### nuScenes

Please follow the instructions from [here](https://github.com/open-mmlab/mmdetection3d/blob/master/docs/en/datasets/nuscenes_det.md) to download and preprocess the nuScenes dataset. Please remember to download both detection dataset and the map extension (for BEV map segmentation). After data preparation, you will be able to see the following directory structure (as is indicated in mmdetection3d):

```
mmdetection3d
├── mmdet3d
├── tools
├── configs
├── data
│   ├── nuscenes
│   │   ├── maps
│   │   ├── samples
│   │   ├── sweeps
│   │   ├── v1.0-test
|   |   ├── v1.0-trainval
│   │   ├── nuscenes_database
│   │   ├── nuscenes_infos_train.pkl
│   │   ├── nuscenes_infos_val.pkl
│   │   ├── nuscenes_infos_test.pkl
│   │   ├── nuscenes_dbinfos_train.pkl
│   │   ├── nuscenes_radar
│   │   │   ├── nuscenes_radar_infos_train_radar.pkl
│   │   │   ├── nuscenes_radar_infos_val_radar.pkl
```

### Evaluation

We also provide instructions for evaluating our models: 

```bash
./tools/download_pretrained.sh
```

### Training

We provide instructions to reproduce our results on nuScenes.

For example, if you want to train the C+R+L for our model, please run:

```bash
torchpack dist-run -np 4 python tools/train.py configs/groupmix-qkv.yaml
```
Note:
Train on the basis of [XXX.pth], please add --load_from [checkpoint name].pth.

Restore progress training to [XXX.pth], please add --resume_from [checkpoint name].pth
### Testing/Val

```bash
torchpack dist-run -np 4 python tools/test.py [config file path] pretrained/[checkpoint name].pth --eval map
```

Note: We will then update all checkpoint files



## Acknowledgements

----

## Citation

---
# GM-MFA
