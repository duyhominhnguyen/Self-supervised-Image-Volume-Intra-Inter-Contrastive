# Joint Self-Supervised Image-Volume Representation Learning with Intra-inter Contrastive Clustering

We release the algorithm in our paper accepted in [AAAI Conference on Artificial Intelligence 2023](https://ojs.aaai.org/index.php/AAAI/article/view/26687). 

- :construction: Downstream task illustrations.
- :mega: **09.01.2024**: Updating scripts to train 2D self-supervised baselines.
- :mega: **24.12.2023**: 3D self-supervised parts in the proposed framework are ready.

<p align="center">
   <img src="./figures/Overview.png" alt="Overview" width="698.5"/>
</p>

  
## 2D SSL Baselines Training

### Anaconda Environment

Install VISSL for 2D SSL training by following [this instruction](https://github.com/facebookresearch/vissl/blob/main/INSTALL.md). You should place the `vissl` folder inside this repository for convenient.

### Data Preparation
The algorithm takes 2D images in whichever format supported by VISSL (PNG, JPG, etc.). An example for the folder structure of training data is given in the `training_data_2D` folder. The structure must be as follows
```
training_data_2D
   |
   + train
   |   |
   |   + Dataset_1
   |   |   |
   |   |   + 001.jpg
   |   |   + 002.jpg
   |   |   + ...
   |   |
   |   + Dataset_2
   |   |   |
   |   |   + 001.jpg
   |   |   + 002.jpg
   |   |   + ...
   |   |
   |   + Dataset_3
   |   |   |
   |   |   + 001.jpg
   |   |   + 002.jpg
   |   |   + ...
   |   ...
   |
   + test
   |   |
   |   + Dataset_1
   |   |   |
   |   |   + 001.jpg
   |   |   + 002.jpg
   |   |   + ...
   |   |
   |   + Dataset_2
   |   |   |
   |   |   + 001.jpg
   |   |   + 002.jpg
   |   |   + ...
   |   |
   |   + Dataset_3
   |   |   |
   |   |   + 001.jpg
   |   |   + 002.jpg
   |   |   + ...
   |   ...
   |
   + valid
       |
       + Dataset_1
       |   |
       |   + 001.jpg
       |   + 002.jpg
       |   + ...
       |
       + Dataset_2
       |   |
       |   + 001.jpg
       |   + 002.jpg
       |   + ...
       |
       + Dataset_3
       |   |
       |   + 001.jpg
       |   + 002.jpg
       |   + ...
       ...
```

After preparing the training data, add these lines (using absolute paths) to the file `vissl/configs/config/dataset_catalog.json` in your VISSL installation location
```
"ssl_2d":{
   "train": ["/path/to/Training_Data_2D/train/", "<unused>"],
   "val": ["/path/to/Training_Data_2D/valid/", "<unused>"],
   "test": ["/path/to/Training_Data_2D/test/", "<unused>"]
},
``` 

### Training Scripts

Use the following commands to train 2D SSL methods using 2 GPUs. You may change the settings to suit your needs.

```bash
# For DeepCluster method
python ./vissl/tools/run_distributed_engines.py \
  config=pretrain/deepcluster_v2/deepclusterv2_2crops_resnet \
  config.DATA.TRAIN.DATA_SOURCES=[disk_folder] \
  config.DATA.TRAIN.DATASET_NAMES=[ssl_2d] \
  config.CHECKPOINT.DIR="./weights/2D_SSL_DeepCluster" \
  config.OPTIMIZER.num_epochs=105 \
  config.DISTRIBUTED.NUM_PROC_PER_NODE=2 \
  config.DISTRIBUTED.NUM_NODES=1 \
  config.DISTRIBUTED.RUN_ID=auto \
  config.HOOKS.TENSORBOARD_SETUP.USE_TENSORBOARD=true
# For SwAV method
python ./vissl/tools/run_distributed_engines.py \
  config=pretrain/swav/swav_8node_resnet \
  config.DATA.TRAIN.DATA_SOURCES=[disk_folder] \
  config.DATA.TRAIN.DATASET_NAMES=[ssl_2d] \
  config.CHECKPOINT.DIR="./weights/2D_SSL_SwAV" \
  config.OPTIMIZER.num_epochs=105 \
  config.DISTRIBUTED.NUM_PROC_PER_NODE=2 \
  config.DISTRIBUTED.NUM_NODES=1 \
  config.DISTRIBUTED.RUN_ID=auto \
  config.HOOKS.TENSORBOARD_SETUP.USE_TENSORBOARD=true
```

After training, the checkpoints in `.torch` format will be saved in the folder indicated by `config.CHECKPOINT.DIR`. Please use `vissl/extra_scripts/convert_vissl_to_torchvision.py` to convert them to torchvision format for the next step. We also provide converted pre-trained weights for 2D SSL methods in this repository (`resnet_deepcluster2D.pth` and `resnet_swav2D.pth` in the folder `weights`) to save your time.

## 3D SSL Training

### Anaconda Environment

Due to conflicts in different versions of nVIDIA Apex, one must deactivate the anaconda environment used in 2D SSL training before creating the environment for 3D SSL training.

First, create anaconda environment from file as
```
conda env create -f env.yml
conda activate joint-ssl
```
Then, install nVIDIA Apex as
```
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --config-settings "--build-option=--cpp_ext" --config-settings "--build-option=--cuda_ext" ./
```

### Data Preparation
The algorithm takes 3D images of size `H x W x C` in NPY format as inputs. An example for the folder structure of training data is given in the `training_data_3D` folder. The structure must be as follows
```
training_data_3D
   |
   + Dataset_1
   |    |
   |    + 001.npy
   |    + 002.npy
   |    + ...
   |
   + Dataset_2
   |    |
   |    + 001.npy
   |    + 002.npy
   |    + ...
   |
   + Dataset_3
   |    |
   |    + 001.npy
   |    + 002.npy
   |    + ...
   ...
```

### Training Scripts

First, we train the mask embedding. This step requires weights from 2D SSL (included in this repository). The commands are as follows
```bash
# For DeepCluster method
python -m torch.distributed.launch --nproc_per_node=2 mask_deepcluster_3D.py
# For SwAV method
python -m torch.distributed.launch --nproc_per_node=2 mask_swav_3D.py
```
After training the mask embedding, we will obtain the weights `mask_deepcluster.pth` and `mask_swav.pth` in the folder `weights`.

Then, we train joint 2D-3D SSL. This step requires weights from 2D SSL (`resnet_deepcluster2D.pth` or `resnet_swav2D.pth`) and corresponding weights from embedding training (`mask_deepcluster.pth` or `mask_swav.pth`). These weights are available in the folder `weights`. The commands are as follows
```bash
# For DeepCluster method
python -m torch.distributed.launch --nproc_per_node=2 ssl_deepcluster_3D.py
# For SwAV method
python -m torch.distributed.launch --nproc_per_node=2 ssl_swav_3D.py
```
After training, we will obtain the weights `deepcluster3D.pth` and `swav3D.pth` in the current folder.

## Downstream Tasks
After obtaining the weights `deepcluster3D.pth` and `swav3D.pth` from joint 2D-3D SSL training, we use these weights for downstream tasks. In this repository, we provide the scripts for demonstrating the segmentation task in MMWHS dataset (both CT and MRI). 

### Data Preparation
First, download the dataset from [this link](https://zmiclab.github.io/zxh/0/mmwhs/data.html). Please note that we only use the folders `ct_train` and `mri_train` since the test sets do not contain ground-truths. The train/test splits are as follows for both CT and MRT datasets (the numbers are image indices)
```
train:
1001, 1003, 1005, 1006, 1008, 1009, 1011,
1014, 1015, 1016, 1017, 1018, 1020

validation:
1004, 1007, 1013

test:
1002, 1010, 1012, 1019
```
Then, after downloading and organizing the data, use the scripts `mmwhs_ct.py` and `mmwhs_mr.py` in the folder `./downstream/preprocessing/` to pre-process the NII images (adjust the folder paths accordingly).

### Fine-tuning and Testing
First, update the path of `deepcluster3D.pth` and `swav3D.pth` in
```
./downstream/segmentation_models.pytorch/segmentation_models_pytorch/encoders/resnet.py
```
by editing lines 79 and 80. Then, go to `./downstream/segmentation_models.pytorch/` and run
```
pip install -e .
```
to install our customized `segmentation_models.pytorch` package.

Then, updates the paths in `./downstream/train_MMWHS_argument.py` in lines 34-52 to the preprocessed datasets created in the previous `Data Preparation` step.

Finally, run the following commands for fine-tuning and testing (suppose that the current folder is `downstream`)
```
python train_MMWHS_argument.py -pt joint-deepcluster -ds mmwhs_ct -od ./checkpoints/swav_ct -gpu 0 -l 0.01
```
where
* `-pt`: `joint-deepcluster` or `joint-swav`, name of checkpoint for the joint-ssl methods.
* `-ds`: downstream dataset, `mmwhs_ct` for MMWHS-CT and `mmwhs_mri` for MMWHS-MRI.
* `-od`: output directory for fine-tuned weights.
* `-gpu`: GPU ID (equivalent to `CUDA_VISIBLE_DEVICES`).
* `-l`: learning rate.

## Citation
```bib
@inproceedings{nguyen2023joint,
  title={Joint self-supervised image-volume representation learning with intra-inter contrastive clustering},
  author={Nguyen, Duy MH and Nguyen, Hoang and Mai, Truong TN and Cao, Tri and Nguyen, Binh T and Ho, Nhat and Swoboda, Paul and Albarqouni, Shadi and Xie, Pengtao and Sonntag, Daniel},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={37},
  number={12},
  pages={14426--14435},
  year={2023}
}
```
