
## Image Captioning Eos tests

This repository was used for the experiments described in [SacreEOS](https://arxiv.org/abs/2305.12254).

### Requirements
* python >= 3.7 
* numpy
* torch
* h5py
* sacreeos

### Data preparation

The MS-COCO 2014 visual inputs are generously provided by 
[here](https://github.com/peteanderson80/bottom-up-attention), where features are extracted by a Faster-RCNN 
using the ResNet-101 backbone. The data can be converted into the required format of 
this repository using our `features_generation.py` script by specifying the source file location:
```
python features_generation.py --tsv_features_path trainval_file_path
```

Download `dataset_coco.json` [here](https://drive.google.com/drive/folders/1z-kVvVsOhcW6QSPqB27h5ta5XsRGG5sv?usp=sharing)
and place it in the directory `github_ignore_material/raw_data/`. <br>

### Usage

XE Training: 
```
python train.py $(cat confs/xe.conf) &> xe_output.txt &
```

SCST with eos training: 
```
python train.py $(cat confs/rf_standard.conf) &> rl_std_output.txt &
```

SCST without eos training: 
```
python train.py $(cat confs/rf_no_eos.conf) &> rl_noeos_output.txt &
```

## Credits

If you find this repository useful, please consider citing (no obligation):
```
@article{hu2023request,
  title={A request for clarity over the End of Sequence token in the Self-Critical Sequence Training},
  author={Hu, Jia Cheng and Cavicchioli, Roberto and Capotondi, Alessandro},
  journal={arXiv preprint arXiv:2305.12254},
  year={2023}
}
```