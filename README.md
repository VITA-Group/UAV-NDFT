# UAV-AdversarialLearning
## PyTorch Code for 'Delving into Robust Object Detection from Unmanned Aerial Vehicles: A Deep Nuisance Disentanglement Approach'

## Introduction

PyTorch Implementation of our ICCV 2019 paper ["Delving into Robust Object Detection from Unmanned Aerial Vehicles: A Deep Nuisance Disentanglement Approach
"](https://arxiv.org/abs/1908.03856).

## Environment
* Python 2.7 or 3.6
* Pytorch 0.4.0
* CUDA 8.0 or higher
## Multiple GPU Training
```{r, engine='bash', count_lines}
#!/bin/bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python trainval_net_monitor.py --use_tfb  --cuda --mGPUs --monitor_discriminator --use_restarting --use_adversarial_loss --gamma_altitude 0.01 --gamma_angle 0.01 --gamma_weather 0.01 --angle_thresh 0.9 --altitude_thresh 0.9 --weather_thresh 0.9 --bs 8
```
## Single GPU Testing
```{r, engine='bash', count_lines}
#!/bin/bash
for ((i=1; i<=7; i++))
do
        epoch=$(($i*1000/3138+1))
        ckpt=$(($i*1000%3138))
        echo "$epoch"
        echo "$ckpt"
        CUDA_VISIBLE_DEVICES=1 python test_net.py --cuda --checkepoch "$epoch" --checkpoint "$ckpt" --gamma_altitude 0.0 --gamma_angle 0.01 --gamma_weather 0.01 --use_restarting
done

```
## Pretrained Model on UAVDT
Google Drive: https://drive.google.com/file/d/1kw-QpnBW5RkKfoG9iH83DxH9uhCgIxix/view?usp=sharing

## UAVDT Data (Training+Testing) in Pascal VOC Format
Google Drive: https://drive.google.com/file/d/1pvMjEr6LsrISpx-GuOLJW53toF35JkRy/view?usp=sharing

## UAVDT Trained Model (w/o Adversarial Loss and w/ Adversarial Loss)
Google Drive: https://drive.google.com/file/d/1rxqr0Cq0y9cXhdWyNd_R_8cd68exD1wn/view?usp=sharing

## VisDrone Meta-data Link
Google Drive: https://drive.google.com/file/d/1FcdPJXggs31HpYsNfqyZFMsI3qwPB54Q/view?usp=sharing

## Project Directory Layout
```
.
├── cfgs
├── data              # UAVDT dataset with annotation
├── images
├── lib
├── logs              # TensorBoard event files
├── models            # Trained model (w/ adversarial loss and w/o adversarial loss)
├── output
├── summaries         # Summary files recording the training and validation performance
├── README.md
├── _init_paths.py
├── bash_run.sh       # Run the testing in batch
├── demo.py
├── requirements.txt
├── test_net.py
├── trainval_net.py
└── trainval_net_monitor.py
```
## Citation

If you find this code useful, please cite the following paper:
```BibTex
@article{wu2019delving,
  title={Delving into Robust Object Detection from Unmanned Aerial Vehicles: A Deep Nuisance Disentanglement Approach},
  author={Wu, Zhenyu and Suresh, Karthik and Narayanan, Priya and Xu, Hongyu and Kwon, Heesung and Wang, Zhangyang},
  journal={arXiv preprint arXiv:1908.03856},
  year={2019}
}
```
