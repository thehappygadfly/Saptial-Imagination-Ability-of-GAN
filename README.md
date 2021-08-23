# Unsupervised single-image 3D reconstruction using GAN's imagination ability


Hi, I am Weijian.
Our model has two goals. The first goal is to generate multi-views of the same 
object. The second goal is to realize 3D re construction.

———
Some of the codes are based on:
https://github.com/elliottwu/unsup3d
https://github.com/XingangPan/GAN2Shape
https://github.com/rosinality/stylegan2-pytorch
———
Code dependency
Pytorch == 1.2.0
neural_render_pytorch
———
Using command below to setup environment
!pip install torch==1.2.0 torchvision==0.4.0 moviepy==1.0.0 moviepy pyyaml tensorboardX mmcv neural_renderer_pytorch
—————
Train
!python run.py --config ./experiments/train_cat.yml 
--gpu 0
