import os
import argparse

import torch
import torch.nn.functional as F
from torchvision import utils
from model import Generator
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import norm


def generate(args, g_ema, device, mean_latent):
    with torch.no_grad():
        g_ema.eval()
        count = 0
        samples = []
        samples_w = []
        for i in tqdm(range(args.pics)):
           sample_z = torch.randn(args.sample, args.latent, device=device)
           sample_w = g_ema.style_forward(sample_z)

           sample, _ = g_ema([sample_w], truncation=args.truncation, truncation_latent=mean_latent, input_is_w=True)
           sample_w = mean_latent + args.truncation * (sample_w - mean_latent)
           
           samples.append(sample)
           samples_w.append(sample_w)
    return samples, samples_w

def interpolate_points(p1, p2, n_steps=4):
	# interpolate ratios between the points
	ratios = torch.linspace(0, 1, n_steps)
	# linear interpolate vectors
	vectors = list()
	for ratio in ratios:
		v = (1.0 - ratio) * p1 + ratio * p2
		vectors.append(v)
	return vectors

# create a plot of generated images
def plot_generated(examples, n):
    fig = plt.figure()

    for i in range(n):
        ax = fig.add_subplot(5,2,i+1)
        image = examples[i].squeeze(0).permute(1, 2, 0).cpu().numpy().shape
        ax.plot(image)
    fig.savefig('a.png')


 

if __name__ == '__main__':
    device = 'cuda'

    parser = argparse.ArgumentParser()

    parser.add_argument('--size', type=int, default=1024)
    parser.add_argument('--sample', type=int, default=1)
    parser.add_argument('--pics', type=int, default=8)
    parser.add_argument('--truncation', type=float, default=0.7)
    parser.add_argument('--truncation_mean', type=int, default=4096)
    parser.add_argument('--ckpt', type=str, default="stylegan2-ffhq-config-f.pt")
    parser.add_argument('--channel_multiplier', type=int, default=2)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--save_path', type=str, default='intepolation')
    parser.add_argument('--n_step', type=int, default=4)

    args = parser.parse_args()

    args.latent = 512
    args.n_mlp = 8

    torch.manual_seed(args.seed) # also sets cuda seeds

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
        os.makedirs(args.save_path + '/latents')

    g_ema = Generator(
        args.size, args.latent, args.n_mlp, channel_multiplier=args.channel_multiplier
    ).to(device)
    checkpoint = torch.load(args.ckpt)
    g_ema.load_state_dict(checkpoint['g_ema'], strict=False)

    if args.truncation < 1:
        with torch.no_grad():
            mean_latent = g_ema.mean_latent(args.truncation_mean)
    else:
        mean_latent = None

    samples, samples_w = generate(args, g_ema, device, mean_latent)
    print(len(samples))
    with torch.no_grad():
        g_ema.eval()
        interpolated = interpolate_points(samples_w[0], samples_w[1], args.n_step)
        new_samples = []
        for j in range(len(interpolated)):
            new_sample, _ = g_ema([interpolated[j]], truncation=args.truncation, truncation_latent=mean_latent, input_is_w=True)
            new_samples.append(new_sample)

        interpolated1 = interpolate_points(samples_w[0], samples_w[2], args.n_step)
        for j in range(len(interpolated)):
            new_sample, _ = g_ema([interpolated1[j]], truncation=args.truncation, truncation_latent=mean_latent, input_is_w=True)
            new_samples.append(new_sample)
        
        interpolated2 = interpolate_points(samples_w[0], samples_w[3], args.n_step)
        for j in range(len(interpolated)):
            new_sample, _ = g_ema([interpolated2[j]], truncation=args.truncation, truncation_latent=mean_latent, input_is_w=True)
            new_samples.append(new_sample)

        interpolated3 = interpolate_points(samples_w[0], samples_w[4], args.n_step)
        for j in range(len(interpolated)):
            new_sample, _ = g_ema([interpolated3[j]], truncation=args.truncation, truncation_latent=mean_latent, input_is_w=True)
            new_samples.append(new_sample)
    
        count=0
        for j in range(len(new_samples)):
            utils.save_image(
                    new_samples[j],
                    f'{args.save_path}/{str(count).zfill(6)}.png',
                    nrow=1,
                    normalize=True,
                    range=(-1, 1),
                )
            count+=1

