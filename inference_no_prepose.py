"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
import argparse
import os
import ast
import numpy as np
import torch as th
import torch.distributed as dist

from diffusion import logger
from diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)
from config.parse_config import parse_config
from scipy.signal import savgol_filter
from dataloader.diffusion import Diffusiondataset, val_collate_fn
from torch.utils.data import DataLoader

import joblib as jl
from pymo.viz_tools import *
from pymo.writers import *

import warnings
warnings.filterwarnings('ignore')

def main(config, pre_config):
    args = create_argparser().parse_args()

    logger.configure()

    dataset = Diffusiondataset(config, pre_config, phase='val')
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn = val_collate_fn)

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(use_prepose=False,
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.load_state_dict(
        th.load(args.model_path, map_location="cpu")
    )
    model.to("cuda:1")
    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()

    logger.log("sampling...")
    for file_idx, batch in enumerate(dataloader) :
        name, clip_length, samples = batch
        print(name)
        output_pose = [list() for _ in range(round(clip_length * 30))]
        for idx, sample in enumerate(samples) :
            pose = sample['pose']
            audio = sample['audio']
            text = sample['text']
            start_time = sample['start']
            end_time = sample['end']
            start_frame = round(start_time * 30)
            end_frame = round(end_time * 30)

            pose, audio, text = th.tensor(pose).float().to('cuda:1'), th.tensor(audio).float().to('cuda:1'), th.tensor(text).float().to('cuda:1')
            pose, audio, text = pose.unsqueeze(0), audio.unsqueeze(0), text.unsqueeze(0)

            sample_fn = (
                diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
            )

            sample = sample_fn(
                model,
                (1, 128, 128),
                clip_denoised=args.clip_denoised,
                progress=True,
                model_kwargs={'audio' : audio, 'text' : text},
            )
            sample = sample.squeeze(0)

            #pose = pose.squeeze(0)
            #pose = dataset.recon_pose_models(pose)
            sample = dataset.recon_pose_models(sample)

            for frame_idx in range(len(sample)) :
                if frame_idx + start_frame < len(output_pose) :
                    output_pose[frame_idx + start_frame].append(sample[frame_idx])

        for i in range(len(output_pose)) :
            output_pose[i] = np.mean(np.array(output_pose[i]), axis=0)
        output_pose = np.array(output_pose)
        output_pose = savgol_filter(output_pose, 9, 3, axis=0)
        convert(output_pose, name, config)


    logger.log("sampling complete")


def convert(predicted_gesture, name, config, structure="full") :
    if structure == "full":
        predicted_gesture[:, 21:24] = np.mean(predicted_gesture[:, 21:24], axis=0)
        predicted_gesture[:, 27] = np.clip(predicted_gesture[:, 27], -9999, 0.6)
        predicted_gesture[:, 39] = np.clip(predicted_gesture[:, 39], -9999, -2.0)
        predicted_gesture[:, 40] = np.clip(predicted_gesture[:, 40], -9999, 0.4)
        predicted_gesture[:, 41] = np.clip(predicted_gesture[:, 41], -9999, 0.4)
        predicted_gesture[:, 45] = np.clip(predicted_gesture[:, 45], -9999, 0.6)
        predicted_gesture[:, 12] = np.clip(predicted_gesture[:, 12], -9999, 1.4)
        predicted_gesture[:, 3] = np.clip(predicted_gesture[:, 3], -9999, 1.4)
        pipeline = jl.load("ckpts/pipeline/pipeline_expmap_full.sav")

    else:
        predicted_gesture[:, 21 - 18:24 - 18] = np.mean(predicted_gesture[:, 21 - 18:24 - 18], axis=0)
        predicted_gesture[:, 27 - 18] = np.clip(predicted_gesture[:, 27 - 18], -9999, 0.6)
        predicted_gesture[:, 39 - 18:42 - 18] = np.mean(predicted_gesture[:, 39 - 18:42 - 18], axis=0)
        predicted_gesture[:, 45 - 18] = np.clip(predicted_gesture[:, 45 - 18], -9999, 0.6)
        pipeline = jl.load("ckpts/pipeline/pipeline_expmap_upper.sav")

    bvh_data = pipeline.inverse_transform([predicted_gesture])[0]
    writer = BVHWriter()
    os.makedirs(os.path.join(config.outputpath, "no_prepose"), exist_ok=True)
    with open(os.path.join(config.outputpath, "no_prepose", name + '.bvh'), 'w') as f:
        writer.write(bvh_data, f, framerate=30)

def create_argparser():
    defaults = dict(
        clip_denoised=True,
        num_samples=10000,
        batch_size=1,
        use_ddim=False,
        model_path="C://Users//PC//PycharmProjects_2023//GENEA2023//ckpts//diffusion_no_prepose//model039000.pt",
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser
if __name__ == "__main__":
    config = parse_config('diffusion')
    pre_config = parse_config('pretrain')
    main(config, pre_config)
