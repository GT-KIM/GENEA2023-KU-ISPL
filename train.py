import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"
import argparse

from diffusion import logger
from diffusion.resample import create_named_schedule_sampler
from diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)
from diffusion.train_util import TrainLoop

from config.parse_config import parse_config
from dataloader.diffusion  import Diffusiondataset, collate_fn, collate_fn_no_prepose
from torch.utils.data import DataLoader

def main(config, pre_config) :
    args = create_argparser().parse_args()

    logger.configure()
    logger.log("creating model and diffusion...")

    model, diffusion = create_model_and_diffusion(config.use_prepose,
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.to('cuda:0')
    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)

    logger.log("creating data loader...")
    dataset = Diffusiondataset(config, pre_config, phase='train')
    if config.use_prepose :
        data = DataLoader(dataset, batch_size=config.batch_size, shuffle=True, collate_fn = collate_fn)
    else :
        data = DataLoader(dataset, batch_size=config.batch_size, shuffle=True, collate_fn = collate_fn_no_prepose)


    logger.log("training...")
    TrainLoop(
        model=model,
        diffusion=diffusion,
        data=data,
        batch_size=args.batch_size,
        microbatch=args.microbatch,
        lr=args.lr,
        ema_rate=args.ema_rate,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        resume_checkpoint=args.resume_checkpoint,
        use_fp16=args.use_fp16,
        fp16_scale_growth=args.fp16_scale_growth,
        schedule_sampler=schedule_sampler,
        weight_decay=args.weight_decay,
        lr_anneal_steps=args.lr_anneal_steps,
        pose_model = dataset.pose_model,
    ).run_loop()


def create_argparser():
    defaults = dict(
        data_dir="",
        schedule_sampler="uniform",
        lr=1e-4,
        weight_decay=0.0,
        lr_anneal_steps=0,
        batch_size=8,
        microbatch=-1,  # -1 disables microbatches
        ema_rate="0.9999",  # comma-separated list of EMA values
        log_interval=10,
        save_interval=10000,
        resume_checkpoint=None, #"ckpts/diffusion_large/model039000.pt",
        use_fp16=False,
        fp16_scale_growth=1e-3,
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    config = parse_config('diffusion')
    pre_config = parse_config('pretrain')
    main(config, pre_config)
