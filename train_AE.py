import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from torch.utils.data import DataLoader

from config.parse_config import parse_config
from dataloader.pretrain import Pretraindataset
from models.pretrain import *
from trainer.pretrain import *

def main(config) :
    train_dataset = Pretraindataset(config, phase='train')
    val_dataset = Pretraindataset(config, phase='val')

    train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)

    pose_embedder = PoseEmbedder(config)
    motion_embedder = MotionEmbedder(config)
    motion_decoder = MotionDecoder(config)
    pose_decoder = PoseDecoder(config)

    model = [pose_embedder, motion_embedder, motion_decoder, pose_decoder]

    pretrain_pose_trainer(train_dataloader, val_dataloader, model, config)
    pretrain_motion_trainer(train_dataloader, val_dataloader, model, config)

if __name__ == "__main__" :
    config = parse_config("pretrain")
    main(config)