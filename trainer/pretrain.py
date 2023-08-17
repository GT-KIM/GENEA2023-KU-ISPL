import os
import torch
from torch import nn, optim

def pretrain_pose_trainer(train_dataloader, val_dataloader, model, config) :
    pose_embedder = model[0].to('cuda:0')
    pose_decoder = model[3].to('cuda:0')

    optimizer = optim.AdamW(list(pose_embedder.parameters()) + list(pose_decoder.parameters()), lr = config.learning_rate)
    lr_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2, eta_min = config.learning_rate * 1e-3)
    criterion = nn.L1Loss()

    for epoch in range(config.epochs) :
        pose_embedder.train()
        pose_decoder.train()
        epoch_loss = 0.
        for train_step, batch in enumerate(train_dataloader) :
            optimizer.zero_grad()
            data = batch.float().to('cuda:0')

            embed = pose_embedder(data)
            recon = pose_decoder(embed)

            loss = criterion(recon, data)
            loss.backward()

            optimizer.step()
            #if train_step % 25 == 0 :
            #    print("[{}/{}][{}/{}] loss : {}".format(epoch, config.epochs, train_step, len(train_dataloader), loss.item()))

            epoch_loss += loss.item()
        epoch_loss /= len(train_dataloader)
        print("[{}/{}] train loss : {} lr : {}".format(epoch, config.epochs, epoch_loss, optimizer.param_groups[0]['lr']))
        lr_scheduler.step()

        pose_embedder.eval()
        pose_decoder.eval()
        epoch_loss = 0.
        for val_step, batch in enumerate(val_dataloader):
            data = batch.float().to('cuda:0')

            with torch.no_grad() :
                embed = pose_embedder(data)
                recon = pose_decoder(embed)

            loss = criterion(recon, data)
            #if val_step % 25 == 0:
            #    print("[{}/{}][{}/{}] loss : {}".format(epoch, config.epochs, val_step, len(val_dataloader), loss.item()))
            epoch_loss += loss.item()
        epoch_loss /= len(val_dataloader)
        print("[{}/{}] val loss : {}".format(epoch, config.epochs, epoch_loss))

        model_save_dir = os.path.join(config.ckptpath, "poseAE")
        os.makedirs(model_save_dir, exist_ok=True)
        state_dict = {'pose_embedder' : pose_embedder.state_dict(), 'pose_decoder' : pose_decoder.state_dict()}
        torch.save(state_dict, os.path.join(model_save_dir, "latest.pth"))

def pretrain_motion_trainer(train_dataloader, val_dataloader, model, config):
    pose_embedder = model[0].to('cuda:0')
    motion_embedder = model[1].to('cuda:0')
    motion_decoder = model[2].to('cuda:0')
    pose_decoder = model[3].to('cuda:0')

    state_dict = torch.load(os.path.join(config.ckptpath, "poseAE", "latest.pth"))
    pose_embedder.load_state_dict(state_dict['pose_embedder'])
    pose_decoder.load_state_dict(state_dict['pose_decoder'])

    optimizer = optim.AdamW(list(motion_embedder.parameters()) + list(motion_decoder.parameters()), lr = config.learning_rate)
    lr_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2, eta_min = config.learning_rate * 1e-3)
    criterion = nn.L1Loss()

    pose_embedder.eval()
    pose_decoder.eval()
    for epoch in range(config.epochs) :
        motion_embedder.train()
        motion_decoder.train()
        epoch_loss = 0.
        for train_step, batch in enumerate(train_dataloader) :
            optimizer.zero_grad()
            data = batch.float().to('cuda:0')

            embed = pose_embedder(data)
            embed = motion_embedder(embed)
            embed = motion_decoder(embed)
            recon = pose_decoder(embed)

            loss = criterion(recon, data)
            loss.backward()

            optimizer.step()
            #if train_step % 25 == 0 :
            #    print("[{}/{}][{}/{}] loss : {}".format(epoch, config.epochs, train_step, len(train_dataloader), loss.item()))

            epoch_loss += loss.item()
        epoch_loss /= len(train_dataloader)
        print("[{}/{}] train loss : {} lr : {}".format(epoch, config.epochs, epoch_loss, optimizer.param_groups[0]['lr']))
        lr_scheduler.step()

        pose_embedder.eval()
        pose_decoder.eval()
        epoch_loss = 0.
        for val_step, batch in enumerate(val_dataloader):
            data = batch.float().to('cuda:0')

            with torch.no_grad() :
                embed = pose_embedder(data)
                embed = motion_embedder(embed)
                embed = motion_decoder(embed)
                recon = pose_decoder(embed)

            loss = criterion(recon, data)
            #if val_step % 25 == 0:
            #    print("[{}/{}][{}/{}] loss : {}".format(epoch, config.epochs, val_step, len(val_dataloader), loss.item()))
            epoch_loss += loss.item()
        epoch_loss /= len(val_dataloader)
        print("[{}/{}] val loss : {}".format(epoch, config.epochs, epoch_loss))

        model_save_dir = os.path.join(config.ckptpath, "motionAE")
        os.makedirs(model_save_dir, exist_ok=True)
        state_dict = {'pose_embedder' : pose_embedder.state_dict(), 'pose_decoder' : pose_decoder.state_dict(),
                      'motion_embedder' : motion_embedder.state_dict(), 'motion_decoder' : motion_decoder.state_dict()}
        torch.save(state_dict, os.path.join(model_save_dir, "latest.pth"))
