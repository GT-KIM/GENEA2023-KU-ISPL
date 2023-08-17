import os
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset
from utils import text_util

from models.pretrain import *
from transformers import AutoTokenizer, AutoModel
import torch.nn.functional as F
from tqdm import tqdm

class Diffusiondataset(Dataset) :
    def __init__(self, config, pre_config, phase='test') :
        self.config = config
        self.pre_config = pre_config
        self.datapath = config.picklepath

        self.load_pose_models()
        self.load_text_models()

        if phase == 'test' :
            self.data = self.parse_tst_data()

    def __len__(self) :
        return len(self.data)

    def load_pose_models(self) :
        pose_embedder, motion_embedder, motion_decoder, pose_decoder = PoseEmbedder(self.pre_config), MotionEmbedder(self.pre_config), MotionDecoder(self.pre_config), PoseDecoder(self.pre_config)
        state_dict = torch.load(os.path.join(self.config.ckptpath, "motionAE", "latest.pth"))
        pose_embedder.load_state_dict(state_dict['pose_embedder'])
        motion_embedder.load_state_dict(state_dict['motion_embedder'])
        motion_decoder.load_state_dict(state_dict['motion_decoder'])
        pose_decoder.load_state_dict(state_dict['pose_decoder'])

        pose_embedder = pose_embedder.to("cuda:1").eval()
        motion_embedder = motion_embedder.to("cuda:1").eval()
        motion_decoder = motion_decoder.to("cuda:1").eval()
        pose_decoder = pose_decoder.to("cuda:1").eval()

        self.pose_model = [pose_embedder, motion_embedder, motion_decoder, pose_decoder]

    def load_text_models(self):
        self.tokenizer = AutoTokenizer.from_pretrained('intfloat/e5-large')
        self.text_model = AutoModel.from_pretrained('intfloat/e5-large').to("cuda:1")

    def inference_single_pose_model(self, pose) :
        pose = torch.tensor(pose).float().to("cuda:1").unsqueeze(0)
        with torch.no_grad() :
            pose = self.pose_model[0](pose).squeeze(0)
        return pose.cpu().numpy()

    def inference_pose_models(self, pose) :
        pose = torch.tensor(pose).float().to("cuda:1").unsqueeze(0)
        with torch.no_grad() :
            pose = self.pose_model[0](pose)
            pose = self.pose_model[1](pose).squeeze(0)
        return pose.cpu().numpy()

    def recon_pose_models(self, pose) :
        pose = torch.tensor(pose).float().to("cuda:1")
        with torch.no_grad() :
            pose = self.pose_model[2](pose)
            pose = self.pose_model[3](pose)
        return pose.cpu().numpy()


    def inference_text_models(self, text) :
        batch_dict = self.tokenizer(text, max_length=512, padding=True, truncation=True, return_tensors='pt')
        batch_dict = batch_dict.to("cuda:1")

        with torch.no_grad() :
            outputs = self.text_model(**batch_dict)
        embeddings = self.average_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
        embeddings = F.normalize(embeddings, p=2, dim=1)
        return embeddings.squeeze(0).cpu().numpy()

    def parse_data(self) :
        genea_path = os.path.join(self.datapath, "tst_main-agent_v0")
        genea_files = os.listdir(genea_path)
        data = list()
        for idx, genea_file in tqdm(enumerate(genea_files)) :
            genea_filepath = os.path.join(genea_path, genea_file)
            with open(genea_filepath, 'rb') as f :
                g_data = dict(pickle.load(f))

            name = g_data['name']
            mfcc = g_data['mfcc']
            mel = g_data['melspectrogram']
            prosody = g_data['prosody']
            text = g_data['text']

            clip_length = mfcc.shape[0] / 30 # second

            for start_time in np.arange(0, clip_length - self.config.frame_length / 30 + self.config.traindata_interval, self.config.traindata_interval) :
                end_time = start_time + self.config.frame_length / 30
                start_frame = int(start_time * self.config.fps)
                end_frame = int(end_time * self.config.fps)

                audio_data = np.concatenate([mfcc[start_frame:end_frame], mel[start_frame:end_frame], prosody[start_frame:end_frame]], axis=-1)

                word_seq = text_util.get_words_in_time_range(word_list=text, start_time=start_time, end_time=end_time)
                word_seq = ' '.join(word_seq)
                text_data = self.inference_text_models(word_seq)

                data.append({'audio' : audio_data, 'text' : text_data})

            if self.config.quick_test and idx == 4 :
                break
        return data

    def parse_tst_data(self) :
        genea_path = os.path.join(self.datapath, "tst_main-agent_v0")
        genea_files = os.listdir(genea_path)
        data = list()
        for idx, genea_file in tqdm(enumerate(genea_files)):
            genea_filepath = os.path.join(genea_path, genea_file)
            with open(genea_filepath, 'rb') as f:
                g_data = dict(pickle.load(f))

            name = g_data['name']
            mfcc = g_data['mfcc']
            mel = g_data['melspectrogram']
            prosody = g_data['prosody']
            text = g_data['text']

            clip_length = mfcc.shape[0] / 30  # second

            samples = {'name' : name, 'clip_length' : clip_length, 'data' : list()}
            for start_time in np.arange(0, clip_length - self.config.frame_length / 30 + self.config.traindata_interval, self.config.traindata_interval):
                end_time = start_time + self.config.frame_length / 30
                start_frame = int(start_time * self.config.fps)
                end_frame = int(end_time * self.config.fps)

                audio_data = np.concatenate(
                    [mfcc[start_frame:end_frame], mel[start_frame:end_frame], prosody[start_frame:end_frame]], axis=-1)
                if len(audio_data) != self.config.frame_length:
                    audio_data = np.concatenate((audio_data, np.zeros((self.config.frame_length - audio_data.shape[0], audio_data.shape[1]))), axis=0)

                word_seq = text_util.get_words_in_time_range(word_list=text, start_time=start_time, end_time=end_time)
                word_seq = ' '.join(word_seq)
                text_data = self.inference_text_models(word_seq)

                samples['data'].append({'audio': audio_data, 'text': text_data, 'start' : start_time, 'end' : end_time})

            data.append(samples)

            if self.config.quick_test and idx == 4:
                break
        return data
    def average_pool(self, last_hidden_states, attention_mask):
        last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
        return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

    def __getitem__(self, i) :
        return self.data[i]

def collate_fn(batch) :
    audio = [x['audio'] for x in batch]
    text = [x['text'] for x in batch]

    audio = torch.tensor(np.array(audio)).float()
    text = torch.tensor(np.array(text)).float()

    return {'audio' : audio, 'text' : text}

def collate_fn_no_prepose(batch) :
    audio = [x['audio'] for x in batch]
    text = [x['text'] for x in batch]

    audio = torch.tensor(np.array(audio)).float()
    text = torch.tensor(np.array(text)).float()

    return {'audio' : audio, 'text' : text}

def tst_collate_fn(batch) :
    name = batch[0]['name']
    clip_length =batch[0]['clip_length']
    data = batch[0]['data']
    return name, clip_length, data

if __name__ == "__main__" :
    import sys

    [sys.path.append(i) for i in ['.', '..']]

    from config.parse_config import parse_config
    from torch.utils.data import DataLoader

    config = parse_config('diffusion')
    pre_config = parse_config('pretrain')
    dataset = Diffusiondataset(config, pre_config, phase='test')
    data = DataLoader(dataset, batch_size=config.batch_size, shuffle=True, collate_fn=collate_fn)
