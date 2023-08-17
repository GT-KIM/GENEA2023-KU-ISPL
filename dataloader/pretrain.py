import os
import pickle
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset
from utils import text_util
class Pretraindataset(Dataset) :
    def __init__(self, config, phase='train') :
        self.config = config
        self.datapath = config.picklepath
        if phase == 'train' :
            self.data = self.parse_trn_data()
        elif phase == 'val' :
            self.data = self.parse_val_data()
        else :
            raise ValueError

    def __len__(self) :
        return len(self.data)

    def parse_trn_data(self) :
        genea_path = os.path.join(self.datapath, "trn_main-agent_v0")
        genea_files = os.listdir(genea_path)
        data = list()
        mean_pose = list()
        for genea_file in tqdm(genea_files) :
            genea_filepath = os.path.join(genea_path, genea_file)
            with open(genea_filepath, 'rb') as f :
                g_data = dict(pickle.load(f))

            full_motion = g_data['expmap_full']
            clip_length = full_motion.shape[0] / 30 # second

            for start_time in np.arange(0, clip_length - self.config.frame_length / 30, self.config.traindata_interval) :
                end_time = start_time + self.config.frame_length / 30
                start_frame = round(start_time * self.config.fps)
                end_frame = round(end_time * self.config.fps)

                cropped_motion = full_motion[start_frame:end_frame]
                data.append(cropped_motion)
        if self.config.use_beat :
            beat_path = os.path.join(self.datapath, "beat")
            beat_files = os.listdir(beat_path)
            data = list()
            for beat_file in tqdm(beat_files) :
                beat_filepath = os.path.join(beat_path, beat_file)
                with open(beat_filepath, 'rb') as f :
                    b_data = dict(pickle.load(f))

                full_motion = b_data['expmap_full']
                clip_length = full_motion.shape[0] / 30 # second

                for start_time in np.arange(0, clip_length - self.config.frame_length / 30, self.config.traindata_interval) :
                    end_time = start_time + self.config.frame_length / 30
                    start_frame = round(start_time * self.config.fps)
                    end_frame = round(end_time * self.config.fps)

                    cropped_motion = full_motion[start_frame:end_frame]
                    data.append(cropped_motion)

        return data

    def parse_val_data(self) :
        genea_path = os.path.join(self.datapath, "val_main-agent_v0")
        genea_files = os.listdir(genea_path)
        data = list()
        for genea_file in tqdm(genea_files) :
            genea_filepath = os.path.join(genea_path, genea_file)
            with open(genea_filepath, 'rb') as f :
                g_data = dict(pickle.load(f))

            full_motion = g_data['expmap_full']
            clip_length = full_motion.shape[0] / 30 # second

            for start_time in np.arange(0, clip_length - self.config.frame_length / 30, self.config.valdata_interval) :
                end_time = start_time + self.config.frame_length / 30
                start_frame = round(start_time * self.config.fps)
                end_frame = round(end_time * self.config.fps)

                cropped_motion = full_motion[start_frame:end_frame]
                data.append(cropped_motion)
        return data

    def __getitem__(self, i) :
        return self.data[i]