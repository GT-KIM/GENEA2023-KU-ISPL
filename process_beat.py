import numpy as np
import h5py
import librosa
import os
import io
import string
from tqdm import tqdm
from sklearn.pipeline import Pipeline
import joblib as jl
from pymo.parsers import BVHParser
from pymo.preprocessing import *
from pymo.viz_tools import *
import librosa
import librosa.display
import soundfile as sf
import argparse
import pickle
import re

from utils.tool import *

def load_textgrid(textgrid) :
    with open(textgrid, 'r', encoding='utf-8') as f:
        data = f.readlines()
    # print data #Use this to view how the code would look like after the program has opened the files
    parsed_data = list()
    flag = 0
    for idx, lines in enumerate(data) :  # informations needed begin on the 9th lines
        line = re.sub('\n', '', lines)  # as there's \n at the end of every sentence.
        line = re.sub('\t', '', line)  # as there's \n at the end of every sentence.
        line = re.sub('^ *', '', line)
        if 'item [1]:' in lines :
            flag = 1
        elif 'item [2]:' in lines :
            flag = 0
        if flag :
            if 'intervals [' in lines :
                sample = list()
                for i in range(idx+1, idx+4, 1) :
                    linepair = data[i].replace('\n', '').replace('\t', '').split('=')
                    if len(linepair) == 2:
                        if linepair[0] == 'xmin ':
                            xmin = linepair[1][1:]
                            sample.append(xmin)
                        if linepair[0] == 'xmax ':
                            xmax = linepair[1][1:]
                            sample.append(xmax)
                        if linepair[0] == 'text ':
                            text = linepair[1][1:]
                            if text.strip().startswith('"') and text.strip().endswith('"'):
                                text = text[1:-1]
                                sample.append(text)
                if len(sample) == 3:
                    parsed_data.append(sample)
    return parsed_data

def load_metadata(metadata, participant):
    assert participant in ("main-agent", "interloctr"), "`participant` must be either 'main-agent' or 'interloctr'"

    metadict_byfname = {}
    metadict_byindex = {}
    speaker_ids = []
    finger_info = []
    with open(metadata, "r") as f:
        # NOTE: The first line contains the csv header so we skip it
        for i, line in enumerate(f.readlines()[1:]):
            (
                fname,
                main_speaker_id,
                main_has_finger,
                ilocutor_speaker_id,
                ilocutor_has_finger,
            ) = line.strip().split(",")

            if participant == "main-agent":
                has_finger = (main_has_finger == "finger_incl")
                speaker_id = int(main_speaker_id) - 1
            else:
                has_finger = (ilocutor_has_finger == "finger_incl")
                speaker_id = int(ilocutor_speaker_id) - 1

            finger_info.append(has_finger)
            speaker_ids.append(speaker_id)

            metadict_byindex[i] = has_finger, speaker_id
            metadict_byfname[fname + f"_{participant}"] = has_finger, speaker_id

    speaker_ids = np.array(speaker_ids)
    finger_info = np.array(finger_info)
    num_speakers = np.unique(speaker_ids).shape[0]
    # assert num_speakers == spks.max(), "Error speaker info!"
    # print("Number of speakers: ", num_speakers)
    # print("Has Finger Ratio:", np.mean(finger_info))

    return num_speakers, metadict_byfname, metadict_byindex


def load_bvh_jointselector(bvhfile):
    parser = BVHParser()
    parsed_data = parser.parse(bvhfile)

    mexp_full = Pipeline([
        ('jtsel', JointSelector(["b_root", "b_spine0", "b_spine1", "b_spine2", "b_spine3", "b_neck0", "b_head", "b_r_shoulder",
                                 "b_r_arm", "b_r_arm_twist",
                                 "b_r_forearm", "b_r_wrist_twist",
                                 "b_r_wrist", "b_l_shoulder",
                                 "b_l_arm", "b_l_arm_twist",
                                 "b_l_forearm", "b_l_wrist_twist",
                                 "b_l_wrist", "b_r_upleg", "b_r_leg",
                                 "b_r_foot", "b_l_upleg", "b_l_leg", "b_l_foot"], include_root=True)),
        ('param', MocapParameterizer('expmap')),
        ('cnst', ConstantsRemover_withroot()),
        ('np', Numpyfier()),
    ])
    fullexpdata = mexp_full.fit_transform([parsed_data])[0]

    mexp_upperbody = Pipeline([
        ('jtsel', JointSelector(["b_root", "b_spine0", "b_spine1", "b_spine2", "b_spine3", "b_neck0", "b_head", "b_r_shoulder",
                                 "b_r_arm",
                                 "b_r_arm_twist",
                                 "b_r_forearm",
                                 "b_r_wrist_twist",
                                 "b_r_wrist", "b_l_shoulder",
                                 "b_l_arm",
                                 "b_l_arm_twist",
                                 "b_l_forearm",
                                 "b_l_wrist_twist",
                                 "b_l_wrist"
                                 ], include_root=False)),
        ('param', MocapParameterizer('expmap')),
        ('cnst', ConstantsRemover_()),
        ('np', Numpyfier()),
    ])
    upperexpdata = mexp_upperbody.fit_transform([parsed_data])[0]

    return fullexpdata, upperexpdata


def load_audio(audiofile):
    audio, sr = librosa.load(audiofile, sr=None)
    return audio, sr

def load_tsv(tsvfile):
    sentences = []
    sentence = []
    offset = 0
    with open(tsvfile, "r") as f:
        for i, line in enumerate(f.readlines()):
            start, end, raw_word = line.strip().split("\t")
            start = float(start)
            end = float(end)

            if start - offset > .05 and i > 0:
                if sentence[-1][1] - sentence[0][0] > .2: # if duration is long enough
                    sentences.append(sentence)
                sentence = [[start, end, raw_word]]
            else:
                sentence.append([start, end, raw_word])

            offset = end

    durations = [s[-1][1] - s[0][0] for s in sentences]
    sentence_lengths = [len(s) for s in sentences]
    return sentences, durations, sentence_lengths


def load_tsv_unclipped(tsvfile):
    sentence = []
    with open(tsvfile, "r") as f:
        for i, line in enumerate(f.readlines()):
            line = line.strip().split("\t")
            if len(line) == 3:
                start, end, raw_word = line
                start = float(start)
                end = float(end)
                sentence.append([start, end, raw_word])
    return sentence


def find_timestamp_from_timings(timestamp, timings):
    output = None
    for i, (start, end) in enumerate(timings):
        if start <= timestamp < end:
            output = i
            break
    return output

def prepare_pickle(beat_path, beat_original_path, picklefile) :

    sample_list = os.listdir(beat_path)
    for sample_num in sample_list :
        bvh_path = os.path.join(beat_path, sample_num)
        feat_path = os.path.join(beat_original_path, sample_num)
        bvh_files = os.listdir(bvh_path)
        for bvh_file in bvh_files :
            name = bvh_file[:-10]
            bvh_filename = os.path.join(bvh_path, bvh_file)
            wav_filename = os.path.join(feat_path, name + ".wav")
            text_filename = os.path.join(feat_path, name + ".TextGrid")

            g_data = dict()

            audio, sr = load_audio(wav_filename)
            prosody = extract_prosodic_features(wav_filename)
            mfcc = calculate_mfcc(audio, sr)
            melspec = calculate_spectrogram(audio, sr)

            full, upper = load_bvh_jointselector(bvh_filename)

            crop_length = min(mfcc.shape[0], prosody.shape[0], melspec.shape[0], full.shape[0], upper.shape[0])
            prosody = prosody[:crop_length]
            mfcc = mfcc[:crop_length]
            melspec = melspec[:crop_length]
            full = full[:crop_length]
            upper = upper[:crop_length]

            g_data["has_finger"] = -1
            g_data["speaker_id"] = -1

            # g_audiodata.create_dataset("raw_audio", data=(audio*32768).astype(np.int16), dtype=np.int16)
            g_data["mfcc"] = mfcc
            g_data["melspectrogram"] = melspec
            g_data["prosody"] = prosody
            g_data["expmap_full"] = full
            g_data["expmap_upper"] = upper

            # Process the txt
            # Align txt with audio
            sentence = load_textgrid(text_filename)

            g_data["text"] = sentence

            os.makedirs(picklefile, exist_ok=True)
            with open(os.path.join(picklefile, name + '.pickle'), 'wb') as f:
                pickle.dump(g_data, f)
            print(name)
    print()
    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset_path", type=str, default="GENEA2023//genea2023_dataset")
    parser.add_argument("--beat_path", dest='beat_path', type=str, default="GENEA2023//beat_genea")
    parser.add_argument("--beat_original_path", dest='beat_original_path', type=str, default="BEATS//original//beat_english_v0.2.1")
    args = parser.parse_args()

    save_path = os.path.join(args.dataset_path, "pickle")
    os.makedirs(save_path, exist_ok=True)

    prepare_pickle(args.beat_path, args.beat_original_path, f"{save_path}//beat")