from datetime import datetime
import os
import json
import librosa
import torch

import numpy as np 

DATA_PATH = "ADReSS-IS2020/"
model = torch.hub.load('RF5/simple-speaker-embedding', 'convgru_embedder')
model.eval()

dementia_embeddings = []
control_embeddings = []

for filename in os.listdir(DATA_PATH + "train/cd"):
    f = os.path.join(DATA_PATH + "train/cd", filename)
    # checking if it is a wav file
    if os.path.isfile(f) and f.endswith(".wav"):
        wav, _ = librosa.load(f, sr=16000)
        wav = torch.from_numpy(wav).float()
        embedding = model(wav[None])
        dementia_embeddings.append(embedding.detach().cpu().numpy()[0])

for filename in os.listdir(DATA_PATH + "train/cc"):
    f = os.path.join(DATA_PATH + "train/cc", filename)
    # checking if it is a wav file
    if os.path.isfile(f) and f.endswith(".wav"):
        wav, _ = librosa.load(f, sr=16000)
        wav = torch.from_numpy(wav).float()
        embedding = model(wav[None])
        control_embeddings.append(embedding.detach().cpu().numpy()[0])


dementia_mean = np.mean(dementia_embeddings, axis=0)
dementia_std = np.std(dementia_embeddings,  axis=0)

control_mean = np.mean(control_embeddings, axis=0)
control_std = np.std(control_embeddings, axis=0)

direction_to_go = control_mean - dementia_mean
std_ratio = control_std / dementia_std

for file in os.listdir(DATA_PATH + "train/cd"):
    filePath = os.path.join(DATA_PATH + "train/cd",file)
    if os.path.isfile(filePath) and file.endswith(".json"):
        with open(filePath) as file:
            embedding = np.array(json.load(file))
            embedding = (embedding + direction_to_go) * np.sqrt(std_ratio)
            with open(DATA_PATH + "train/cd/first-modified/"+filePath.split("/")[-1].split(".")[0]+"-first-modified"+".json", "w") as f:
                json.dump(embedding.tolist(),f)

for file in os.listdir(DATA_PATH + "test"):
    filePath = os.path.join(DATA_PATH + "test",file)
    if os.path.isfile(filePath) and file.endswith(".json"):
        with open(filePath) as file:
            embedding = np.array(json.load(file))
            embedding = (embedding + direction_to_go) * np.sqrt(std_ratio)
            with open(DATA_PATH + "test/first-modified/"+filePath.split("/")[-1].split(".")[0]+"-first-modified"+".json", "w") as f:
                json.dump(embedding.tolist(),f)