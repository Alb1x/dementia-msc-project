import json
import torch
import numpy as np
import os

DATA_PATH = "ADReSS-IS2020/"

speaker_embeddings = {}

#Test data
for file in os.listdir(DATA_PATH + "test/second-modified"):
    filePath = os.path.join(DATA_PATH + "test/second-modified",file)
    if os.path.isfile(filePath) and file.endswith(".json"):
        speaker = file.split("-")[0]
        if speaker not in speaker_embeddings:
            speaker_embeddings[speaker] = []
        with open(filePath) as file:
            embedding = json.load(file)
            speaker_embeddings[speaker].append(embedding)
        
for speaker in speaker_embeddings.keys():
    embedding = np.array(speaker_embeddings[speaker]).mean(axis=0)
    print(speaker, embedding.shape)
    with open(DATA_PATH + "test/second-modified/mean/"+speaker+"-second-modified.json", "w") as f:
        json.dump(embedding.tolist(),f)

#Train data
for file in os.listdir(DATA_PATH + "train/cd/second-modified"):
    filePath = os.path.join(DATA_PATH + "train/cd/second-modified",file)
    if os.path.isfile(filePath) and file.endswith(".json"):
        speaker = file.split("-")[0]
        if speaker not in speaker_embeddings:
            speaker_embeddings[speaker] = []
        with open(filePath) as file:
            embedding = json.load(file)
            speaker_embeddings[speaker].append(embedding)
        
for speaker in speaker_embeddings.keys():
    embedding = np.array(speaker_embeddings[speaker]).mean(axis=0)
    print(speaker, embedding.shape)
    with open(DATA_PATH + "train/cd/second-modified/mean/"+speaker+"-second-modified.json", "w") as f:
        json.dump(embedding.tolist(),f)