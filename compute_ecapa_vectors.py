import numpy as np

from tqdm import tqdm
import json

import librosa
import os
import torch

from speechbrain.pretrained import EncoderClassifier
classifier = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", run_opts={"device":"cuda"})

DATA_PATH = "ADReSS-IS2020/"

def compute_embeddings():
    print("------ Computing embeddings for control samples ------")
    for file in tqdm(os.listdir(DATA_PATH + "train/cc")):
        filePath = os.path.join(DATA_PATH + "train/cc",file)
        if os.path.isfile(filePath) and file.endswith(".wav"):
            compute_embedding(filePath)

    print("------ Computing embeddings for dementia samples ------") 
    for file in tqdm(os.listdir(DATA_PATH + "train/cd")):
        filePath = os.path.join(DATA_PATH + "train/cd",file)
        if os.path.isfile(filePath) and file.endswith(".wav"):
            compute_embedding(filePath)

    print("------ Computing embeddings for test samples ------") 
    for file in tqdm(os.listdir(DATA_PATH + "test")):
        filePath = os.path.join(DATA_PATH + "test",file)
        if os.path.isfile(filePath) and file.endswith(".wav"):
            compute_embedding(filePath)
    
def compute_embedding(filePath):
    wav, _ = librosa.load(filePath, sr=16000)

    wav = torch.from_numpy(wav).float()
    wav = wav[80_000:160_000] #5sec to 10sec
    
    embedding = classifier.encode_batch(wav[None]).detach().cpu().numpy()[0][0]

    newfilePath = filePath.split("/")
    newfilePath.insert(-1, "ecapa-vectors")
    newfilePath = "/".join(newfilePath)

    with open(newfilePath.split(".")[0]+".json","w") as d_file:
        d_file.write(json.dumps(embedding.tolist()))

def main():
    compute_embeddings()
    return

if __name__ == "__main__":
    main()
