# Msc Project 2023

The ADReSS dataset must be inside the root folder of the repo with the following structure :
```
ADReSS-IS2020
	├── test
		└── test_samples.wav
	├── train
		├── cd
			└── test_samples.wav
		└── cc
			└── test_samples.wav
	└── test_labels.txt
```
---

### Python requirements
```bash
python -m venv venv

source venv/bin/activate

pip install -r requirements.txt
```
---
### Compute embeddings
```
├── compute_ecapa_vectors.py
├── compute_x_vectors.py
└── compute_sse.py
```
---
### Use the mean-shift approach
```bash
python mean-shift-approach.py
```
### Train  models
#### Train discriminator on [simple-speaker-embedding](https://github.com/RF5/simple-speaker-embedding)
```
└── discriminator_embeddings_train.py
```
#### Train the embeddings modifier with [simple-speaker-embedding](https://github.com/RF5/simple-speaker-embedding) and modify the embeddings 
```
└── feature_noise_train.py
```
#### Retrain the discriminator on modified embeddings
```bash
python discriminator_modified-embeddings_train.py {modification_number}
```