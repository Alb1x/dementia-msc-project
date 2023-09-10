import tqdm
import json
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from random import randint
import matplotlib.pyplot as plt

DATA_PATH = "ADReSS-IS2020/"
batch_size = 256
epochs = 30
learning_rate = 0.00005

discriminator = torch.jit.load("output/discriminator_100.pth")
discriminator.eval()

class FeatureNoise(nn.Module):
    def __init__(self):
        super(FeatureNoise, self).__init__()
        self.model = nn.Sequential(nn.Linear(256,256),
                                   nn.Linear(256,256)
                                   )
    def forward(self,x):
        return self.model(x)


def load_embeddings():
    train_data = []
    for file in os.listdir(DATA_PATH + "train/cd"):
        filePath = os.path.join(DATA_PATH + "train/cd",file)
        if os.path.isfile(filePath) and file.endswith(".json"):
            with open(filePath) as file:
                embedding = json.load(file)
                train_data.append(embedding)

    labelmap = {}
    with open(DATA_PATH + "test_labels.txt") as file:
        for line in file.readlines():
            if line.startswith("ID"):
                continue
            line = line.split(";")
            labelmap[line[0].strip()] = int(line[3].strip())

    test_data = []
    for file in os.listdir(DATA_PATH + "test"):
        filePath = os.path.join(DATA_PATH + "test",file)
        if os.path.isfile(filePath) and file.endswith(".json"):
            with open(filePath) as file:
                embedding = json.load(file)  
                speaker = filePath.split("/")[3].split("-")[0].strip()
                if labelmap[speaker]:
                    test_data.append(embedding)

    return torch.Tensor(train_data), torch.Tensor(test_data)


def average_discriminator(embeddings):
    return discriminator(embeddings).mean()

def train(train_data, test_data):
    device = torch.device("cuda"  if torch.cuda.is_available() else "cpu")
    loader_train = DataLoader(train_data, batch_size, shuffle=True)
    loader_test = DataLoader(test_data, batch_size, shuffle=False)

    model = FeatureNoise().to(device)
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Total number of parameters is: {}".format(params))
    print(model)
    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    LRscheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer,0.995)

    def discriminator_loss(predictions):
        # Calculate Binary Cross Entropy (BCE) loss
        labels = torch.ones(predictions.size())
        criterion = nn.BCELoss()
        mse = nn.MSELoss()
        loss_bce = criterion(predictions, labels)
        loss_mse = mse(predictions,labels)
        

        # Regularization terms to reduce overfitting
        # loss_l1 = l1_loss(predicted_prob,labels)

        # Combine BCE loss and regularization loss
        loss =  2 * loss_bce + loss_mse

        return loss

    def loss_f(input, output):
        discriminator_average = average_discriminator(output) 
        loss_discriminator= 0.225 * discriminator_loss(discriminator(output)) #We want to remove the dementia feature
        loss_distance = 1 * (output - input).pow(2).sum(1).sqrt().mean() #We don't want to move too far away to keep the user identity
        l1_loss = 0
        for param in model.parameters():
            l1_loss += torch.sum(torch.abs(param))

        l1_loss *= 0.0001

        loss = loss_distance - loss_discriminator + l1_loss
        return loss, loss_discriminator, discriminator_average, loss_distance

    train_losses = []
    train_losses_disc = []
    train_losses_dist = []
    test_losses = []
    test_discriminator_average = []
    epoch_lr = []
    for epoch in range(epochs):  
        model.train()
        running_loss = 0
        running_loss_disc = 0
        running_loss_dist = 0
        with tqdm.tqdm(loader_train, unit="batch") as tepoch: 
            for batch_idx, input in enumerate(tepoch):   
                optimizer.zero_grad()

                output = model(input)

                loss, loss_discriminator, discriminator_average, loss_distance = loss_f(input, output)
                loss.backward()

                running_loss += loss.item()
                running_loss_disc += loss_discriminator.item()
                running_loss_dist += loss_distance.item()
                
                optimizer.step()

        train_losses.append(running_loss/len(loader_train))
        train_losses_disc.append(running_loss_disc/len(loader_train))
        train_losses_dist.append(running_loss_dist/len(loader_train))
        epoch_lr.append(LRscheduler.get_last_lr())
        LRscheduler.step()
        print("Epoch "+str(epoch)+":")
        print("Epoch loss: "+ str(running_loss/len(loader_train)))

        #Validation
        model.eval()
        running_loss = 0
        running_average = 0
        with tqdm.tqdm(loader_test, unit="batch") as tepoch: 
            for batch_idx, input in enumerate(tepoch):   
                output = model(input)

                loss, loss_discriminator, discriminator_average, loss_distance = loss_f(input, output)

                running_average += discriminator_average.item()
                running_loss += loss.item()

        
        test_losses.append(running_loss/len(loader_test))
        test_discriminator_average.append(running_average/len(loader_test))
    
    torch.jit.script(model).save(f"output/feature_noise_{epochs}.pth")

    return (train_losses, train_losses_disc, train_losses_dist, test_losses, test_discriminator_average,epoch_lr)

def save_figs(stats):
    train_losses, train_losses_disc, train_losses_dist, test_losses, test_discriminator_average, train_lr = stats

    plt.plot(train_losses, label='Training Loss')
    plt.plot(train_losses_disc, ":", label='Training Loss Discriminator')
    plt.plot(train_losses_dist, ":",label='Training Loss Distance')
    plt.plot(test_losses, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training & Test loss per Epoch')
    plt.savefig('output/epoch_loss_plot.png')
    plt.close()

    plt.plot(train_lr)
    plt.xlabel('Epoch')
    plt.ylabel('LR')
    plt.title('Training learning rate per Epoch')
    plt.savefig('output/epoch_lr_plot.png')
    plt.close()    

    plt.plot(test_discriminator_average)
    plt.xlabel('Epoch')
    plt.ylabel('Discriminator average')
    plt.title('Discriminator prob. per Epoch')
    plt.savefig('output/test_discriminator_plot.png')
    plt.close()   

def modify_embeddings():
    model = torch.jit.load(f"output/embeddings_modifier/feature_noise_{epochs}.pth") 
    model.eval()

    for file in os.listdir(DATA_PATH + "train/cd"):
        filePath = os.path.join(DATA_PATH + "train/cd",file)
        if os.path.isfile(filePath) and file.endswith(".json"):
            with open(filePath) as file:
                embedding = torch.Tensor(json.load(file))
                modified_embedding = model(embedding).tolist()
                with open(DATA_PATH + "train/cd/second-modified/"+filePath.split("/")[-1].split(".")[0]+"-second-modified"+".json", "w") as f:
                    json.dump(modified_embedding,f)

    for file in os.listdir(DATA_PATH + "test"):
        filePath = os.path.join(DATA_PATH + "test",file)
        if os.path.isfile(filePath) and file.endswith(".json"):
            with open(filePath) as file:
                embedding = torch.Tensor(json.load(file))
                modified_embedding = model(embedding).tolist()
                with open(DATA_PATH + "test/second-modified/"+filePath.split("/")[-1].split(".")[0]+"-second-modified"+".json", "w") as f:
                    json.dump(modified_embedding,f)

    
                
def main():
    train_data, test_data = load_embeddings()
    stats = train(train_data, test_data)
    save_figs(stats)
    modify_embeddings()
    return

if __name__ == "__main__":
    main()
