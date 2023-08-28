import tqdm
import json
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.ensemble import RandomForestClassifier
from random import randint
import matplotlib.pyplot as plt

DATA_PATH = "ADReSS-IS2020/"
batch_size = 128
epochs = 100
learning_rate = 0.00013

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(nn.Linear(256,1),
                                   nn.Sigmoid()
                                   )
    def forward(self,x):
        return self.model(x)

def load_data():
    global train_dat, test_dat

    inputs = []
    labels = []
    for file in os.listdir(DATA_PATH + "train/cc"):
        filePath = os.path.join(DATA_PATH + "train/cc",file)
        if os.path.isfile(filePath) and file.endswith(".json"):
            with open(filePath) as file:
                embedding = json.load(file)
                inputs.append(embedding)
                labels.append(0)

    for file in os.listdir(DATA_PATH + "train/cd"):
        filePath = os.path.join(DATA_PATH + "train/cd",file)
        if os.path.isfile(filePath) and file.endswith(".json"):
            with open(filePath) as file:
                embedding = json.load(file)
                inputs.append(embedding)
                labels.append(1)

    train_dat = [(inputs[i],labels[i]) for i in range(len(inputs))]

    inputs = []
    labels = []
    labelmap = {}

    with open(DATA_PATH + "test_labels.txt") as file:
        for line in file.readlines():
            if line.startswith("ID"):
                continue
            line = line.split(";")
            labelmap[line[0].strip()] = int(line[3].strip())

    for file in os.listdir(DATA_PATH + "test"):
        filePath = os.path.join(DATA_PATH + "test",file)
        if os.path.isfile(filePath) and file.endswith(".json"):
            with open(filePath) as file:
                embedding = json.load(file)
                inputs.append(embedding)
                speaker = filePath.split("/")[-1].split("-")[0].strip()
                labels.append(labelmap[speaker])


    test_dat = [(inputs[i],labels[i]) for i in range(len(inputs))]

    print(f"Train len: {len(train_dat)}")
    print(f"Test len: {len(test_dat)}")

    
def train():
    device = torch.device("cuda"  if torch.cuda.is_available() else "cpu")
    loader_train = DataLoader(train_dat, batch_size, shuffle=True)
    loader_test = DataLoader(test_dat, batch_size, shuffle=False)

    model = Discriminator().to(device)
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Total number of parameters is: {}".format(params))
    print(model)
    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    LRscheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer,0.998)

    def loss_f(predictions, labels):
        # Calculate Binary Cross Entropy (BCE) loss
        criterion = nn.BCELoss()
        mse = nn.MSELoss()
        loss_bce = criterion(predictions, labels)
        loss_mse = mse(predictions,labels)
        
        # Regularization terms to reduce overfitting
        # loss_l1 = l1_loss(predicted_prob,labels)

        # Combine BCE loss and regularization loss
        loss =  2 * loss_bce + loss_mse

        return loss
    
    train_losses = []
    test_losses = []
    test_accuracy = []
    test_precision = []
    test_recall = []
    epoch_lr = []
    for epoch in range(epochs):  
        model.train()
        running_loss = 0
        with tqdm.tqdm(loader_train, unit="batch") as tepoch: 
            for batch_idx, data in enumerate(tepoch):   
                
                input = torch.stack(data[0]).squeeze().transpose(0,1).to(torch.float32)
                labels = data[1].unsqueeze(1).to(torch.float32)

                optimizer.zero_grad()

                predicted_prob = model(input)

                loss = loss_f(predicted_prob, labels)
                loss.backward()

                running_loss += loss.item()
                
                optimizer.step()

        train_losses.append(running_loss/len(loader_train))
        epoch_lr.append(LRscheduler.get_last_lr())
        LRscheduler.step()
        print("Epoch "+str(epoch)+":")
        print("Epoch loss: "+ str(running_loss/len(loader_train)))

        #Validation
        model.eval()
        running_loss = 0
        accuracy = 0
        precision = 0
        recall = 0
        with tqdm.tqdm(loader_test, unit="batch") as tepoch: 
            for batch_idx, data in enumerate(tepoch):   
                input = torch.stack(data[0]).squeeze().transpose(0,1).to(torch.float32)
                labels = data[1].unsqueeze(1).to(torch.float32).round()

                predicted_prob = model(input)
                predicted_label = predicted_prob.round()

                correct_predictions  = (predicted_label == labels.round()).sum().item()
                accuracy += correct_predictions / labels.size(0)
                
                true_positives = torch.sum((predicted_label == 1) & (labels == 1)).item()
                false_positives = torch.sum((predicted_label == 1) & (labels == 0)).item()
                true_negatives = torch.sum((predicted_label == 0) & (labels == 0)).item()
                false_negatives = torch.sum((predicted_label == 0) & (labels == 1)).item()

                precision += true_positives / (true_positives + false_positives) if true_positives > 0 else 0
                recall += true_positives / (true_positives + false_negatives) if true_positives > 0 else 0

                loss = loss_f(predicted_prob, labels)
                running_loss += loss.item()

        
        test_losses.append(running_loss/len(loader_test))
        test_accuracy.append(accuracy/len(loader_test)*100)
        test_precision.append(precision/len(loader_test))
        test_recall.append(recall/len(loader_test))
    
    torch.jit.script(model).save(f"output/discriminator_{epochs}.pth")

    return (train_losses, test_losses, test_accuracy,test_precision,test_recall,epoch_lr)

def train_rf():
    rf_classifier = RandomForestClassifier()

    x_train, y_train = [], []
    for sample,label in train_dat:
        x_train.append(sample)
        y_train.append(label)
    # Train the Random Forest classifier
    rf_classifier.fit(x_train, y_train)

    x_test, y_test = [], []
    for sample,label in test_dat:
        x_test.append(sample)
        y_test.append(label)
    # Make predictions on the test set
    print(f"RF Accuracy: {rf_classifier.score(x_test,y_test)*100}%")

def save_losses(stats):
    train_losses, test_losses, test_accuracy,test_precision,test_recall,train_lr = stats
    plt.plot(train_losses, label='Training Loss')
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

    plt.plot(test_accuracy)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy %')
    plt.title('Testing accuracy per Epoch')
    plt.savefig('output/test_accuracy_plot.png')
    plt.close()   

    f = 2 * (np.array(test_precision) * np.array(test_recall)) / (np.array(test_precision) + np.array(test_recall))

    plt.plot(f, label="Test F metric")
    plt.plot(test_precision, label='Test precision')
    plt.plot(test_recall, label='Test recall')
    plt.xlabel('Epoch')
    plt.ylabel('Performance Metric Value')
    plt.legend()
    plt.title('Test F, precision and recall per Epoch')
    plt.savefig('output/F_recall_precision_plot.png')
    plt.close()
    

def test():
    model = torch.jit.load(f"output/discriminator_{epochs}.pth")
    index = randint(0,len(test_dat)-1)
    data = test_dat[index][0]
    label = test_dat[index][1]

    predicted_label = model(torch.Tensor(data))

    print("True label: " + str(label))
    print("Predicted label: "+str(predicted_label.float()))

def main():
    load_data()
    stats = train()
    train_rf()
    save_losses(stats)
    test()

if __name__ == "__main__":
    main()