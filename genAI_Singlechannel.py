import soundfile as sf      #https://github.com/bastibe/python-soundfile
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.optim.lr_scheduler as lr_scheduler

#Indecees of start and end values of a array. 
startIdx = 10000
endIdx = 20000     

#define column names
col0 = "0"
col1 = "1"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device is ", device)

#get song from .wav
data, samplerate = sf.read(r"C:\Users\Matt\Documents\Pytorch_ML\Generative\music_example.wav")
a_before_splitting = pd.DataFrame(data, columns = [col0, col1])

a = a_before_splitting.mean(axis = 0)
print(a.head)

loc = r"C:\Users\Matt\Documents\Pytorch_ML\Generative\songtocsv_singlechannel.csv"
a = a_before_splitting.iloc[startIdx:endIdx]
a.insert(0, 'Time Step', range(len(a)))
a.to_csv(loc)



#dataset / loader objects for batching
class TimeSeriesDataset(Dataset):
    def __init__(self, dataframe):
        self.data = dataframe
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        col = self.data.iloc[idx,:]
        #print(f"col 0 is {col[0]}, col 1 is {col[1]}")#, col 2 is {col[2]}")
        label = torch.tensor(col[0]).float().to(device)
        return idx, label
    
    
    #yes this is stupid
class TimeSeriesDatasetCreate(Dataset):
    def __init__(self, dataframe):
        self.data = dataframe
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return idx
    
# Create the dataset
dataset = TimeSeriesDataset(a)

# Create the DataLoader
batch_size = 64
dataLoader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

input_size = 1
hidden_size = 800
num_layers = 2

class song(nn.Module):
    def __init__(self, input_size, hidden_size, batch_size, num_layers):
        #variables
        super(song, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_size = batch_size

        #layers
        self.convSmol = nn.Conv1d(in_channels = input_size, out_channels = hidden_size,kernel_size = 2, padding = "same")  
        self.convLarge = nn.Conv1d(in_channels = hidden_size, out_channels = hidden_size,kernel_size = 16, padding = "same")  
        self.LSTM = nn.LSTM(input_size=hidden_size, hidden_size=hidden_size, num_layers=num_layers,batch_first = True)
        self.Long_LSTM = nn.LSTM(input_size=hidden_size, hidden_size=hidden_size*2, num_layers=num_layers,batch_first = True)
        self.avgpool = nn.AvgPool1d(kernel_size = 2, stride = 2)
        self.bn1d = nn.BatchNorm1d(hidden_size)
        self.bn1dDouble = nn.BatchNorm1d(hidden_size*2)
        
        self.endNote1 = nn.Linear(hidden_size, 1)

        self.float()

    def forward(self, x):

        h0 = torch.zeros(self.num_layers, self.hidden_size).float().to(device)
        c0 = torch.zeros(self.num_layers, self.hidden_size).float().to(device)
        h1 = torch.zeros(self.num_layers, self.hidden_size*2).float().to(device)
        c1 = torch.zeros(self.num_layers, self.hidden_size*2).float().to(device)
        

        #TODO - eddit architecture to be something more robust
        #TODO - add nonlinearities as required

       # print(x.shape)

        x = x.permute(1,0)
        x1 = F.leaky_relu(self.convSmol(x))
        x1 = F.leaky_relu(self.convLarge(x1))
        x1 = x1.permute(1,0)
        x1, _ = self.LSTM(x1, (h0, c0)) 
        x1 = self.bn1d(x1)
        x1, _ = self.Long_LSTM(x1, (h1, c1)) 
        x1 = self.bn1dDouble(x1)
        x1 = self.avgpool(x1)
                     

        x1 = self.endNote1(x1)
        outputs = x1
        return outputs

# Instantiate the model and move it to the GPU
model = song(input_size, hidden_size, batch_size, num_layers).to(device)

# Model Hyperparameters
epochs = 20
clip_value = 0.7 # Gradient clipping value
learning_rate = 0.6

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.6, patience=0)


for epoch in range(epochs):
    model.train()
    total_loss = 0
    label1_Correct = 0
    total_samples = 0

    for times, label1 in dataLoader:
        times = times.unsqueeze(-1).to(torch.float32).to(device)
        label1s = label1.unsqueeze(-1).to(torch.float32).to(device)

        optimizer.zero_grad()
        outputs1 = model(times)
        loss = criterion(outputs1, label1s)
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)
        optimizer.step()
        total_loss +=loss.item()

    before_lr = optimizer.param_groups[0]["lr"]
    scheduler.step(loss)
    after_lr = optimizer.param_groups[0]["lr"]

    # Print gradient norms
    total_norm = 0
    total = 0
    for p in model.parameters():
        param_norm = p.grad.detach().data.norm(2)
        total_norm += param_norm.item() ** 2
    
    total_norm = total_norm ** 0.5

    total += label1s.size(0)
    label1_Correct += ((torch.pow((outputs1 - label1s), 2)).sum()) / total
    total_samples += label1s.size(0)

    # Calculate the accuracy for this epoch
    accuracy_column1 = 100 * label1_Correct / total_samples

    if (epoch + 1) % 1 == 0:  # Print every epoch
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}, Gradient Norm: {total_norm:.4f}")
        print(f'Error for column 1 is = {accuracy_column1:.2f}%')
        print(f"Before update: {before_lr}, after update {after_lr}")
        print("\n")
        


print("_______________TESTING_____________")


last = a.index[-1] - a.index[0] + 1
test_len = 5000         #user set param

#print("Last value of a is", last)       #debug
#dataPoints = np.arange(last, last + test_len) # - changed in next line, delete if works
dataPoints = a_before_splitting.iloc[last:last + test_len]

#print("Datapoints head is ", dataPoints.head())                #debug
datasetTest = TimeSeriesDataset(dataPoints)
dataLoader = DataLoader(datasetTest, batch_size=batch_size, shuffle=False)

outputs1_list = []

with torch.no_grad():
    for times, label1s in dataLoader:
        times = times.unsqueeze(-1).to(torch.float32).to(device)
        label1s = label1s.unsqueeze(-1).to(torch.float32).to(device)

        outputs1 = model(times)

    total += label1s.size(0)
    #print(f"DEBUG: size of outputs1 is {outputs1.shape}, size of label1s is {label1s.shape}")
    label1_Correct += ((torch.pow((outputs1 - label1s), 2)).sum()) / total
    total_samples += label1s.size(0)

    # Calculate the accuracy for this epoch
    accuracy_column1 = 100 * label1_Correct / total_samples

    print("\n")
    print(f'Error for column 1 is = {accuracy_column1:.2f}%')
    print("\n")
    







print("_____________CREATING__________________")

make_music = False
save_music = False
seconds = 0.1

print("\n")
print("\n")

if make_music:
    print("Make music is true! ")
    last = a.index[-1] - a.index[0] + 1
    #print("Last value of a is", last)       #debug
    dataPoints = np.arange(last, last + seconds * samplerate)
    dataPoints = pd.DataFrame(dataPoints)

    #print("Datapoints head is ", dataPoints.head())                #debug
    datasetCreate = TimeSeriesDatasetCreate(dataPoints)
    dataLoader = DataLoader(datasetCreate, batch_size=batch_size, shuffle=False)
    
    outputs1_list = []
    with torch.no_grad():
        for times in dataLoader:
            times = times.unsqueeze(-1).to(torch.float32).to(device)
            outputs1 = model(times)
            
            outputs1_list.append(outputs1.cpu().numpy())

    #model outputs are stored as column vectors outputs1 and outputs2
    outputs1 = np.concatenate(outputs1_list, axis=0)

    dataPoints = dataPoints.to_numpy()       #convert "dataPoints" to usable format

    outputsDF = pd.DataFrame({"Time Step": dataPoints.flatten(), col0: outputs1.flatten()})
    #print("The head of outputsDF is ", outputsDF.head(), "and the size is", outputsDF.size)
    a = pd.concat([a, outputsDF], ignore_index=True)

    a.to_csv(r"C:\Users\Matt\Documents\Pytorch_ML\Generative\music_example_with_outputs.csv")

    if save_music:
        print("Save music is true!")
        sf.write(r"C:\Users\Matt\Documents\Pytorch_ML\Generative\music_example_with_outputs.wav", a.values, samplerate)
    else:
        print("Music is not saved")
else:
    print("Music not made or saved.")
