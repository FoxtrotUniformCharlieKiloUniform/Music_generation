import soundfile as sf      #https://github.com/bastibe/python-soundfile
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.optim.lr_scheduler as lr_scheduler
import math

def scinot(num):
    #print scientific notation
    if num == 0:
        print("0.0e+0")
    else:
        exponent = int(math.log10(abs(num)))
        coefficient = num / 10**exponent
        #print(f"{coefficient}")
    return f"{coefficient:.1e}, e: {exponent}"

def make_divisible(number, divisor):
  remainder = number % divisor
  if remainder != 0:
    number += divisor - remainder
  return number

#Indecees of start and end values of a array. 
startIdx = 10000
endIdx = 50000     

#define column names
col0 = "0"
col1 = "1"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device is ", device)
torch.manual_seed(123)

#get song from .wav
data, samplerate = sf.read(r"C:\Users\Matt\Documents\Pytorch_ML\Generative\lofi-orchestra-162306.wav")
a_before_splitting = pd.DataFrame(data, columns = [col0, col1])

a_before_splitting = a_before_splitting.mean(axis = 1).to_frame()

loc = r"C:\Users\Matt\Documents\Pytorch_ML\Generative\songtocsv_singlechannel.csv"
a = a_before_splitting.iloc[startIdx:endIdx,:]
a.insert(0, 'Time Step', range(len(a)))
print(a)


#dataset / loader objects for batching
class TimeSeriesDataset(Dataset):
    def __init__(self, dataframe):
        self.data = dataframe
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        
        col = self.data.iloc[idx,:]
        #print(f"col 0 is {col[0]}, col 1 is {col[1]}")#, col 2 is {col[2]}")
        label = torch.tensor(col.iloc[0]).float().to(device)
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
batch_size = 16
dataLoader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

input_size = 1
hidden_size = 12
num_layers = 2
endConstant = 4     #Output of final LSTM layer should match size of input to attention layer

num_heads = hidden_size
embed_dim = num_heads * endConstant

class song(nn.Module):
    def __init__(self, input_size, hidden_size, batch_size, num_layers):
        super(song, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_size = batch_size

        self.conv1 = nn.Conv1d(in_channels=input_size, out_channels=hidden_size, kernel_size=2, padding='same')
        self.conv2 = nn.Conv1d(in_channels=hidden_size, out_channels=hidden_size, kernel_size=4, padding='same')
        self.conv3 = nn.Conv1d(in_channels=hidden_size, out_channels=hidden_size, kernel_size=16, padding='same')

        self.avgpool = nn.AdaptiveAvgPool1d(1)

        #self.LSTM1 = nn.LSTM(input_size=hidden_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.LSTM1 = nn.LSTM(input_size=batch_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.LSTM2 = nn.LSTM(input_size=hidden_size, hidden_size=hidden_size * 2, num_layers=num_layers, batch_first=True)
        self.LSTM3 = nn.LSTM(input_size=hidden_size*2, hidden_size=hidden_size * endConstant, num_layers=num_layers, batch_first=True)

        #output in theory is (batch, seq, feature)
        self.attentionLayer = nn.MultiheadAttention(embed_dim, num_heads)
        self.attention_to_midln = nn.Linear(hidden_size * 4, 100)

        self.midLin = nn.Linear(hidden_size * 4, 100)
        self.endNote1 = nn.Linear(100, 1)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, self.hidden_size).float().to(device)
        c0 = torch.zeros(self.num_layers, self.hidden_size).float().to(device)
        h1 = torch.zeros(self.num_layers, self.hidden_size*2).float().to(device)
        c1 = torch.zeros(self.num_layers, self.hidden_size*2).float().to(device)
        h2 = torch.zeros(self.num_layers, self.hidden_size*4).float().to(device)
        c2 = torch.zeros(self.num_layers, self.hidden_size*4).float().to(device)
      
        x = x.permute(1, 0)

        #print(x.shape)

        x1 = F.relu(self.conv1(x))
        x2 = F.relu(self.conv2(x1))
        x3 = F.relu(self.conv3(x2))
        x3 = x3.permute(1,0)
        x4 = self.avgpool(x3)
        x4 = x4.permute(1,0)
        

        # Residual connection
        x_residual = x + x4
        #x_residual = x_residual.permute(1, 0)

        x_residual, _ = self.LSTM1(x_residual, (h0, c0))
        x_residual, _ = self.LSTM2(x_residual, (h1, c1))
        x_residual, _ = self.LSTM3(x_residual, (h2, c2))

        #because we are employing self attention in this case, we shrimply pass in the x_residual vector 3X. If we were going based on other vectors, we'd pass in the other vectors for key and  value (obviously)
        x_attn,_ = self.attentionLayer(x_residual,x_residual,x_residual)

        #print(f"DEBUG: x_LSTM3 output is {x_residual.shape} x_atteion shape is {x_attn.shape}")
        x_residual = self.midLin(x_attn)

        x_residual = self.endNote1(x_residual)
        outputs = x_residual
        
        return outputs

# Instantiate the model and move it to the GPU
model = song(input_size, hidden_size, batch_size, num_layers).to(device)

# Model Hyperparameters
epochs = 1
clip_value = 0.1 # Gradient clipping value
learning_rate = 0.001       #best run: 0.6

# Define loss function and optimizer    
criterion = nn.HuberLoss(delta = 1)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=1)


for epoch in range(epochs):
    model.train()
    total_loss = 0
    label1_Correct = 0
    total_samples = 0

    for times, label1 in dataLoader:
        times = times.unsqueeze(-1).to(torch.float32).to(device)
        label1s = label1.unsqueeze(-1).to(torch.float32).to(device)
        #print(times)

        optimizer.zero_grad()

        outputs1 = model(times)
        #print(f"DEBUG: outputs shape is {outputs1.shape}, inputs shape is {times.shape}")
        loss = criterion(outputs1, label1s)
        loss.backward()
        
        # Gradient clipping
        clip_value = clip_value
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)
        optimizer.step()
        total_loss += loss.item()

    before_lr = optimizer.param_groups[0]["lr"]
    scheduler.step(total_loss)
    after_lr = optimizer.param_groups[0]["lr"]

    # Print gradient norms
    total_norm = 0
    total = 0
    #print(model.parameters())

    for p in model.parameters():
        if p.grad is not None:
            #print(p)
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
        print(f'Error for column 1 is = {accuracy_column1:.2f}% scientific notation:',scinot(accuracy_column1),')')
        print(f"Learning rate before update: {before_lr}, after update {after_lr}")
        print("\n")
        


print("_______________TESTING_____________")


last = a.index[-1] - a.index[0] + 1
test_len = 5008         #user set param

#print("Last value of a is", last)       #debug
#dataPoints = np.arange(last, last + test_len) # - changed in next line, delete if works
dataPoints = a_before_splitting.iloc[last:last + test_len]

#print("Datapoints head is ", dataPoints.head())                #debug
datasetTest = TimeSeriesDataset(dataPoints)
dataLoader = DataLoader(datasetTest, batch_size=batch_size, shuffle=False)

outputs1_list = []
i=0

# Define the testing method
def test_model(model, dataLoader, criterion, device):
    model.eval()  # Set the model to evaluation mode
    total_loss = 0.0
    total_samples = 0
    label1_Correct = 0

    with torch.no_grad():  # Disable gradient calculation
        for i, (times, label1s) in enumerate(dataLoader):
            #print("Batch Index: ",i)
            # Ensure the input data is in the correct shape
            times = times.unsqueeze(-1).to(torch.float32).to(device)
            label1s = label1s.unsqueeze(-1).to(torch.float32).to(device)

            # Forward pass
            outputs1 = model(times)

            # Compute the loss
            #loss = criterion(outputs1, label1s)
            #total_loss += loss.item()

            # Calculate the accuracy for column 1
            total_samples += label1s.size(0)
            label1_Correct += ((torch.pow((outputs1 - label1s), 2)).sum()).item()

        # Compute average loss and accuracy
        accuracy_column1 = 100 * (1 - (label1_Correct / total_samples))

    print(f'Accuracy for column 1: {accuracy_column1:.2f}%')

test_model(model, dataLoader, criterion, device)





print("_____________CREATING__________________")

make_music = True
save_music = True
seconds = 3/16
inp = seconds * samplerate
inp = make_divisible(inp, batch_size)
print(inp)

print("\n")
print("\n")

if make_music:
    print("Make music is true! ")
    last = a.index[-1] - a.index[0] + 1
    print("Last value of a is", last,"inp is",inp)       #debug
    dataPoints = np.arange(last, last + inp)
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

    outputs1 = [item for item in outputs1 for _ in range(batch_size)]
    outputs1 = torch.tensor(outputs1)
    print(dataPoints.__len__(),"   ", outputs1.__len__())
    outputsDF = pd.DataFrame({"Time Step": dataPoints.flatten(), "0": outputs1.flatten()})
    #print("The head of outputsDF is ", outputsDF.head(), "and the size is", outputsDF.size)
    
    print("_____________STYLIZING DATA FOR OUTPUT READIBILITY OF OUR SOUNDFILE METHOD__________________")

    
    outputsDF.columns = a.columns
    a = pd.concat([a, outputsDF.astype(a.dtypes)], ignore_index=True)
    a = a.drop(a.columns[0],axis = 1)
    a = a.astype("float32")

    a.to_csv(r"C:\Users\Matt\Documents\Pytorch_ML\Generative\music_example_with_outputs.csv")

    if save_music:
        print("Save music is true!")
        sf.write(r"C:\Users\Matt\Documents\Pytorch_ML\Generative\music_example_with_outputs.wav", a.values, samplerate)
    else:
        print("Music is not saved")
else:
    print("Music not made or saved.")
