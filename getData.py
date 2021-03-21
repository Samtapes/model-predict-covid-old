import pandas
import numpy as np
import torch


torch.cuda.set_device(0)
device = torch.device('cuda:0')


data = pandas.read_csv('./covid02/data/caso.csv', usecols=['confirmed'])



inputs = []
targets = []



def adjustDataset(data):
  a = 0 
  for i in range(len(data.values) - 1):
    a+=1
    
    if(a == 6):
      targets.append(round(data.values.item(i) / 1000, 4))
      a = 0

    else:
      inputs.append(round(data.values.item(i) / 1000, 4))



adjustDataset(data)




inputs = np.array(inputs)
inputs = np.resize(inputs, inputs.size - 4)
inputs = inputs.reshape(-1,5)

inputs = torch.from_numpy(inputs)
inputs = inputs.float()




targets = np.array(targets)
targets = np.resize(targets, targets.size - 0)
targets = targets.reshape(-1,1)

targets = torch.from_numpy(targets)
targets = targets.float()