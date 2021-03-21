import torch

import torch.nn as nn

from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader

import torch.nn.functional as F


# Model
model = nn.Linear(5,1)


# Best weight and bias
with torch.no_grad():
  model.weight = nn.Parameter(torch.tensor([[-0.1951,  0.2752,  0.2379,  0.2699,  0.4041]], requires_grad=True))
  model.bias = nn.Parameter(torch.tensor([-0.0608], requires_grad=True))


day1 = float(input('day 1: '))
day2 = float(input('day 2: '))
day3 = float(input('day 3: '))
day4 = float(input('day 4: '))
day5 = float(input('day 5: '))

pred = model(torch.tensor([[day5,day4,day3,day2,day1]]))
pred = torch.round(pred)


print("Day 6:", pred.item())