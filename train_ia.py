import torch

import torch.nn as nn

from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader

import torch.nn.functional as F

import getData


def get_default_device():
  if torch.cuda.is_available():
    return torch.device('cuda')
  else:
    return torch.device('cpu')


device = get_default_device()

def to_device(data, device):
  if isinstance(data, (list,tuple)):
    return [to_device(x, device) for x in data]
  return data.to(device, non_blocking=True)


class DeviceDataLoader():

  def __init__(self, dl, device):
    self.dl = dl
    self.device = device

  def __iter__(self):
    for b in self.dl:
      yield to_device(b, self.device)

  def __len__(self):
    return len(self.dl)



# Importing data
inputs = getData.inputs
targets = getData.targets


# Moving inputs and targets to gpu
# inputs = DeviceDataLoader(inputs, device)
# targets = DeviceDataLoader(targets, device)



# Dataloader
train_ds = TensorDataset(inputs,targets)
train_dl = DataLoader(train_ds, 10000, shuffle=True)


# Moving dataloader to GPU
train_dl = DeviceDataLoader(train_dl, device)



# Model
model = nn.Linear(5,1)

model.to(device)



# Loss mse
loss_fn = F.mse_loss


# optmizer
opt = torch.optim.SGD(model.parameters(), lr=1e-6)





# Training the model
def fit(epoch_nums, model, loss_fn, opt):
  for epoch in range(epoch_nums):
    for xy, yb in train_dl:

      # Predicting
      pred = model(xy)

      # loss
      loss = loss_fn(pred,yb)

      if(loss < 0.01):
        print("=====LOSS=====")
        print(loss)
        print(model.weight)
        print(model.bias)
        print("======================")

      # Calculating gradient
      loss.backward()

      # Subtracting weight and bias
      opt.step()

      # Reseting grads to 0
      opt.zero_grad()



# Train the model
fit(1000, model, loss_fn, opt)


# Best weight and bias
# with torch.no_grad():
#   model.weight = nn.Parameter(torch.tensor([[ 0.0582,  0.2522, -0.2788,  0.1546,  0.7714]], requires_grad=True))
#   model.bias = nn.Parameter(torch.tensor([-0.0720], requires_grad=True))


pred = model(torch.tensor([[109.639,110.689,112.353,114.348,115.148]]))


pred = torch.round(pred)
# targets = torch.round(targets)


print(pred * 100)




# =====LOSS=====
# tensor(0.0169, grad_fn=<MseLossBackward>)
# Parameter containing:
# tensor([[-0.1853,  0.3006, -0.1308,  0.4548,  0.5549]], requires_grad=True)
# Parameter containing:
# tensor([-0.0137], requires_grad=True)
# ======================



# =====LOSS=====
# tensor(0.0156, grad_fn=<MseLossBackward>)
# Parameter containing:
# tensor([[-0.1880,  0.2961, -0.1292,  0.4562,  0.5593]], requires_grad=True)
# Parameter containing:
# tensor([-0.0136], requires_grad=True)
# ======================







# =====LOSS=====
# tensor(0.0219, device='cuda:0', grad_fn=<MseLossBackward>)
# Parameter containing:
# tensor([[ 0.4079, -0.0405,  0.2446,  0.1401,  0.2434]], device='cuda:0',
#        requires_grad=True)
# Parameter containing:
# tensor([-0.1446], device='cuda:0', requires_grad=True)
# ======================





# =====LOSS=====
# tensor(0.0097, device='cuda:0', grad_fn=<MseLossBackward>)
# Parameter containing:
# tensor([[-0.0811,  0.4126,  0.2241,  0.1881,  0.2513]], device='cuda:0',
#        requires_grad=True)
# Parameter containing:
# tensor([-0.0634], device='cuda:0', requires_grad=True)
# ======================





# =====LOSS=====
# tensor(0.0093, device='cuda:0', grad_fn=<MseLossBackward>)
# Parameter containing:
# tensor([[-0.0659,  0.4286,  0.2203,  0.1771,  0.2349]], device='cuda:0',
#        requires_grad=True)
# Parameter containing:
# tensor([-0.0636], device='cuda:0', requires_grad=True)
# ======================





# =====LOSS=====
# tensor(0.0115, device='cuda:0', grad_fn=<MseLossBackward>)
# Parameter containing:
# tensor([[-0.0593,  0.4354,  0.2184,  0.1724,  0.2278]], device='cuda:0',
#        requires_grad=True)
# Parameter containing:
# tensor([-0.0637], device='cuda:0', requires_grad=True)
# ======================




# =====LOSS=====
# tensor(0.0093, device='cuda:0', grad_fn=<MseLossBackward>)
# Parameter containing:
# tensor([[0.0036, 0.4995, 0.1998, 0.1288, 0.1649]], device='cuda:0',
#        requires_grad=True)
# Parameter containing:
# tensor([-0.0643], device='cuda:0', requires_grad=True)
# ======================




# =====LOSS=====
# tensor(0.0083, device='cuda:0', grad_fn=<MseLossBackward>)
# Parameter containing:
# tensor([[0.0109, 0.5068, 0.1924, 0.1175, 0.1505]], device='cuda:0',
#        requires_grad=True)
# Parameter containing:
# tensor([-0.0644], device='cuda:0', requires_grad=True)
# ======================




# =====LOSS=====
# tensor(0.0073, device='cuda:0', grad_fn=<MseLossBackward>)
# Parameter containing:
# tensor([[-0.0858,  0.4076,  0.2253,  0.1914,  0.2565]], device='cuda:0',
#        requires_grad=True)
# Parameter containing:
# tensor([-0.0634], device='cuda:0', requires_grad=True)
# ======================



# =====LOSS=====
# tensor(0.0081, device='cuda:0', grad_fn=<MseLossBackward>)
# Parameter containing:
# tensor([[-0.0887,  0.4045,  0.2260,  0.1935,  0.2596]], device='cuda:0',
#        requires_grad=True)
# Parameter containing:
# tensor([-0.0633], device='cuda:0', requires_grad=True)
# ======================



# =====LOSS=====
# tensor(0.0098, device='cuda:0', grad_fn=<MseLossBackward>)
# Parameter containing:
# tensor([[-0.1085,  0.3830,  0.2304,  0.2067,  0.2831]], device='cuda:0',
#        requires_grad=True)
# Parameter containing:
# tensor([-0.0630], device='cuda:0', requires_grad=True)
# ======================



# =====LOSS=====
# tensor(0.0073, device='cuda:0', grad_fn=<MseLossBackward>)
# Parameter containing:
# tensor([[-0.1369,  0.3514,  0.2360,  0.2267,  0.3167]], device='cuda:0',
#        requires_grad=True)
# Parameter containing:
# tensor([-0.0626], device='cuda:0', requires_grad=True)
# ======================



# =====LOSS=====
# tensor(0.0071, device='cuda:0', grad_fn=<MseLossBackward>)
# Parameter containing:
# tensor([[-0.1469,  0.3398,  0.2371,  0.2336,  0.3299]], device='cuda:0',
#        requires_grad=True)
# Parameter containing:
# tensor([-0.0624], device='cuda:0', requires_grad=True)
# ======================



# =====LOSS=====
# tensor(0.0080, device='cuda:0', grad_fn=<MseLossBackward>)
# Parameter containing:
# tensor([[-0.1476,  0.3390,  0.2372,  0.2340,  0.3308]], device='cuda:0',
#        requires_grad=True)
# Parameter containing:
# tensor([-0.0624], device='cuda:0', requires_grad=True)
# ======================




# =====LOSS=====
# tensor(0.0063, device='cuda:0', grad_fn=<MseLossBackward>)
# Parameter containing:
# tensor([[-0.1522,  0.3333,  0.2378,  0.2375,  0.3370]], device='cuda:0',
#        requires_grad=True)
# Parameter containing:
# tensor([-0.0622], device='cuda:0', requires_grad=True)
# ======================



# =====LOSS=====
# tensor(0.0064, device='cuda:0', grad_fn=<MseLossBackward>)
# Parameter containing:
# tensor([[-0.1649,  0.3180,  0.2397,  0.2465,  0.3535]], device='cuda:0',
#        requires_grad=True)
# Parameter containing:
# tensor([-0.0620], device='cuda:0', requires_grad=True)
# ======================


# =====LOSS=====
# tensor(0.0071, device='cuda:0', grad_fn=<MseLossBackward>)
# Parameter containing:
# tensor([[-0.1678,  0.3145,  0.2411,  0.2495,  0.3581]], device='cuda:0',
#        requires_grad=True)
# Parameter containing:
# tensor([-0.0619], device='cuda:0', requires_grad=True)
# ======================




# =====LOSS=====
# tensor(0.0067, device='cuda:0', grad_fn=<MseLossBackward>)
# Parameter containing:
# tensor([[-0.1875,  0.2868,  0.2395,  0.2639,  0.3897]], device='cuda:0',
#        requires_grad=True)
# Parameter containing:
# tensor([-0.0612], device='cuda:0', requires_grad=True)
# ======================




# =====LOSS=====
# tensor(0.0056, device='cuda:0', grad_fn=<MseLossBackward>)
# Parameter containing:
# tensor([[-0.1901,  0.2828,  0.2391,  0.2660,  0.3946]], device='cuda:0',
#        requires_grad=True)
# Parameter containing:
# tensor([-0.0611], device='cuda:0', requires_grad=True)
# ======================



# =====LOSS=====
# tensor(0.0056, device='cuda:0', grad_fn=<MseLossBackward>)
# Parameter containing:
# tensor([[-0.1951,  0.2752,  0.2379,  0.2699,  0.4041]], device='cuda:0',
#        requires_grad=True)
# Parameter containing:
# tensor([-0.0608], device='cuda:0', requires_grad=True)
# ======================