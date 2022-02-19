import pandas as pd
import numpy as np
import torch
import torch.utils.data as Data
from torch.utils.data import DataLoader, Dataset
from torch import cuda
import torch.optim.lr_scheduler as sched

class CustomScheduler(torch.optim.lr_scheduler._LRScheduler):
  def __init__(self, optimizer):
    super(CustomScheduler, self).__init__(optimizer)
    self.lr = 0.002

  def get_lr(self):
    if self.last_epoch == 0:
      return [group['lr'] for group in self.optimizer.param_groups]
    return [self.lr for group in self.optimizer.param_groups]

  def update_lr(self, lr):
    self.lr = lr

class DatasetUnknownFunc(Dataset):
  def __init__(self, min_val = 1000, max_val = 2000, balance_data = True):
    xy = np.mgrid[min_val:max_val, min_val:max_val].reshape(2, -1)
    df = pd.DataFrame({'X' : xy[0], 'Y' : xy[1]})

    df['Z'] = df['X'] + df['Y']

    if balance_data == True:
      df1 = df[df.X == df.Y]
      df2 = df[df.X != df.Y].sample(len(df1))
      self.data = pd.concat([df1, df2], ignore_index=True, sort=False).reset_index(drop = True)
    else:
      self.data = df
      
  def __len__(self):
    return len(self.data)
  
  def __getitem__(self, index):
    return {
      'X': torch.tensor(self.data.X[index], dtype=torch.float),
      'Y': torch.tensor(self.data.Y[index], dtype=torch.float),
      'Z': torch.tensor(self.data.Z[index], dtype=torch.float),
      'features': torch.tensor([self.data.X[index], self.data.Y[index]], dtype=torch.float)
    } 

class Net(torch.nn.Module):
  def __init__(self, n_feature, n_hidden, n_output):
    super(Net, self).__init__()

    self.hidden = torch.nn.Linear(n_feature, n_hidden)
    self.relu = torch.nn.LeakyReLU()
    self.hidden1 = torch.nn.Linear(n_hidden, 100)
    self.relu1 = torch.nn.LeakyReLU(),
    self.predict = torch.nn.Linear(n_hidden, n_output)

  def forward(self, xy):
    xy = self.hidden(xy)
    xy = self.relu(xy)
    z = self.predict(xy)

    return z


def rmspe_func(y_pred, y_true):
  error = 0
  for val1, val2 in zip(y_pred.cpu().numpy(), y_true.cpu().numpy()):
      error += (val2 - val1)*(val2 - val1)
  return error

class TaskWrapper():
  def __init__(self, debug = False, epochs = 500, lr=0.0002):
    BATCH_SIZE = 30
    self.num_epochs = epochs
    min_val = 1000
    max_val = 2000
    n_hidden = 100
    self.epoch = 0
    self.debug = debug

    self.device = 'cpu'
    if torch.cuda.is_available():
        self.device="cuda"

    train_dataset = DatasetUnknownFunc(min_val = min_val, max_val = max_val, balance_data = True)
    self.training_loader = Data.DataLoader(
      dataset=train_dataset, 
      batch_size=BATCH_SIZE, 
      shuffle=True, num_workers=2)

    self.model = Net(n_feature=2, n_hidden=n_hidden, n_output = 1)
    self.model.to(self.device)
    self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)
    self.scheduler = CustomScheduler(self.optimizer)
    self.loss_function = torch.nn.MSELoss()

    self.model.train()

  def train_epoch(self):
    rmse = 0
    nb_tr_examples = 0

    rmses = []
    for step ,data in enumerate(self.training_loader, 0):
      xy = data['features'].to(self.device, dtype = torch.float)
      Z = data['Z'].to(self.device, dtype = torch.float)
      outputs = self.model(xy).to(self.device, dtype = torch.float)

      loss = self.loss_function(outputs.view(-1), Z.view(-1))
      rmses.append(loss.item())

      self.optimizer.zero_grad()
      loss.backward()
      self.optimizer.step()

    self.epoch += 1
    
    rmse, std = np.mean(rmses), np.std(rmses)
    
    if self.debug:
      lr = self.scheduler.get_last_lr()
      print(f'   epoch {self.epoch}, rmse {rmse}, std {std}, lr = {lr}')
      
    return rmse, std

  def done(self):
    return self.epoch == self.num_epochs

  def set_scheduler_lr(self, lr):
    self.scheduler.update_lr(lr)
    self.scheduler.step()
    
  