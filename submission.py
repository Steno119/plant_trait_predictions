import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import pandas as pd
import lightning as L
import os
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from torchmetrics import R2Score
from torchvision.io import read_image
from os import sys

train_attribute_df = pd.read_csv(os.path.join("plant_data", "train.csv"))
train_attribute_df = train_attribute_df.iloc[:, 1:164]
attribute_means = torch.tensor(train_attribute_df.mean(axis=0), dtype=torch.float32)
attribute_stds = torch.tensor(train_attribute_df.std(axis=0), dtype=torch.float32)

target_df = pd.read_csv(os.path.join("plant_data", "train.csv"))
target_df = target_df.iloc[:, -6:]
target_maxs = torch.log10(torch.tensor(target_df.max(axis=0), dtype=torch.float32))
target_mins = torch.log10(torch.tensor(target_df.min(axis=0), dtype=torch.float32))

class PlantDataset(Dataset):
  def __init__(self, train, trait = "id", transform = None):
    self.transform = transform
    self.train = train
    self.trait = trait
    if (train):  
      self.img_labels = pd.read_csv(os.path.join("plant_data", "train.csv"))
      self.img_dir = os.path.join("plant_data", "train_images")
    else:
      self.img_labels = pd.read_csv(os.path.join("plant_data", "test.csv"))
      self.img_dir = os.path.join("plant_data", "test_images")

  def __len__(self):
    return len(self.img_labels)
  
  def __getitem__(self, index):
    img_path = os.path.join(self.img_dir, str(self.img_labels.iloc[index, 0]) + ".jpeg")
    image = read_image(img_path).to(torch.float32)
    if (self.transform):
      image = self.transform(image)

    if not self.trait == "id":
      label = torch.log10(torch.tensor(self.img_labels.iloc[index, -6:], dtype=torch.float32))
      label = (label - target_mins) / (target_maxs - target_mins)
    else:
      label = torch.tensor([self.img_labels.loc[index, self.trait]])

    attributes = torch.tensor(self.img_labels.iloc[index, 1:164].to_numpy(), dtype=torch.float32)
    attributes = (attributes - attribute_means) / attribute_stds
    
    return image, label, attributes

def de_normalize(value_batch):
  return 10 ** (value_batch * (target_maxs - target_mins) + target_mins)


class some_shitty_ml_model(L.LightningModule):
  def __init__(self, num_additional_attributes = 163):
    super(some_shitty_ml_model, self).__init__()
    self.model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
    self.model.classifier[-1] = nn.Linear(self.model.classifier[-1].in_features, out_features=24)

    self.attribute_model = nn.Sequential(
      nn.Linear(num_additional_attributes, 96),
      nn.ReLU(),
      nn.Dropout(0.3),
      nn.Linear(96, 96),
      nn.ReLU(),
      nn.Dropout(0.3),
      nn.Linear(96, 24),
    )

    self.fcn = nn.Sequential(
      nn.Linear(48, 48),
      nn.ReLU(),
      nn.Dropout(0.3),
      nn.Linear(48, 24),
      nn.ReLU(),
      nn.Dropout(0.3),
      nn.Linear(24, 6),
    )

    self.criterion = nn.MSELoss()
    self.training_r2 = [R2Score().to(device='cuda')] * 6
    self.test_r2 = [R2Score().to(device='cuda')] * 6
  
  def forward(self, input, attributes = torch.tensor([])):
    image_features = self.model(input)
    attribute_features = self.attribute_model(attributes)
    output = self.fcn(torch.cat((image_features, attribute_features), dim=1))
    return output

  def configure_optimizers(self):
    # optimizer = torch.optim.SGD(self.parameters(), nesterov=True, momentum=0.9)
    optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
    return [optimizer], [scheduler]
  
  def _calculate_loss(self, batch_iterator, mode, batch_idx = None):
    image_batch, label_batch, attribute_batch = batch_iterator
    prediction_batch = self(image_batch, attribute_batch)

    batch_loss = self.criterion(prediction_batch, label_batch)
    self.log(f'{mode} loss', batch_loss, on_step=True)
    
    # if (batch_idx == 0):
    #   print(prediction_batch)
    #   print(attribute_batch)
    #   print(label_batch)

    if (mode == "Training"):
      score = 0
      for idx, metric in enumerate(self.training_r2):
        score += metric(prediction_batch[:, idx], label_batch[:, idx])
      score /= len(self.training_r2)
      self.log(f'{mode} R2 score', score, on_step=False, on_epoch=True)
      # if (batch_idx == 0):
      #   print(f'First batch loss: {batch_loss}\nFirst batch R2 Score: {score}')
      #   print(preds)
      #   print(labels)
    else:
      score = 0
      for idx, metric in enumerate(self.test_r2):
        score += metric(prediction_batch[:, idx], label_batch[:, idx])
      score /= len(self.training_r2)
      # if (batch_idx == 0):
      #   print(preds)
      #   print(labels)
      self.log(f'{mode} R2 score', score, on_step=False, on_epoch=True)

    return batch_loss

  def training_step(self, batch_iterator, batch_idx):
    loss = self._calculate_loss(batch_iterator, mode="Training", batch_idx=batch_idx)
    return loss

  def test_step(self, batch_iterator, batch_idx):
    self._calculate_loss(batch_iterator, mode="Testing", batch_idx=batch_idx)
  
  def validation_step(self, batch_iterator, batch_idx):
    self._calculate_loss(batch_iterator, mode="Validation", batch_idx=batch_idx)
  
  def on_train_epoch_end(self):
    [score.reset() for score in self.training_r2]
  
  def on_test_epoch_end(self):
    [score.reset() for score in self.test_r2]
  
  def on_validation_epoch_end(self):
    [score.reset() for score in self.test_r2]
  
  def predict_step(self, batch_iterator, batch_idx):
    image_batch, id_batch, attribute_batch = batch_iterator
    outputs = self(image_batch, attribute_batch)
    return id_batch, outputs

DEV = False

if __name__ == '__main__':

  color_jitter = [transforms.ColorJitter(0.4, 0.4, 0.4, 0.4)]
  transformations = [
    # transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    transforms.RandomHorizontalFlip(p=0.4),
    transforms.RandomApply(transforms=color_jitter, p=0.4),
  ]
  transform = transforms.Compose(transformations)

  plant_dataset = PlantDataset(train=True, transform=transform, trait='')

  dataset_size = len(plant_dataset)
  test_size = dataset_size // 10 # 10% for testing
  train_size = dataset_size - test_size # 90% for training
  train_dataset, test_dataset = random_split(plant_dataset, [train_size, test_size])
  train_loader = DataLoader(dataset=train_dataset, batch_size=200, shuffle=True, num_workers=(0 if DEV else 2), persistent_workers=(not DEV))
  test_loader = DataLoader(dataset=test_dataset, batch_size=80, num_workers=(0 if DEV else 2), persistent_workers=(not DEV))

  trainer = L.Trainer(max_epochs=15, fast_dev_run=DEV)
  my_shitty_model = some_shitty_ml_model(num_additional_attributes=163)
  trainer.fit(model=my_shitty_model, train_dataloaders=train_loader, val_dataloaders=test_loader)
  trainer.test(model=my_shitty_model, dataloaders=test_loader)
  
  normalizer = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
  eval_data = PlantDataset(train=False, transform=normalizer)
  eval_loader = DataLoader(dataset=eval_data, batch_size=80)

  trainer = L.Trainer(default_root_dir='../', fast_dev_run=DEV)
  predictions = trainer.predict(model=my_shitty_model, dataloaders=eval_loader)
  ids = []
  preds = torch.tensor([])
  test = True
  for batch in predictions:
    id_batch, pred_batch = batch
    ids += torch.flatten(id_batch).tolist()
    preds = torch.cat((preds, de_normalize(pred_batch)))

  temp_dict = {
    'id': ids,
    'X4': preds[:, 0].tolist(),
    'X11': preds[:, 1].tolist(),
    'X18': preds[:, 2].tolist(),
    'X26': preds[:, 3].tolist(),
    'X50': preds[:, 4].tolist(),
    'X3112': preds[:, 5].tolist(),
  }
  frame = pd.DataFrame(temp_dict)
  pd.DataFrame.to_csv(frame, '20890769_Hu.csv', index=False)