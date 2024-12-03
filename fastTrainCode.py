from fastTrain import FastTrain
from dataset import TrainData, TestData
print("importing libraries")
import pytorch_lightning as pl
import torch
import torch.nn as nn
from pytorch_lightning import Trainer
from torch.utils.data import Dataset
from sentence_transformers import SentenceTransformer
import json 
import tqdm
import pickle

device = "cuda" if torch.cuda.is_available() else "cpu"
# read fastTrain pickle
print("reading config file")
with open('fastTrain.pickle', 'rb') as handle:
    fastTrain = pickle.load(handle)

embedding_models = fastTrain.models

glosses = fastTrain.trainData.defs
outs = fastTrain.trainData.words_embds

print(f"length of defs {len(glosses)} and outputs {len(outs)}")

def model_maker(sen_trans=None, input_size=None):
  text_model = SentenceTransformer(sen_trans,
                                  device=device)

  glosses_emb = text_model.encode(glosses, batch_size = fastTrain.batch_size, device=device)

  loss = torch.nn.MSELoss()
  inputs = torch.tensor(glosses_emb)
  finals = torch.tensor(outs)

  class CustomDataset(Dataset):
      def __init__(self, input_vectors, output_vectors):
          self.input_vectors = inputs
          self.output_vectors = finals

      def __len__(self):
          return len(self.input_vectors)

      def __getitem__(self, idx):
          input_vector = self.input_vectors[idx]
          output_vector = self.output_vectors[idx]
          return input_vector, output_vector


  class MLP(pl.LightningModule):
      def __init__(self, input_size, output_size, model_specs):
          super(MLP, self).__init__()

          self.layers = nn.ModuleList()  # Dynamically define layers
          self.num_layers = model_specs["layers"]
          self.layer_size = model_specs["layer_size"]
          self.auto_decrease = model_specs.get("auto_decrease", False)

          # Initialize hidden layers
          current_input_size = input_size
          current_layer_size = self.layer_size

          for i in range(self.num_layers):
              if self.auto_decrease:
                  # Calculate the input and output size for each layer based on the pattern
                  next_layer_size = current_layer_size * 2 ** (self.num_layers - i - 1)
                  self.layers.append(nn.Linear(current_input_size, next_layer_size))
                  current_input_size = next_layer_size
                  current_layer_size = next_layer_size // 2
              else:
                  self.layers.append(nn.Linear(current_input_size, current_layer_size))
                  current_input_size = current_layer_size

          # Output layer
          self.out = nn.Linear(current_input_size, output_size)

          # Activation function (based on fastTrain)
          if fastTrain.activiation == "gelu":
              self.activation = nn.GELU()
          elif fastTrain.activiation == "relu":
              self.activation = nn.ReLU()
          else:
              self.activation = nn.GELU()

          # Dropout
          if fastTrain.dropout:
            self.dropout = nn.Dropout(p=fastTrain.dropout)
          else: 
            self.dropout = None

      def forward(self, x):
          for layer in self.layers:
              x = layer(x)
              x = self.activation(x)
              if self.dropout:
                x = self.dropout(x)
          return self.out(x)

      def configure_optimizers(self):
          if fastTrain.optimizer == "adam":
              optim = torch.optim.AdamW(self.parameters(), lr=fastTrain.lr)
          elif fastTrain.optimizer == "sgd":
              optim = torch.optim.SGD(self.parameters(), lr=fastTrain.lr)
          return optim

      def training_step(self, batch, batch_idx):
          x, y = batch
          output = self(x)
          loss = self.loss_fn(output, y)
          return {"loss": loss}

      def train_dataloader(self):
          inpt = inputs
          out = finals
          dataset = CustomDataset(inpt, out)
          train_loader = torch.utils.data.DataLoader(
              dataset=dataset, batch_size=fastTrain.batch_size, shuffle=True
          )
          return train_loader

      def loss_fn(self, output, target):
          # Define loss function (e.g., MSE or CrossEntropy)
          return nn.functional.mse_loss(output, target)


  trainer = Trainer(max_epochs=fastTrain.epochs, devices=[0])
  model = MLP(input_size=input_size, output_size=fastTrain.output_size, model_specs=fastTrain.model_specs)


  return trainer, model

def evaluate_model(model, sen_trans):
  print("eval the test file")
  device = "cuda" if torch.cuda.is_available() else "cpu"

  dev_texts = fastTrain.testData.defs
  dev_ids = fastTrain.testData.ids
  # eval the test data
  text_model = SentenceTransformer(sen_trans,
                                device=device)
  
  
  glosses_emb = torch.tensor(text_model.encode(dev_texts, batch_size = fastTrain.embd_batch_size,
                                  device=device))
  
  model.eval()
  with torch.no_grad():
    outputs = model(glosses_emb)

  preds = []
  for o in range(len(dev_ids)):
    preds.append({"id":dev_ids[o], fastTrain.given_model:outputs[o].cpu().tolist()})

  # Save predictions to a JSON file
  model_name = sen_trans.split("/")[1]
  output_file = f"test_{model_name}_{fastTrain.given_model}_preds.json"
  with open(output_file, "w") as f:
      json.dump(preds, f)


for sen_tran_name, input_size in embedding_models.items():
  
  torch.cuda.empty_cache()
  print(f"start training {sen_tran_name=}")
  trainer, model = model_maker(sen_tran_name, input_size)
  trainer.fit(model)
  evaluate_model(model, sen_tran_name)
