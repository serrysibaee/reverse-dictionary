# build and train models (using pytorch lighting)
# Training code 
import pydantic
from datetime import datetime
import pytorch_lightning as pl
import torch
import torch.nn as nn
from pytorch_lightning import Trainer
from torch.utils.data import Dataset
import json
from embeddings import TrainEmbeddings, TestEmbeddings
# import Train and Test embeddings 

class RDTrainer(pydantic.BaseModel):
  train_embds: TrainEmbeddings
  test_embds:  TestEmbeddings
  lr:float
  epochs:int
  trained_model_name:str
  batch_size:int = 128
  _model:pl.LightningModule = None
  ready_model:pl.LightningModule = None
  now:str = datetime.now().strftime("%Y-%m-%d|%H:%M:%S")
  
  class Config:
    arbitrary_types_allowed = True

  def model_maker(self,):
    lr = self.lr
    batch_size = self.batch_size
    input_size = len(self.train_embds.embeds[0])
    glosses_emb = self.train_embds.embeds 

    loss = torch.nn.MSELoss()
    inputs = torch.tensor(glosses_emb)
    outputs = torch.tensor(self.train_embds.outputs)

    class CustomDataset(Dataset):
        def __init__(self, input_vectors, output_vectors):
            self.input_vectors = inputs
            self.output_vectors = outputs

        def __len__(self):
            return len(self.input_vectors)

        def __getitem__(self, idx):
            input_vector = self.input_vectors[idx]
            output_vector = self.output_vectors[idx]
            return input_vector, output_vector


    class MLP(pl.LightningModule):
        def __init__(self, input_size, h1,h2,h3,h4, output_size):
            super(MLP, self).__init__()
            self.fc1 = nn.Linear(input_size, h1)
            self.fc2 = nn.Linear(h1,h2)
            self.fc3 = nn.Linear(h2, h3)
            self.fc4 = nn.Linear(h3,h4)
            self.relu = nn.GELU()
            self.dropout = nn.Dropout(p=0.2)

            self.out = nn.Linear(h4,output_size)

        def forward(self, x):
            x = self.fc1(x)
            x = self.relu(x)
            x = self.dropout(x)
            x = self.fc2(x)
            x = self.relu(x)
            x = self.dropout(x)
            x = self.fc3(x)
            x = self.relu(x)
            x = self.dropout(x)
            x = self.fc4(x)
            return self.out(x)

        def configure_optimizers(self):
          optim = torch.optim.AdamW(self.parameters(), lr=lr)
          return optim

        def training_step(self,batch,batch_idx):
          x,y = batch

          output = self(x)

          lost = loss(output,y)

          return {"loss":lost}

        def train_dataloader(self):
          inpt = inputs
          out  = outputs
          dataset = CustomDataset(inpt,out)
          train_loader = torch.utils.data.DataLoader(dataset=dataset,
                                                    batch_size=batch_size,
                                                    shuffle=False)
          return train_loader


    trainer = Trainer(max_epochs=self.epochs, devices=[0])
    model = MLP(input_size,512*8,512*4,512*2,512,len(self.train_embds.outputs[0]))

    return trainer, model

  def train(self):
    print(f"creating the model {self.trained_model_name} ...")
    if self.ready_model:
      model = self.ready_model
      trainer = Trainer(max_epochs=self.epochs, devices=[0])
    else:
      trainer, model = self.model_maker() 
    print(f"Training the model ... {self.trained_model_name} time {self.now}")
    trainer.fit(model)
    self._model = model
    print(f"Saving the model ... time {self.now}")
  
  def eval(self):
      if not self._model:
          print("No trained model yet")
          return

      # Set the model to evaluation mode
      model = self._model
      model.eval()

      # Extract embeddings and IDs
      test_embeds = torch.tensor(self.test_embds.embeds)  # Convert to a tensor
      test_ids = self.test_embds.ids

      # Run inference with no_grad
      with torch.no_grad():
          outputs = model(test_embeds)

      # Convert outputs to a format suitable for saving
      preds = []
      for i in range(len(test_ids)):
          preds.append({
              "id": test_ids[i],  # ID from the test data
              "output": outputs[i].cpu().tolist()  # Convert to a Python list for JSON
          })

      # Generate a filename for the predictions
      model_name = self.trained_model_name
      dt_string = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
      output_file = f"{model_name}_test_preds_{dt_string}.json"

      # Save predictions to a JSON file
      with open(output_file, "w") as f:
          json.dump(preds, f, indent=4)

      print(f"Predictions saved to {output_file}")





def main():
    # train_embds: TrainEmbeddings
    # test_embds:  TestEmbeddings
    # lr:float
    # epochs:int
    # trained_model_name:str
    # batch_size:int = 128
    # ready_model:pl.LightningModule = None
    # now:str = datetime.now().strftime("%Y-%m-%d|%H:%M:%S")
  
  rdTrainer = RDTrainer(
      train_embds=trainEmbeddingsBERT,  # Corrected field name
      test_embds=testEmbeddingsBERT,   # Corrected field name
      lr=1e-4,
      epochs=1,
      trained_model_name="FirstTryBERT")
  
  rdTrainer.train()
  rdTrainer.eval()


if __name__ == "__main__":
    main()