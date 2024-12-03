from dataset import TrainData, TestData
import pickle 
from pydantic import BaseModel


class FastTrain(BaseModel):
  trainData: TrainData
  testData: TestData
  models: dict
  batch_size: int # training
  embd_batch_size: int # embedding
  epochs: int
  optimizer: str # from list (adam, sgd)
  activiation: str # from list (gelu, relu)
  given_model: str
  lr: float
  dropout: float | None
  output_size: int
  model_specs: dict

  def run_train(self):
    # run python code given the specs
    print("saving fastTrain specs")
    with open('./fastTrain.pickle', 'wb') as handle:
        pickle.dump(self, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print("Saved")
    # run python code
    print("Now use `python fastTrainCode.py` to run the training")