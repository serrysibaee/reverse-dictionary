# build embedding system with hugginface models for texts
print("Loading libraries...")
from pydantic import BaseModel
from typing import Optional
from dataset import TrainData, TestData
from sentence_transformers import SentenceTransformer

device = "cuda" if torch.cuda.is_available() else "cpu"

class Embeddings(BaseModel):
  embeds:   list[list[float]]
  outputs:  list[list[float]]

def create_embs_sen_tran(model_name:str,
                        data:TrainData, 
                        batch_size:int=0):
  model = SentenceTransformer(model_name,
                                  device=device)
  defs = data.defs
  words = data.words
  embds = data.words_embds
  print("ebedding the defs...")
  defs_embds = model.encode(defs, 
                                  batch_size = batch_size,
                                   device=device)
  print("ebedding the words...")
  words_embds = model.encode(words, 
                                  batch_size = batch_size,
                                  device=device) if not embds else embds
  return Embeddings(embeds=[], outputs=[])

def create_embs_bert(data:TrainData):
  ...




def main():
    trainData = TrainData(words=['train', 'test'],
                      defs=["def1","def2"],
                      words_embds=[[1.0]]
                      )
    
    embeds = create_embs_sen_tran("model_name", trainData)
    print(embeds) 

if __name__ == "__main__":
    main()