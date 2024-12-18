# build embedding system with hugginface models for texts
print("Loading Embeddings libraries...")
from pydantic import BaseModel
from typing import Optional
from reverse_dictionary.dataset import TrainData, TestData
from sentence_transformers import SentenceTransformer
import transformers # import models 
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import torch
device = "cuda" if torch.cuda.is_available() else "cpu"

class TrainEmbeddings(BaseModel):
  embeds:   list[list[float]]
  outputs:  list[list[float]]

class TestEmbeddings(BaseModel):
  embeds: list[list[float]]
  ids:    list[str]

def create_train_embds_ST(model_name:str, data:TrainData,batch_size:int=128):
  print("Preparing the Train Sentence Model ...")
  model = SentenceTransformer(model_name, device=device)
  defs = data.defs
  words = data.words
  embds = data.words_embds
  print("Embedding the Train defs  ...")
  defs_embds = model.encode(defs, batch_size = batch_size, device=device)
  print("Embedding the Train words ...")
  words_embds = model.encode(words, batch_size = batch_size, device=device) if not embds else embds
  return TrainEmbeddings(embeds=defs_embds, outputs=words_embds)

def create_test_embds_ST(model_name:str, data:TestData,batch_size:int=128):
  print("preparing the Test Sentence Model ...")
  model = SentenceTransformer(model_name, device=device)
  defs = data.defs
  ids = data.ids
  print("ebedding the Test defs  ...")
  defs_embds = model.encode(defs, batch_size = batch_size, device=device)
  return TestEmbeddings(embeds=defs_embds, ids=ids)

def create_train_embds_BERT(model_name:str,data: TrainData,batch_size:int=128):
    print("Preparing the Train BERT Model ...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()
    
    def get_embeddings(sentences):
        embeddings = []
        pbar = tqdm(range(0, len(sentences), batch_size), desc="Embedding Train sentences")
        with torch.no_grad():
            for i in pbar:
                batch_sentences = sentences[i:i + batch_size]
                inputs = tokenizer(batch_sentences, return_tensors="pt", padding=True, truncation=True).to(device)
                outputs = model(**inputs)
                embeddings.extend(outputs.last_hidden_state.mean(dim=1).cpu().numpy())
                pbar.set_postfix({'Processed': i + len(batch_sentences), 'Total': len(sentences)})
        return embeddings

    print("Embedding the Train defs ...")
    defs_embds = get_embeddings(data.defs)
    
    print("Embedding the Train words ...")
    words_embds = get_embeddings(data.words) if data.words_embds is None else torch.tensor(data.words_embds)

    return TrainEmbeddings(embeds=defs_embds, outputs=words_embds)

def create_test_embds_BERT(model_name:str, data: TestData,batch_size:int= 128):
    print("Preparing the Test Sentence Model ...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()

    def get_embeddings(sentences):
        embeddings = []
        pbar = tqdm(range(0, len(sentences), batch_size), desc="Embedding Test sentences")
        with torch.no_grad():
            for i in pbar:
                batch_sentences = sentences[i:i + batch_size]
                inputs = tokenizer(batch_sentences, return_tensors="pt", padding=True, truncation=True).to(device)
                outputs = model(**inputs)
                embeddings.extend(outputs.last_hidden_state.mean(dim=1).cpu().numpy())
                pbar.set_postfix({'Processed': i + len(batch_sentences), 'Total': len(sentences)})
        return embeddings

    print("Embedding the Test defs ...")
    defs_embds = get_embeddings(data.defs)

    return TestEmbeddings(embeds=defs_embds, ids=data.ids)