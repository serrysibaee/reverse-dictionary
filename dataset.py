# Build dataset object
import json
import csv
from pydantic import BaseModel
from pathlib import Path
from tqdm import tqdm


# models for th pipline
class TrainData(BaseModel):
  words: list[str]
  defs: list[str]
  words_embds: list[list[float]] | None

class TestData(BaseModel):
  defs: list[str]
  ids: list[str]



# handle the files and extarct needed data
class TrainFile(BaseModel):
  path: Path
  kind: str
  def _extract_data_json(self,word_h, definition_h, emb_h:str=None):
    # open json
    with open(self.path, "r") as f:
      data = json.load(f)
    words = []
    defs = []
    embds = [] if emb_h else None
    for i in tqdm(data, desc="extracting data"):
      words.append(i[word_h])
      defs.append(i[definition_h])
      embds.append(i[emb_h]) if emb_h else None
    return TrainData(words=words, defs=defs, words_embds=embds)
  def _extract_data_csv(self,word_h, definition_h, emb_h:str=None):
    # This function is not tested yet 
    # open csv
    with open(self.path, "r") as f:
      data = csv.reader(f)
    words = []
    defs = []
    embds = [] if emb_h else None
    for i in data:
      words.append(i[word_h])
      defs.append(i[definition_h])
      embds.append(i[emb_h]) if emb_h else None
    
    return TrainData(words=words, defs=defs, words_embds=embds)

  def extract_data(self,word_h, definition_h, emb_h:str=None):
    if self.kind == "json":
      return self._extract_data_json(word_h, definition_h, emb_h)
    elif self.kind == "csv":
      return self._extract_data_csv(word_h, definition_h, emb_h)


class TestFile(BaseModel):
  path: Path
  kind: str
  def _extract_data_json(self,def_h:str, id_h:str):
    # open json
    with open(self.path, "r") as f:
      data = json.load(f)
    
    defs = []
    ids = []
    for i in data:
      defs.append(i[def_h])
      ids.append(i[id_h])
    return TestData(defs=defs, ids=ids)

  def _extract_data_csv(self,def_h:str, id_h:str):
    ...

  def extract_data(self, def_h:str, id_h:str):
    if self.kind == "json":
      return self._extract_data_json(def_h, id_h) 
    if self.kind == "csv":
      return self._extract_data_csv(def_h, id_h)
  


