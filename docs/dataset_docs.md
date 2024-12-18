# Documentation

This code provides a structured framework for handling and processing datasets in `JSON` and `CSV` formats. It defines models for organizing data and classes for extracting the relevant fields.

---

## Classes Overview

### 1. `TrainData`
- **Purpose:** Represents the training dataset.
- **Attributes:**
  - `words`: List of words.
  - `defs`: List of corresponding definitions.
  - `words_embds`: (Optional) List of word embeddings (each embedding is a list of floats).

---

### 2. `TestData`
- **Purpose:** Represents the testing dataset.
- **Attributes:**
  - `defs`: List of definitions.
  - `ids`: List of corresponding IDs.

---

### 3. `TrainFile`
- **Purpose:** Handles the extraction of data for training purposes from JSON or CSV files.
- **Attributes:**
  - `path`: The path to the dataset file.
  - `kind`: The file type, either `json` or `csv`.
- **Methods:**
  - `_extract_data_json(word_h, definition_h, emb_h=None)`:  
    - Extracts words, definitions, and optional embeddings from a JSON file.
  - `_extract_data_csv(word_h, definition_h, emb_h=None)`:  
    - Extracts words, definitions, and optional embeddings from a CSV file (not fully tested yet).
  - `extract_data(word_h, definition_h, emb_h=None)`:  
    - Determines the file type and calls the appropriate extraction function.

---

### 4. `TestFile`
- **Purpose:** Handles the extraction of data for testing purposes from JSON or CSV files.
- **Attributes:**
  - `path`: The path to the dataset file.
  - `kind`: The file type, either `json` or `csv`.
- **Methods:**
  - `_extract_data_json(def_h, id_h)`:  
    - Extracts definitions and IDs from a JSON file.
  - `_extract_data_csv(def_h, id_h)`:  
    - Extracts definitions and IDs from a CSV file using `csv.DictReader`.
  - `extract_data(def_h, id_h)`:  
    - Determines the file type and calls the appropriate extraction function.

---

## Workflow

1. **Define Your Dataset File:**  
   - Create an instance of either `TrainFile` or `TestFile`, providing the `path` and `kind` (file type).
   
2. **Specify Column Headers:**  
   - Provide the column names (headers) in the dataset for words, definitions, IDs, or embeddings when calling the `extract_data` method.

3. **Data Extraction:**  
   - The code processes the file based on its type (`json` or `csv`) and returns a structured object (`TrainData` or `TestData`) containing the extracted fields.

---

## Example Usage

```python
# Example for training data
train_file = TrainFile(path="train.json", kind="json")
train_data = train_file.extract_data(word_h="word", definition_h="definition", emb_h="embedding")

# Example for testing data
test_file = TestFile(path="test.csv", kind="csv")
test_data = test_file.extract_data(def_h="definition", id_h="id")
