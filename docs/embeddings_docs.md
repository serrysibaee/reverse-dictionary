# Documentation

This code provides functions and models for generating text embeddings using Hugging Face models (`SentenceTransformer` and BERT) for both training and testing datasets. The embeddings are used for tasks such as reverse dictionary lookups.

---

## Classes Overview

### 1. `TrainEmbeddings`
- **Purpose:** Represents the embeddings generated for training data.
- **Attributes:**
  - `embeds`: List of embeddings for the definitions (each embedding is a list of floats).
  - `outputs`: List of embeddings for the words (each embedding is a list of floats).

---

### 2. `TestEmbeddings`
- **Purpose:** Represents the embeddings generated for testing data.
- **Attributes:**
  - `embeds`: List of embeddings for the definitions (each embedding is a list of floats).
  - `ids`: List of corresponding IDs for the test data.

---

## Functions Overview

### 1. `create_train_embds_ST`
- **Purpose:** Creates embeddings for training data using a `SentenceTransformer` model.
- **Parameters:**
  - `model_name` (str): Name of the pre-trained model (e.g., `all-MiniLM-L6-v2`).
  - `data` (TrainData): Training data object containing words, definitions, and optional word embeddings.
  - `batch_size` (int): Batch size for embedding generation (default: 128).
- **Returns:** A `TrainEmbeddings` object with embeddings for definitions and words.

---

### 2. `create_test_embds_ST`
- **Purpose:** Creates embeddings for testing data using a `SentenceTransformer` model.
- **Parameters:**
  - `model_name` (str): Name of the pre-trained model.
  - `data` (TestData): Testing data object containing definitions and IDs.
  - `batch_size` (int): Batch size for embedding generation (default: 128).
- **Returns:** A `TestEmbeddings` object with embeddings for definitions and corresponding IDs.

---

### 3. `create_train_embds_BERT`
- **Purpose:** Creates embeddings for training data using a Hugging Face BERT model.
- **Parameters:**
  - `model_name` (str): Name of the pre-trained BERT model.
  - `data` (TrainData): Training data object containing words, definitions, and optional word embeddings.
  - `batch_size` (int): Batch size for embedding generation (default: 128).
- **Process:**
  - Tokenizes input sentences and processes them through the BERT model.
  - Extracts sentence embeddings by averaging the `last_hidden_state` output of the BERT model.
- **Returns:** A `TrainEmbeddings` object with embeddings for definitions and words.

---

### 4. `create_test_embds_BERT`
- **Purpose:** Creates embeddings for testing data using a Hugging Face BERT model.
- **Parameters:**
  - `model_name` (str): Name of the pre-trained BERT model.
  - `data` (TestData): Testing data object containing definitions and IDs.
  - `batch_size` (int): Batch size for embedding generation (default: 128).
- **Process:**
  - Tokenizes input sentences and processes them through the BERT model.
  - Extracts sentence embeddings by averaging the `last_hidden_state` output of the BERT model.
- **Returns:** A `TestEmbeddings` object with embeddings for definitions and corresponding IDs.

---

## Workflow

1. **Define Your Dataset:**  
   - Use `TrainData` or `TestData` objects to organize the dataset before generating embeddings.

2. **Choose the Model Type:**  
   - Use either a `SentenceTransformer` model for simplicity or a Hugging Face BERT model for custom processing.

3. **Generate Embeddings:**  
   - Call one of the functions (`create_train_embds_ST`, `create_test_embds_ST`, `create_train_embds_BERT`, `create_test_embds_BERT`) to create embeddings.

4. **Batch Processing:**  
   - Large datasets are processed in batches using `tqdm` for progress tracking.

---

## Example Usage

```python
from reverse_dictionary.dataset import TrainData, TestData

# Training data
train_data = TrainData(
    words=["word1", "word2"],
    defs=["definition1", "definition2"],
    words_embds=None
)

# Create training embeddings using SentenceTransformer
train_embeddings = create_train_embds_ST(
    model_name="all-MiniLM-L6-v2", 
    data=train_data
)

# Testing data
test_data = TestData(
    defs=["definition1", "definition2"],
    ids=["id1", "id2"]
)

# Create testing embeddings using BERT
test_embeddings = create_test_embds_BERT(
    model_name="bert-base-uncased",
    data=test_data
)
