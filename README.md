# Reverse Dictionary Training Library (RDTL)

## Overview

This library facilitates the training of reverse dictionary models using sentence embeddings and neural networks. It provides a simple and modular framework that extracts embeddings from training and test data, builds models using PyTorch Lightning, and allows for easy training and evaluation of reverse dictionary tasks.

## Abstract Workflow

### 1. **Install the Required Dependencies**
   First, clone the repository and install the necessary dependencies:
   ```bash
   !git clone https://github.com/serrysibaee/reverse_dictionary.git
   pip -q install -r "reverse_dictionary/requirements.txt"
```

## 2. Data Preparation

The training and test datasets are loaded and processed using `TrainFile` and `TestFile` classes. These files contain the raw data and are extracted into a suitable format for embedding creation.

### Example:
```python
from reverse_dictionary.dataset import TrainFile, TestFile

# Load the training and testing data
trainFile = TrainFile(path="train_temp.json", kind="json")
testFile = TestFile(path="test_temp.json", kind="json")

# Extract data: specify columns to be used for word/definition and gloss/id mappings
trainData = trainFile.extract_data("word", "def", "electra")
testData = testFile.extract_data("gloss", "id")
```

## 3. Embedding Creation

Sentence embeddings for both the training and test datasets are created using pre-trained models from the `sentence-transformers` library. You can specify different models for training and test data.

### Example:
```python
from reverse_dictionary.embeddings import create_train_embds_ST, create_test_embds_ST

# Create embeddings for training and test data
trainEmbeddings = create_train_embds_ST("paraphrase-MiniLM-L6-v2", trainData)
testEmbeddings = create_test_embds_ST("paraphrase-MiniLM-L6-v2", testData)
```

## 4. Model Training and Evaluation

The `RDTrainer` class handles the creation, training, and evaluation of the model. You can configure parameters like learning rate (`lr`), number of epochs, and the model name. The model is trained using PyTorch Lightning, and once the training is complete, the evaluation process will generate predictions for the test dataset.

### Example:
```python
from reverse_dictionary.train_model import RDTrainer

# Initialize the trainer with embeddings, learning rate, epochs, and model name
rdTrainer = RDTrainer(train_embds=trainEmbeddings,
                      test_embds=testEmbeddings,
                      lr=1e-4,
                      epochs=10,
                      trained_model_name="firstTryElectra")

# Train the model
rdTrainer.train()

# Evaluate the model and save predictions
rdTrainer.eval()
```

## Fast Training

The `FastTrain` class helps quickly set up and save a training configuration. It accepts various parameters, including data, model specifications, and hyperparameters, then stores them in a pickle file for later use.

### Class: `FastTrain`

#### Parameters:
- **trainData**: The training dataset.
- **testData**: The test dataset.
- **models**: Dictionary containing model configurations.
- **batch_size**: Batch size for training.
- **embd_batch_size**: Batch size for embedding.
- **epochs**: Number of epochs for training.
- **optimizer**: Optimizer type ("adam" or "sgd").
- **activation**: Activation function ("gelu" or "relu").
- **given_model**: Model name for training.
- **lr**: Learning rate for training.
- **dropout**: Dropout rate (optional).
- **output_size**: Size of the output layer.
- **model_specs**: Dictionary with specific model specifications.

#### Method: `run_train`
This method saves the training configuration to a pickle file and provides instructions for running the training.

#### Steps:
1. **Save Configuration**: Serializes the configuration parameters into a pickle file (`fastTrain.pickle`).
2. **Instruction**: Prompts the user to run the training with the command `python fastTrainCode.py`.

### Example:
```python
from reverse_dictionary.dataset import TrainData, TestData
from pydantic import BaseModel
import pickle

# Initialize the FastTrain class with parameters
fast_train = FastTrain(
    trainData=train_data,
    testData=test_data,
    models=model_dict,
    batch_size=32,
    embd_batch_size=64,
    epochs=10,
    optimizer="adam",
    activation="gelu",
    given_model="some_model",
    lr=0.001,
    dropout=0.3,
    output_size=128,
    model_specs=model_specs
)

# Run the training setup
fast_train.run_train()
