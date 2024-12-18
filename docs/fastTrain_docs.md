# Fast Training

The `FastTrain` class provides a mechanism for quickly setting up and saving the configuration of a training session. It accepts a variety of parameters, including data, model specifications, and hyperparameters, then stores them in a pickle file for later use.

## Class: `FastTrain`

### Parameters:
- **trainData** (`TrainData`): The training dataset.
- **testData** (`TestData`): The test dataset.
- **models** (`dict`): Dictionary containing model configurations.
- **batch_size** (`int`): The batch size to be used for training.
- **embd_batch_size** (`int`): The batch size to be used for embedding.
- **epochs** (`int`): The number of epochs to train the model.
- **optimizer** (`str`): The optimizer to use, which can be "adam" or "sgd".
- **activation** (`str`): The activation function to use, which can be "gelu" or "relu".
- **given_model** (`str`): The name of the model to use for training.
- **lr** (`float`): The learning rate for the training.
- **dropout** (`float | None`): The dropout rate, optional.
- **output_size** (`int`): The size of the output layer.
- **model_specs** (`dict`): A dictionary containing specific model specifications.

### Method: `run_train`
This method saves the training session's specifications to a pickle file and outputs instructions for running the training.

#### Steps:
1. **Save Configuration**: The configuration parameters are serialized into a pickle file named `fastTrain.pickle`.
2. **Instruction for Running**: The user is prompted to run the training script by executing `python fastTrainCode.py`.

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
```

# FastTrainCode.py
## Model Training and Evaluation

This script is designed to load the configuration from the previously saved `fastTrain.pickle` file and perform model training and evaluation. The abstract structure of the file is as follows:

1. **Loading Configuration**: The saved configuration file (`fastTrain.pickle`) is loaded, containing the necessary parameters for training, such as data, model specifications, and hyperparameters.

2. **Model Creation**: 
   - The `model_maker` function dynamically constructs a neural network model using PyTorch Lightning. 
   - The model architecture is based on user-specified layer sizes, activation functions (ReLU or GELU), and other hyperparameters.
   - It uses pre-trained sentence embeddings (from the `SentenceTransformer` library) to encode the training data into vectors.

3. **Training**:
   - The `MLP` model is trained using the specified optimizer (AdamW or SGD) and loss function (MSE).
   - Training occurs using a custom dataset defined by `CustomDataset` and is managed by the PyTorch Lightning `Trainer`.

4. **Evaluation**:
   - After training, the model is evaluated on the test dataset.
   - Predictions are generated and saved as a JSON file for further analysis.

5. **Multi-Model Training**:
   - The script iterates over a dictionary of embedding models and their respective input sizes, performing training and evaluation for each model configuration.

### Key Points:
- **Training Configuration** is loaded from a pickle file.
- **Custom Dataset** and **MLP Model** are created dynamically.
- **PyTorch Lightning** is used for training and evaluation.
- **Sentence-Transformer** models are used to generate embeddings for text data.
- **Model Evaluations** are saved to JSON files for later use.

