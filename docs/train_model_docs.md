# Documentation

This code provides a framework for building and training neural network models using PyTorch Lightning, specifically for embedding-based tasks like reverse dictionary applications. It handles the full training process, including model definition, dataset preparation, and evaluation.

---

## Class Overview

### 1. `RDTrainer`
- **Purpose:** Manages the entire training and evaluation process for the reverse dictionary task.
- **Attributes:**
  - `train_embds` (TrainEmbeddings): Embeddings and outputs for training data.
  - `test_embds` (TestEmbeddings): Embeddings and IDs for testing data.
  - `lr` (float): Learning rate for the optimizer.
  - `epochs` (int): Number of training epochs.
  - `trained_model_name` (str): Name of the model (used for saving outputs).
  - `save_checkpoint` (bool): Whether to save model checkpoints during training (default: `False`).
  - `batch_size` (int): Batch size for training and testing (default: `128`).
  - `_model` (pl.LightningModule): Stores the trained model after training.
  - `ready_model` (pl.LightningModule): Allows providing a pre-built model to skip model creation.
  - `now` (str): Timestamp for tracking the training process.
- **Config:**
  - `arbitrary_types_allowed`: Allows non-standard types like PyTorch objects.

---

## Functions Overview

### 1. `model_maker`
- **Purpose:** Creates the model, data loaders, and trainer for training.
- **Process:**
  - Defines a custom dataset for PyTorch.
  - Constructs a multi-layer perceptron (MLP) model with:
    - 4 hidden layers of decreasing size.
    - GELU activation and dropout regularization.
  - Configures the optimizer (`AdamW`) and loss function (`MSELoss`).
  - Prepares the PyTorch Lightning trainer for training.
- **Returns:** A tuple containing the `Trainer` and the MLP model.

---

### 2. `train`
- **Purpose:** Trains the model using the training embeddings.
- **Process:**
  - Creates a model using `model_maker` (or uses `ready_model` if provided).
  - Fits the model using PyTorch Lightning's `Trainer`.
  - Stores the trained model in the `_model` attribute.
- **Output:** Prints progress and saves the trained model.

---

### 3. `eval`
- **Purpose:** Evaluates the trained model on test embeddings.
- **Process:**
  - Checks if a trained model exists.
  - Runs inference on the test embeddings to generate outputs.
  - Saves predictions (test IDs and corresponding outputs) to a JSON file.
- **Output:** Saves predictions to a timestamped JSON file.

---

## Internal Components

### Custom Dataset
- **Purpose:** Wraps input and output embeddings into a PyTorch-compatible dataset.
- **Attributes:**
  - `input_vectors`: Embedding vectors for definitions (inputs).
  - `output_vectors`: Embedding vectors for words (outputs).
- **Methods:**
  - `__len__`: Returns the number of samples.
  - `__getitem__`: Retrieves input-output pairs by index.

---

### MLP Model
- **Purpose:** Defines a multi-layer perceptron for mapping definition embeddings to word embeddings.
- **Architecture:**
  - Input layer: Size equals the dimensionality of input embeddings.
  - Hidden layers: 4 layers with progressively smaller dimensions (512 × 8 → 512 × 4 → 512 × 2 → 512).
  - Output layer: Size equals the dimensionality of output embeddings.
  - Activation: GELU.
  - Regularization: Dropout (0.2).
- **Methods:**
  - `forward`: Defines the forward pass.
  - `configure_optimizers`: Sets up the AdamW optimizer.
  - `training_step`: Defines the training step with MSE loss.

---

## Workflow

1. **Initialize `RDTrainer`:**
   - Provide `TrainEmbeddings` and `TestEmbeddings`, along with hyperparameters like `lr`, `epochs`, and `batch_size`.

2. **Train the Model:**
   - Call the `train` method to create and train the model.
   - Optionally, use `ready_model` to provide a pre-built model.

3. **Evaluate the Model:**
   - Call the `eval` method to run inference on test embeddings.
   - Saves predictions to a JSON file.

---

## Example Usage

```python
from reverse_dictionary.embeddings import TrainEmbeddings, TestEmbeddings
from reverse_dictionary.train_model import RDTrainer

# Initialize training and testing embeddings from embeddings file
# ................

# Initialize the RDTrainer
trainer = RDTrainer(
    train_embds=train_embds,
    test_embds=test_embds,
    lr=1e-3,
    epochs=10,
    trained_model_name="reverse_dictionary_model",
    save_checkpoint=True
)

# Train the model
trainer.train()

# Evaluate the model
trainer.eval()
