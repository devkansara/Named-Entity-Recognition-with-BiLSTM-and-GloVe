# ner-bilstm-glove

This repository contains the implementation of a Named Entity Recognition (NER) system using BiLSTM models, with and without pre-trained GloVe embeddings. This project was developed as part of the CSCI544 course at USC.

## Project Structure

- `hw4_dev_kansara.py`: Main script for training and evaluating the NER models.
- `data/`: Directory containing the train, dev, and test datasets.
- `glove.6B.100d.gz`: Pre-trained GloVe word embeddings.
- `blstm1.pt`, `blstm2.pt`: Saved model files for the BiLSTM models.
- `dev1.out`, `dev2.out`, `test1.out`, `test2.out`: Output files for model predictions on the dev and test datasets.

## Model Descriptions
### Task 1: BiLSTM Model Without Pre-trained Word Embeddings
- Data Processing:

  - Data Loading: Loaded training, development, and test datasets.
  - Creating Vocabulary: Extracted words from the training data, substituting rare words with <UNK>.
  - Data Preprocessing: Tokenized sentences and substituted words with their indices in the vocabulary. Transformed tags into indexes.
  - Model Architecture:

- Embedding Layer: Transformed word indexes into dense vectors.
  - Bidirectional LSTM: Captured sequential data in forward and backward directions.
  - Linear Layer: Mapped LSTM output to the tag space.
  - Activation Function: Applied ELU before the output layer.
  - Loss Function: Used Cross-Entropy Loss.

- Training:

  - Optimization: Used Stochastic Gradient Descent (SGD).
  - Batching: Data divided into batches of size 16.
  - Epochs: Trained the model for 30 epochs.
  - Evaluation: Evaluated on the development dataset after every epoch.

- Results:

  - Accuracy: 95.53%
  - Precision: 82.37%
  - Recall: 73.53%
  - F1 Score: 77.70

### Task 2: BiLSTM Model with Pre-trained GloVe Word Embeddings
- Data Processing:

  - GloVe Embeddings: Used 100-dimensional pre-trained GloVe word embeddings.
Additional Feature: Introduced a binary feature indicating word capitalization.
Model Architecture:

  - Embedding Layer: Initialized with GloVe vectors.
Bidirectional LSTM: Included an additional feature for capitalization.
Linear Layer, Activation Function, and Loss Function: Same as Task 1.
Training:

  - Optimization, Batching, and Epochs: Similar to Task 1.
  - Results:

    - Accuracy: 98.18%
    - Precision: 87.48%
    - Recall: 90.17%
    - F1 Score: 88.80
    - Evaluation

### To evaluate the model predictions, use the provided eval.py script:
```
python eval.py -p dev1.out -g data/dev
python eval.py -p dev2.out -g data/dev
```
