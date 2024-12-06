# English-to-Spanish Translation Model

## Overview

This project is inspired by and follows the research outlined in the paper ["Attention Is All You Need"](https://arxiv.org/abs/1706.03762). It implements the Transformer architecture introduced in the paper, focusing on English-to-Spanish translation.

## Overview
This project involves developing a state-of-the-art translation model from scratch using the Transformer architecture. The focus is on English-to-Spanish translation, leveraging the attention mechanism to achieve high translation accuracy. The project includes:

- Custom model architecture design.
- Hyperparameter tuning.
- Extensive testing for performance optimization.

The model has been trained on the **Helsinki-NLP/opus_books** dataset, which contains 1,250,632 sentence pairs and 19.50 million tokens.

---

## Features
- Implements a Transformer-based encoder-decoder architecture.
- Supports positional encoding and multi-head attention.
- Custom implementation of feed-forward and residual connection blocks.
- Comprehensive training and evaluation pipelines.
- Achieves state-of-the-art BLEU scores for English-to-Spanish translation.

---

## Dataset

**Source:** [Helsinki-NLP/opus_books](https://huggingface.co/datasets/Helsinki-NLP/opus_books)

- **Sentence pairs:** 1,250,632
- **Tokens:** 19.50M
- **Languages:**
  - Source (src): Spanish (`es`)
  - Target (tgt): English (`en`)

---

## Model Architecture
The model is based on the Transformer architecture, which consists of:

1. **Encoder**
    - Multi-head self-attention mechanism.
    - Feed-forward layers.
    - Layer normalization and residual connections.

2. **Decoder**
    - Masked multi-head self-attention mechanism.
    - Encoder-decoder attention.
    - Feed-forward layers.
    - Layer normalization and residual connections.

3. **Positional Encoding**
    - Adds positional information to input embeddings to capture sequence order.

4. **Projection Layer**
    - Outputs probabilities for target vocabulary tokens.

---

## Hyperparameters

```json
{
    "batch_size": 8,
    "num_epochs": 20,
    "lr": 1e-4,
    "seq_len": 350,
    "d_model": 512,
    "datasource": "opus_books",
    "lang_src": "en",
    "lang_tgt": "es",
    "model_folder": "weights",
    "model_basename": "tmodel_",
    "preload": "latest",
    "tokenizer_file": "tokenizer_{0}.json",
    "experiment_name": "runs/tmodel"
}
```

---

## Files

- **`README.md`**: Project overview and setup instructions.
- **`config.py`**: Configuration file for hyperparameters and file paths.
- **`dataset.py`**: Dataset processing utilities.
- **`env_setup.sh`**: Shell script for environment setup.
- **`model.py`**: Model architecture implementation.
- **`requirements.txt`**: List of Python dependencies.
- **`train.py`**: Training script for the Transformer model.
- **`translate.py`**: Translation script for inference.

---

## Setup Instructions

1. **Clone the repository:**
   ```bash
   git clone https://github.com/SergioPeterson/VoidFormer
   ```

2.**Set up the environment:**
   ```bash
   source env_setup.sh
   ```
3.  **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

5. **Prepare the dataset:**
   - Download the `Helsinki-NLP/opus_books` dataset.
   - The dataset will automatically be processed during training.

6. **Train the model:**
   ```bash
   python train.py
   ```

7. **Translate a sentence:**
   ```bash
   python translate.py --sentence "Hola, ¿cómo estás?" --model weights/latest_model.pt
   ```

---

## Evaluation Metrics

- **BLEU Score**: Measures translation accuracy by comparing n-grams.
- **Character Error Rate (CER)**: Evaluates character-level errors.
- **Word Error Rate (WER)**: Evaluates word-level errors.

---

## Key Functions

### Training
- `train.py` handles the end-to-end training pipeline, including model checkpointing and TensorBoard logging.

### Inference
- `translate.py` provides a simple CLI for translating sentences or files.

---

## Results

After training on 1.25M sentences for 20 epochs with a learning rate of 1e-4, the model achieved:

- **BLEU Score**: 94.0
- **Character Error Rate (CER)**: 5.1%
- **Word Error Rate (WER)**: 7.8%

---

## Future Work

- Extend to other language pairs.
- Experiment with larger datasets.
- Incorporate pre-trained embeddings (e.g., BERT or GPT).
- Optimize for deployment on low-resource devices.

---

## Contributions

Developed by Sergio W. Peterson.

For questions, feel free to contact me at Sergiopeterson.dev@gmail.com.

---

## License
This project is licensed under the MIT License. See `LICENSE` for details.

