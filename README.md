# Multilingual-Spam-Classifier
A transformer-based SMS spam classifier built with Hugging Face Transformers and PyTorch. Achieves over 99% accuracy and F1 score using BERT on the classic SMS Spam Collection dataset.
# BERT SMS Spam Classifier 📱🤖

Fine-tuned a Hugging Face BERT model to classify SMS messages as spam or ham using PyTorch and Transformers. Achieved **99.1% accuracy** and **99.2% F1-score** on the classic SMS Spam Collection dataset.

## Features
- Tokenization with pretrained tokenizer
- Model training using `Trainer` API
- Evaluation with accuracy & weighted F1
- Runs on CPU, MPS (Mac GPU), or CUDA

## Tech Stack
- PyTorch
- Hugging Face Transformers
- SMS Spam Dataset (UCI)

## 📊 Results
| Metric       | Score    |
|--------------|----------|
| Accuracy     | 99.1%    |
| F1 Score     | 99.2%    |
| Eval Loss    | 0.0607   |

## Files
- `main.py` — full training and evaluation script
- `data/` — cleaned SMS data
- `results/` — training output logs

