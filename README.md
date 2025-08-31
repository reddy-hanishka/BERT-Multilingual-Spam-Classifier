# 🤖 Bert-Multilingual-Spam-Classifier
A transformer-based SMS spam classifier built with Hugging Face Transformers and PyTorch. Achieves over 99% accuracy and F1 score using BERT on the classic SMS Spam Collection dataset.


Fine-tuned a Hugging Face BERT model to classify SMS messages as spam or ham using PyTorch and Transformers. Achieved **99.1% accuracy** and **99.2% F1-score** on the classic SMS Spam Collection dataset.

## Features
- Tokenization with pretrained tokenizer
- Model training using `Trainer` API
- Evaluation with accuracy & weighted F1
- Runs on CPU, MPS (Mac GPU), or CUDA
  
## 📚 Dataset
This project uses the **Multilingual Spam Data** dataset from Kaggle. It consists of SMS messages labeled as **spam** or **ham**, and translated across **four languages**:
- English 🇬🇧  
- Hindi 🇮🇳  
- German 🇩🇪  
- French 🇫🇷

## Tech Stack
- PyTorch
- Hugging Face Transformers
- SMS Spam Dataset (UCI)


## Model Explainability & Zero-Shot Capabilities
To enhance interpretability and versatility of the multilingual spam classifier, we incorporated both model explainability tools and zero-shot learning techniques.

#### 🟢 LIME (Local Interpretable Model-agnostic Explanations)
We used LIME to explain individual model predictions by perturbing the input text and observing the impact on output probabilities.
- Explains why a message was classified as spam or ham
- Highlights the most influential words contributing to a classification decision
- Useful for debugging and understanding model behavior in real-world inputs

#### 🔵 SHAP (SHapley Additive exPlanations)
SHAP provides consistent feature attributions using Shapley values from cooperative game theory.
- Offers global and local explanations of model predictions
- Highlights impact of each word/token on the model’s output score
- Visualized using SHAP’s text or summary_plot

### 🌐 Zero-Shot Learning (ZSL)

Our model can be adapted for zero-shot spam detection in new or low-resource languages by leveraging multilingual embeddings from XLM-RoBERTa.
- No retraining required on new language labels
- Uses Hugging Face’s pipeline("zero-shot-classification") interface
- Example: Classify a message in Italian using English labels like “spam” or “ham”


## 📊 Results
| Metric       | Score    |
|--------------|----------|
| Accuracy     | 99.1%    |
| F1 Score     | 99.2%    |
| Eval Loss    | 0.0607   |

## 📁 Files
- `main.py` — full training and evaluation script
- `data/` — cleaned SMS data
- `results/` — training output logs

