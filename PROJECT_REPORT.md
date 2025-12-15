# Sentiment Analysis Project Report

**Group ID:** 3  
**Team Members:** Ruoxi Li, William Wang, Chunlin An  
**Course:** QMSS5074GR - Machine Learning  
**GitHub Repository:** https://github.com/michaelliruoxi/G1_Project3.git

---

## Executive Summary

This project implements and compares multiple machine learning approaches for binary sentiment classification on the Stanford Sentiment Treebank (SST-2) dataset. We evaluated traditional machine learning models, neural networks, and transfer learning approaches, ultimately achieving **89.74% accuracy** with a fine-tuned DistilBERT model.

---

## Model Performance

### Final Test Set Results

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC | Category |
|-------|----------|-----------|--------|----------|---------|----------|
| **DistilBERT (fine-tuned)** | **0.8974** | **0.9267** | **0.8719** | **0.8985** | **0.9583** | Transfer Learning |
| Tuned Logistic Regression | 0.7731 | 0.7657 | 0.8128 | 0.7885 | 0.8534 | Traditional |
| Linear SVM | 0.7705 | 0.7683 | 0.8005 | 0.7841 | 0.8468 | Traditional |
| CNN | 0.7692 | 0.7568 | 0.8202 | 0.7872 | 0.8645 | Neural |
| MLP (unfrozen embeddings) | 0.7538 | 0.7730 | 0.7463 | 0.7594 | 0.8298 | Neural |
| XGBoost | 0.7359 | 0.7212 | 0.8030 | 0.7599 | 0.7991 | Traditional |
| Random Forest | 0.7295 | 0.7191 | 0.7882 | 0.7521 | 0.7907 | Traditional |

**Key Findings:**
- **Best Overall Model:** Fine-tuned DistilBERT achieved the highest performance across all metrics
- **Best Traditional Model:** Tuned Logistic Regression (C=1.0) with 77.31% accuracy
- **Best Neural Model:** CNN with 76.92% accuracy
- **Statistical Significance:** McNemar's test between DistilBERT and Logistic Regression (χ² = 51.4860, p < 0.0001) confirms the improvement is statistically significant

### Top Per Category Performance

- **Neural Networks:** CNN (Accuracy = 0.7692)
- **Traditional ML:** Tuned Logistic Regression (Accuracy = 0.7731)
- **Transfer Learning:** DistilBERT fine-tuned (Accuracy = 0.8974)

---

## Key Hyperparameters

### Traditional Models

#### Logistic Regression
- **C (Regularization):** 1.0 (tuned via 5-fold cross-validation)
- **Solver:** liblinear
- **Max iterations:** 1000
- **Hyperparameter search space:** C ∈ {0.01, 0.1, 1.0, 10.0, 100.0}
- **Best CV Accuracy:** 0.7714

#### Random Forest
- **n_estimators:** 100
- **Random state:** 42
- **n_jobs:** -1 (parallel processing)

#### XGBoost
- **n_estimators:** 100
- **Random state:** 42

### Neural Network Models

#### CNN Architecture
- **Embedding dimension:** 128 (from-scratch) / 100 (GloVe)
- **Convolutional filters:** 128 per layer
- **Kernel sizes:** 3 and 5
- **Dense units:** 128
- **Dropout:** 0.5
- **Batch size:** 32
- **Epochs:** 20
- **Optimizer:** Adam

#### MLP (Multi-Layer Perceptron)
- **Embedding dimension:** 128
- **Dense units:** 128
- **Batch size:** 32
- **Epochs:** 10

### Transfer Learning (DistilBERT)

#### Fine-tuning Hyperparameters
- **Model:** distilbert-base-uncased
- **Learning rate:** 2e-5 (conservative fine-tuning rate)
- **Batch size:** 16 (per device, limited by GPU memory)
- **Epochs:** 5 (maximum, with early stopping)
- **Weight decay:** 0.01
- **Early stopping patience:** 1 epoch
- **Evaluation strategy:** epoch
- **Save strategy:** epoch
- **Save total limit:** 1 checkpoint
- **Metric for best model:** eval_loss (minimized)

**Rationale:**
- **Learning rate (2e-5):** Conservative rate prevents destruction of pre-trained weights while allowing fine-tuning
- **Batch size (16):** Selected to fit GPU memory reliably
- **Early stopping (patience=1):** Prevents overfitting while allowing performance gains
- **Weight decay (0.01):** L2 regularization to prevent overfitting

### Hyperparameter Optimization (CNN)

Using Keras Tuner (RandomSearch) with the following search space:

- **embedding_dim:** {64, 128, 256}
- **filters:** {64, 128, 256}
- **kernel_size:** {3, 5}
- **dense_units:** {32, 64, 128}
- **dropout:** [0.2, 0.6] (step 0.1)
- **learning_rate:** [1e-4, 3e-3] (log scale)

**Best Configuration Found:**
- **embedding_dim:** 128
- **filters:** 128
- **kernel_size:** 3
- **dense_units:** 128
- **dropout:** 0.5
- **learning_rate:** ≈2.57e-4
- **Best validation accuracy:** 0.7831

**Stopping Criteria:**
- Maximum trials: 10
- Early stopping on validation loss (patience=2)
- Restore best weights enabled

---

## Challenges Faced During Training and Solutions

### 1. Environment Setup and Framework Compatibility

**Challenge:**  
During environment setup, we encountered compatibility issues using **Keras 3** (API/packaging changes and ecosystem friction for NLP tooling), which made parts of the workflow unstable.

**Solution:**
- Switched transformer fine-tuning to **PyTorch** using the **Hugging Face Transformers** stack (DistilBERT), which is the most stable/standard path for BERT-family fine-tuning
- Kept **Keras/TensorFlow** for the CNN and MLP models where it worked reliably
- On Windows, wrote tuning artifacts to a local writable directory (instead of OneDrive) to avoid locked-file/sync issues during Keras Tuner runs

**Impact:**  
Moving transformers to PyTorch provided:
- Better compatibility with Hugging Face Transformers
- A stable fine-tuning loop via the `Trainer` API (epoch evaluation, checkpointing, and early stopping)
- More predictable GPU memory behavior during transformer training

This framework split (Keras for CNNs, PyTorch for transformers) allowed us to leverage the strengths of each framework while avoiding compatibility issues.

---

### 2. GPU Memory Constraints

**Challenge:**  
Limited GPU memory restricted batch size options for transformer models.

**Solution:**
- Reduced batch size to 16 per device for DistilBERT fine-tuning
- Used checkpointing (saving per epoch with `save_total_limit=1`) and `load_best_model_at_end=True` so long runs were recoverable and the best validation checkpoint was retained

**Impact:**  
Successfully trained DistilBERT without out-of-memory errors while maintaining training efficiency.

---

### 3. Overfitting Prevention

**Challenge:**  
Neural networks, especially CNNs and transformer models, showed tendency to overfit on training data.

**Solutions Implemented:**

#### For DistilBERT:
- **Early stopping:** Patience of 1 epoch to stop training when validation loss stops improving
- **Weight decay:** 0.01 L2 regularization
- **Load best model:** `load_best_model_at_end=True` ensures final model is from best validation epoch
- **Limited epochs:** Maximum of 5 epochs prevents excessive training

#### For CNN:
- **Dropout:** 0.5 dropout rate in dense layers
- **Early stopping:** Monitored validation loss with patience=2 during hyperparameter tuning
- **Capacity control:** Tuned model size (filters/units) and dropout to reduce overfitting

**Impact:**  
Validation accuracy improved while maintaining generalization. Best CNN trial achieved 78.31% validation accuracy with controlled overfitting.

---

### 4. Learning Rate Selection for Fine-Tuning

**Challenge:**  
Selecting appropriate learning rate for fine-tuning pre-trained transformers without destroying learned representations.

**Solution:**
- Used conservative learning rate of **2e-5**, which is standard for BERT-family fine-tuning
- This rate is small enough to preserve pre-trained weights while allowing task-specific adaptation
- Avoided larger rates (e.g., 1e-3) that could cause catastrophic forgetting

**Impact:**  
Achieved optimal fine-tuning with 89.74% test accuracy, demonstrating effective transfer learning.

---

### 5. Model Misclassifications

**Challenge:**  
Best model (DistilBERT) still misclassified certain examples, particularly those with:
- Negation patterns ("not good", "never liked")
- Contrastive clauses ("but", "however")
- Sarcasm and irony
- Weak or ambiguous sentiment cues

**Solutions Explored:**
- **Error Analysis:** Identified 20+ misclassified examples and analyzed 5 in detail
- **Data Augmentation:** Implemented synonym swapping, improving Logistic Regression from 77.31% → 77.95% accuracy
- **Future Improvements:** Identified need for:
  - Targeted data augmentation for negation patterns
  - Collecting more examples of contrastive clauses
  - Ensemble methods combining multiple models
  - Threshold calibration for ambiguous cases

**Impact:**  
Error analysis provided actionable insights for model improvement, though not all solutions were fully implemented in this project.

---

### 6. Hyperparameter Search Space Design

**Challenge:**  
Designing effective search space for CNN hyperparameter tuning that balances exploration and computational efficiency.

**Solution:**
- Used Keras Tuner RandomSearch (more efficient than grid search)
- Limited to 10 trials with early stopping
- Defined reasonable ranges based on literature and initial experiments:
  - Embedding dimensions: powers of 2 (64, 128, 256)
  - Filters: powers of 2 (64, 128, 256)
  - Learning rate: log scale from 1e-4 to 3e-3
- Each trial used early stopping to prevent wasted computation

**Impact:**  
Found optimal CNN configuration (val_accuracy=0.7831) efficiently within computational budget.

---

### 7. Model Comparison Across Architectures

**Challenge:**  
Comparing models with fundamentally different architectures (linear vs. tree-based vs. neural vs. transformers) fairly.

**Solution:**
- Evaluated all models on the same held-out test set
- Used consistent metrics: Accuracy, Precision, Recall, F1-Score, ROC-AUC
- Performed statistical significance testing (McNemar's test) between best models
- Considered deployment constraints (latency, memory, interpretability) in analysis

**Impact:**  
Comprehensive comparison table enabled informed model selection based on both performance and practical considerations.

---

### 8. Data Preprocessing Consistency

**Challenge:**  
Ensuring consistent preprocessing across different model types (TF-IDF for traditional, tokenization for neural).

**Solution:**
- Created reusable preprocessing pipeline functions
- Saved preprocessors (TF-IDF vectorizer, tokenizers) to disk for reproducibility
- Documented preprocessing steps clearly
- Used same train/validation/test splits across all models

**Impact:**  
Fair comparison ensured by consistent data handling across all experiments.

---

## Additional Implementations

### Data Augmentation
- Implemented synonym swapping augmentation
- Generated 1,988 augmented samples
- Improved Logistic Regression: 77.31% → 77.95% accuracy (F1: 0.7885 → 0.7943)

### Sentiment Lexicon Integration
- Integrated VADER sentiment scores as features
- Result: TF-IDF + VADER + Logistic Regression achieved 76.28% accuracy (slightly worse than baseline)
- **Finding:** Lexicon features showed domain mismatch/redundancy with learned representations

### Model Deployment
- Deployed best model (DistilBERT) as REST API using FastAPI
- Endpoint: `/predict` accepts text reviews and returns sentiment predictions with probabilities
- Model loaded from checkpoint: `bert_results_20251211-212116-847074000`

---

## Conclusions

1. **Transfer learning significantly outperforms traditional and neural approaches** for this sentiment classification task, with DistilBERT achieving 89.74% accuracy.

2. **Hyperparameter tuning is crucial** - CNN performance improved from baseline to 78.31% validation accuracy through systematic hyperparameter search.

3. **Overfitting management is essential** - Early stopping, dropout, and weight decay were critical for achieving good generalization.

4. **Error analysis provides actionable insights** - Systematic analysis of misclassifications revealed specific linguistic patterns that challenge the model.

5. **Deployment considerations matter** - While DistilBERT performs best, simpler models like Logistic Regression may be preferred for low-latency or interpretability requirements.

---

## Future Work

1. **Enhanced Data Augmentation:** Implement back-translation and more sophisticated augmentation techniques
2. **Ensemble Methods:** Combine multiple models (e.g., DistilBERT + CNN) for improved robustness
3. **Domain-Specific Fine-Tuning:** Fine-tune on domain-specific data if deploying to specific review domains
4. **Model Compression:** Explore quantization or distillation to reduce model size for deployment
5. **Advanced Architectures:** Experiment with larger transformer models (RoBERTa, ALBERT) or ensemble approaches

---

## References

- Stanford Sentiment Treebank (SST-2) Dataset
- Hugging Face Transformers Library
- Keras Tuner for Hyperparameter Optimization
- FastAPI for Model Deployment
- GloVe Embeddings (6B tokens, 100 dimensions)

---

**Project Repository:** https://github.com/michaelliruoxi/G1_Project3.git
