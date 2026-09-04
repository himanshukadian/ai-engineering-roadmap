# Roadmap

Long-term learning progression for AI Engineering.

## Phase 1 — Foundations (Weeks 1–4)

### Mathematics
- Linear algebra: vectors, matrices, matrix multiplication, transpose, inverse, eigenvalues, eigenvectors, SVD
- Calculus: derivatives, gradients, partial derivatives, chain rule, multivariate calculus
- Probability: axioms, conditional probability, Bayes' theorem, common distributions
- Statistics: mean, variance, standard deviation, estimators, MLE, hypothesis testing
- Optimization: convexity, gradient descent, learning rate, convergence

### Python
- NumPy: arrays, broadcasting, vectorization, linear algebra operations
- Pandas: Series, DataFrame, groupby, merging, data cleaning
- Matplotlib/Seaborn: visualization fundamentals

### Machine Learning
- What machine learning is and isn't
- Supervised vs unsupervised vs reinforcement learning
- Linear regression from scratch
- Gradient descent implementation
- Train/validation/test splits
- Overfitting and underfitting

## Phase 2 — Core ML (Weeks 5–10)

### Machine Learning
- Logistic regression
- Decision trees, random forests, gradient boosting
- Support vector machines
- K-nearest neighbors
- Naive Bayes
- Evaluation: accuracy, precision, recall, F1, ROC-AUC, confusion matrix
- Bias-variance tradeoff
- Regularization: L1, L2
- Feature engineering and selection
- Cross-validation

### Deep Learning
- Perceptrons, activation functions
- Backpropagation from scratch
- Neural network architectures
- PyTorch fundamentals
- Training loops, loss functions, optimizers
- Regularization: dropout, batch normalization, weight decay

### Experiments
- Implement linear regression from scratch
- Implement logistic regression from scratch
- Implement a neural network from scratch (no frameworks)
- Compare sklearn vs from-scratch implementations

## Phase 3 — Deep Learning (Weeks 11–18)

### Deep Learning
- CNNs: convolution, pooling, architectures (LeNet, ResNet)
- RNNs: sequential data, LSTMs, GRUs
- Attention mechanism
- Transformers: self-attention, multi-head attention, positional encoding
- Transfer learning
- Fine-tuning pretrained models

### NLP
- Word embeddings: Word2Vec, GloVe
- Tokenization, subword models
- Language modeling (next token prediction)
- Text classification
- Hugging Face ecosystem
- Fine-tuning LLMs

### Computer Vision
- Image classification
- Object detection basics
- Data augmentation

### Experiments
- Train a CNN on CIFAR-10
- Fine-tune a pretrained model
- Compare architectures on a fixed dataset

## Phase 4 — ML Engineering (Weeks 19–26)

### MLOps
- Experiment tracking (MLflow, Weights & Biases)
- Model versioning
- Model serving: REST APIs, batch inference, real-time
- Monitoring: data drift, model degradation
- CI/CD for ML pipelines
- Feature stores

### Data Engineering
- Data pipelines
- ETL fundamentals
- Data quality and validation
- Large-scale data processing

### Systems Design
- ML system design methodology
- Recommendation systems
- Search ranking
- Ads ranking
- Fraud detection
- Real-time inference systems

### Distributed Systems
- Data parallelism
- Model parallelism
- Distributed training
- GPU fundamentals

## Phase 5 — Specialization (Weeks 27+)

Choose a depth area:

### LLM Engineering
- Prompt engineering
- RAG (retrieval-augmented generation)
- Agent architectures
- Evaluation of generative models
- RLHF / alignment

### Production ML
- Large-scale feature engineering
- A/B testing for ML
- Online learning
- Multi-armed bandits

### Research
- Paper reproduction
- Novel architectures
- Benchmark development

### Interview Preparation
- DSA
- ML theory
- ML system design
- Statistics and probability
- Coding (Python, SQL)

## Flagship Project Progression

Ordered by increasing complexity:

1. `linear-regression-from-scratch` — derive, implement, validate against sklearn
2. `neural-network-from-scratch` — forward/backward pass, training loop, no frameworks
3. `image-classifier` — CNN, data augmentation, transfer learning, evaluation
4. `text-classifier` — NLP pipeline, embeddings, fine-tuning
5. `ml-serving-api` — model serialization, REST API, monitoring, deployment
6. `recommendation-engine` — collaborative filtering, content-based, hybrid
7. `llm-fine-tuning` — LoRA, evaluation, deployment

These emerge from the learning process. Do not build them prematurely.
