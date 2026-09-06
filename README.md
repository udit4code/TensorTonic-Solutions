# TensorTonic Solutions

Welcome to my TensorTonic solutions repository!

Here you'll find my solutions to various machine learning and deep learning problems from [TensorTonic](https://tensortonic.com).

## What is TensorTonic?

TensorTonic is a platform where you can implement core algorithms of Machine Learning from scratch.

This repository contains my personal solutions to these problems, automatically synchronized from the platform.

<!-- tensortonic:start -->
# udit4code's TensorTonic Solutions

Verified machine learning implementations completed on [TensorTonic](https://www.tensortonic.com).

<p align="center">
  <img src="https://www.tensortonic.com/api/badge/uks007.svg" alt="TensorTonic Verified Solutions" width="100%" />
</p>

| Problem | Description | Link |
|---|---|---|
| AdaGrad Optimizer | Implement a vectorized AdaGrad update in NumPy with accumulated squared gradients and adaptive per-parameter learning rates. | https://www.tensortonic.com/problems/adagrad-optimizer |
| Implement Adam Optimizer Step | Implement one vectorized Adam optimizer step in NumPy with first and second moments, bias correction, and elementwise parameter updates. | https://www.tensortonic.com/problems/adam-optimizer |
| Bag-of-Words Vector | Build a NumPy bag-of-words count vector from an ordered vocabulary while ignoring out-of-vocabulary tokens. | https://www.tensortonic.com/problems/bag-of-words |
| Batch Shuffling & Mini-Batch Generator | Create shuffled mini-batches from NumPy feature and target arrays with reproducible ordering and final-batch handling. | https://www.tensortonic.com/problems/batch-generator |
| Batch Normalization (Forward) | Implement the batch-normalization forward pass in NumPy using feature-wise statistics, scale, shift, and numerical stability. | https://www.tensortonic.com/problems/batch-normalization |
| Binning | Assign numeric values to ordered bins using supplied boundaries while handling values at interval edges. | https://www.tensortonic.com/problems/binning |
| Implement BM25 Ranking Score | Implement BM25 document ranking with term frequency saturation, inverse document frequency, and length normalization. | https://www.tensortonic.com/problems/bm25 |
| Implement Causal Masking for Attention | Create a causal attention mask that blocks each token from attending to future positions in a sequence. | https://www.tensortonic.com/problems/causal-masking |
| Compute Accuracy, Precision, Recall, F1 | Compute binary accuracy, precision, recall, and F1 score from predicted and true class labels. | https://www.tensortonic.com/problems/classification-metrics |
| Implement Cosine Similarity | Compute cosine similarity between NumPy vectors with dot products, Euclidean norms, and zero-vector handling. | https://www.tensortonic.com/problems/cosine-similarity |
| Compute Covariance Matrix | Compute a sample covariance matrix from centered observations, preserving feature-to-feature relationships. | https://www.tensortonic.com/problems/covariance-matrix |
| Cyclic Encoding | Encode periodic numeric features as sine and cosine coordinates using a specified cycle length. | https://www.tensortonic.com/problems/cyclic-encoding |
| Implement Dot Product | Implement the dot product of equal-length numeric vectors by summing element-wise products without library shortcuts. | https://www.tensortonic.com/problems/dot-product |
| Edit Distance | Compute Levenshtein edit distance between two strings using dynamic programming over insertions, deletions, and substitutions. | https://www.tensortonic.com/problems/edit-distance |
| Compute Entropy for a Node | Compute decision-tree node entropy from class labels using empirical class probabilities and base-two logarithms. | https://www.tensortonic.com/problems/entropy-node |
| ETL Deduplication | Deduplicate ETL records by configured key fields while applying the required policy for repeated entries. | https://www.tensortonic.com/problems/etl-deduplication |
| ETL Dependency Orchestration | Resolve ETL job dependencies into a valid execution order while detecting missing or cyclic dependencies. | https://www.tensortonic.com/problems/etl-dependency-orchestration |
| Implement Euclidean Distance | Compute Euclidean distance between equal-length NumPy vectors as the square root of summed squared differences. | https://www.tensortonic.com/problems/euclidean-distance |
| Expected Value (Discrete Distribution) | Compute the expected value of a discrete distribution from matched outcomes and normalized probabilities. | https://www.tensortonic.com/problems/expected-value-discrete |
| Feature Store Lookup | Combine stored offline and request-time features in input order, using defaults for unknown user IDs. | https://www.tensortonic.com/problems/feature-store-lookup |
| Frequency Encoding | Replace categorical values with their observed frequencies while preserving the original sequence order. | https://www.tensortonic.com/problems/frequency-encoding |
| Implement GELU Activation (Gaussian Error Linear Unit) | Implement the Gaussian Error Linear Unit activation element-wise using the required GELU approximation. | https://www.tensortonic.com/problems/gelu |
| Implement Global Average Pooling | Apply global average pooling to spatial feature maps by averaging each channel across its height and width. | https://www.tensortonic.com/problems/global-avg-pooling |
| Implement Gradient Descent for a 1D Quadratic | Optimize a one-dimensional quadratic with iterative gradient descent and return the parameter trajectory. | https://www.tensortonic.com/problems/gradient-descent-quadratic |
| Jaccard Similarity | Compute Jaccard similarity between two collections as intersection size divided by union size. | https://www.tensortonic.com/problems/jaccard-similarity |
| K-Fold Split (Indices Only) | Generate deterministic K-fold train and validation index splits that use every sample exactly once for validation. | https://www.tensortonic.com/problems/kfold-split |
| Implement Leaky ReLU (with α) | Apply Leaky ReLU element-wise with a configurable negative slope while retaining positive inputs. | https://www.tensortonic.com/problems/leaky-relu |
| Linear Layer Forward | Implement a dense linear layer forward pass by multiplying inputs by weights and adding a bias vector. | https://www.tensortonic.com/problems/linear-layer-forward |
| Linear Regression Closed Form | Fit linear regression with the closed-form normal equation and return coefficients for the supplied design matrix. | https://www.tensortonic.com/problems/linear-regression-closed-form |
| Log Transform | Apply a numerically safe logarithmic transform to numeric features using the required offset or base. | https://www.tensortonic.com/problems/log-transform |
| Logistic Regression Training Loop | Train binary logistic regression in NumPy using sigmoid probabilities, gradient descent, and learned weight and bias parameters. | https://www.tensortonic.com/problems/logistic-regression-training |
| Make Diagonal Matrix | Construct a square diagonal matrix from a one-dimensional vector while setting every off-diagonal entry to zero. | https://www.tensortonic.com/problems/make-diagonal |
| Implement Manhattan Distance | Compute Manhattan distance between equal-length vectors by summing absolute coordinate differences. | https://www.tensortonic.com/problems/manhattan-distance |
| Matrix Factorization SGD Step | Perform one matrix-factorization SGD update for a rated user-item pair with latent factors and regularization. | https://www.tensortonic.com/problems/matrix-factorization-sgd-step |
| Implement Matrix Normalization | Normalize a NumPy matrix using the specified axis and norm while safely handling zero-magnitude slices. | https://www.tensortonic.com/problems/matrix-normalization |
| Matrix Trace | Compute the trace of a square matrix by summing its main diagonal entries without changing the input. | https://www.tensortonic.com/problems/matrix-trace |
| Matrix Transpose | Implement matrix transpose in NumPy without built-in transpose helpers, preserving rectangular shapes and the original input. | https://www.tensortonic.com/problems/matrix-transpose |
| Mean Rating Imputation | Fill missing user-item ratings with the required row or column mean while preserving observed ratings. | https://www.tensortonic.com/problems/mean-rating-imputation |
| Mean Squared Error (MSE) | Compute mean squared error between predictions and targets by averaging their squared element-wise differences. | https://www.tensortonic.com/problems/mean-squared-error |
| Implement Micro-F1 | Compute multiclass micro-F1 by aggregating true positives, false positives, and false negatives across labels. | https://www.tensortonic.com/problems/metrics-f1-micro |
| Min-Max Scaling | Scale numeric values to a requested range using observed minimum and maximum values with constant-input handling. | https://www.tensortonic.com/problems/min-max-scaling |
| Implement Min-Max Normalization | Normalize each NumPy feature to the zero-to-one range with explicit handling for constant columns. | https://www.tensortonic.com/problems/minmax-normalization |
| Model Versioning | Select a production model by highest accuracy, then lower latency, then the most recent timestamp. | https://www.tensortonic.com/problems/model-versioning-basics |
| Monitoring Metrics Selection | Compute the required monitoring metrics for classification, regression, or ranking prediction results. | https://www.tensortonic.com/problems/monitoring-metrics-selection |
| Normalize 3D Vectors | Normalize a 3D vector to unit length in NumPy while returning the required result for a zero vector. | https://www.tensortonic.com/problems/normalize-3d |
| One-Hot Encoding (Multi-class) | Convert multiclass integer labels into a NumPy one-hot matrix with one active column per sample. | https://www.tensortonic.com/problems/one-hot-encoding |
| Ordinal Encoding | Map ordered categorical values to integer ranks using a supplied category ordering and preserve input order. | https://www.tensortonic.com/problems/ordinal-encoding |
| Pad Sequences | Pad or truncate variable-length token ID sequences in NumPy with configurable maximum length and padding values. | https://www.tensortonic.com/problems/pad-sequences |
| Compute Pearson Correlation Matrix | Compute the Pearson correlation matrix between numeric features using centered covariance and standard deviations. | https://www.tensortonic.com/problems/pearson-correlation |
| Percentiles / Quantiles | Calculate requested percentiles from numeric data using the interpolation rule specified by the problem. | https://www.tensortonic.com/problems/percentiles |
| Perplexity Computation | Compute language-model perplexity from token probability distributions and the observed token indices. | https://www.tensortonic.com/problems/perplexity-computation |
| Precision and Recall at K | Compute recommendation precision and recall at K by comparing ranked predictions with relevant items. | https://www.tensortonic.com/problems/precision-recall-at-k |
| Prioritized Experience Replay | Compute prioritized replay sampling probabilities and normalized importance weights from transition priorities. | https://www.tensortonic.com/problems/priority-replay-sample |
| Rank Transform | Replace numeric values with their ranks while applying the specified policy to tied observations. | https://www.tensortonic.com/problems/rank-transform |
| Implement ReLU Activation | Apply the ReLU activation element-wise by replacing negative values with zero and preserving nonnegative inputs. | https://www.tensortonic.com/problems/relu-activation |
| Remove Stopwords | Remove tokens found in a supplied stopword collection while preserving the order of remaining words. | https://www.tensortonic.com/problems/remove-stopwords |
| RMSProp Optimizer (Single Update Step) | Implement one RMSProp update in NumPy using an exponential squared-gradient average and adaptive scaling. | https://www.tensortonic.com/problems/rmsprop-optimizer |
| RNN Step Forward (Tanh Cell) | Implement one vanilla RNN timestep with affine input and recurrent transforms followed by tanh activation. | https://www.tensortonic.com/problems/rnn-step-forward |
| Rolling Standard Deviation | Compute rolling standard deviation over complete time-series windows using the required variance convention. | https://www.tensortonic.com/problems/rolling-standard-deviation |
| Shadow Deployment Evaluation | Compare shadow and production model outcomes using the evaluation criteria defined by the problem. | https://www.tensortonic.com/problems/shadow-deployment-evaluation |
| Implement Sigmoid in NumPy | Implement a vectorized sigmoid activation in NumPy for scalars, lists, vectors, and matrices, including large positive and negative inputs. | https://www.tensortonic.com/problems/sigmoid-numpy |
| Implement Softmax Function | Implement numerically stable softmax by shifting logits before exponentiation and normalizing probabilities. | https://www.tensortonic.com/problems/softmax-function |
| Streaming Min-Max Normalization | Update per-feature running minima and maxima, then normalize each incoming numeric batch with the new state. | https://www.tensortonic.com/problems/streaming-minmax |
| Implement Tanh Activation | Implement the hyperbolic tangent activation element-wise with outputs bounded between minus one and one. | https://www.tensortonic.com/problems/tanh-activation |
| Target Encoding | Encode each categorical value with the mean target observed for its category while preserving row order. | https://www.tensortonic.com/problems/target-encoding |
| Text Chunking | Split text into ordered chunks under the requested size and overlap rules without dropping content. | https://www.tensortonic.com/problems/text-chunking |
| Implement TF-IDF Vectorizer | Build TF-IDF document vectors from token counts and inverse document frequency across a text corpus. | https://www.tensortonic.com/problems/tfidf-vectorizer |
| Top-K Recommendations | Return each user's highest-scoring unseen items with deterministic ranking and a configurable result limit. | https://www.tensortonic.com/problems/top-k-recommendations |
| Detect Train-Serving Skew | Detect train-serving skew by comparing offline and online feature values under configured tolerances. | https://www.tensortonic.com/problems/train-serving-skew |
| Compute 3D Vector Norm | Compute the Euclidean norm of a 3D vector from the square root of summed squared coordinates. | https://www.tensortonic.com/problems/vector-norm-3d |
| Word Count Dictionary | Count token occurrences in text and return a dictionary mapping each distinct word to its frequency. | https://www.tensortonic.com/problems/word-count-dict |
| Implement z-Score Standardization | Standardize NumPy features to zero mean and unit variance with explicit handling for constant columns. | https://www.tensortonic.com/problems/zscore-standardization |
| Fine-tuning Architecture | Build BERT fine-tuning utilities for freezing encoder layers and producing sequence or token classification logits. | https://www.tensortonic.com/research/bert/bert-fine-tuning |
| Masked Language Modeling | Implement BERT masked language modeling with the 80-10-10 replacement strategy, training labels, and vocabulary logits. | https://www.tensortonic.com/research/bert/bert-masked-lm |
| Next Sentence Prediction | Create BERT next-sentence prediction pairs and compute binary classification logits for IsNext and NotNext examples. | https://www.tensortonic.com/research/bert/bert-nsp |
| BERT Pooler | Implement the BERT pooler by projecting the first token's hidden state through a dense layer and tanh activation. | https://www.tensortonic.com/research/bert/bert-pooler |
| Segment Embeddings | Build BERT input embeddings by summing learned token, position, and sentence-segment embedding vectors. | https://www.tensortonic.com/research/bert/bert-segment-embedding |
| WordPiece Tokenization | Implement BERT WordPiece tokenization with greedy longest-match subwords, continuation prefixes, and unknown-token fallback. | https://www.tensortonic.com/research/bert/bert-wordpiece |
| Identity Block | Implement a ResNet identity block with a three-layer bottleneck branch, batch normalization, ReLU, and an unchanged skip path. | https://www.tensortonic.com/research/resnet/resnet-identity-block |
| Backpropagation Through Time | Implement one backpropagation-through-time step using the tanh derivative and hidden-to-hidden weight gradients. | https://www.tensortonic.com/research/rnn/rnn-bptt |
| RNN Cell | Implement an Elman RNN cell that combines the current input and previous hidden state before applying tanh. | https://www.tensortonic.com/research/rnn/rnn-cell |
| Forward Through Sequence | Implement a vanilla RNN forward pass that updates and returns hidden states across every sequence time step. | https://www.tensortonic.com/research/rnn/rnn-forward-sequence |
| Complete Vanilla RNN | Assemble a vanilla RNN that processes sequences into recurrent hidden states and per-time-step output logits. | https://www.tensortonic.com/research/rnn/rnn-full-network |
| Hidden State | Initialize a vanilla RNN hidden state as a floating-point zero matrix for the requested batch and hidden dimensions. | https://www.tensortonic.com/research/rnn/rnn-hidden-state |
| Vanishing Gradients | Simulate vanishing or exploding RNN gradients by repeatedly applying the hidden matrix's spectral norm. | https://www.tensortonic.com/research/rnn/rnn-vanishing-gradients |
| Scaled Dot-Product Attention | Implement scaled dot-product attention in PyTorch using query-key scores, softmax weights, and value aggregation. | https://www.tensortonic.com/research/transformer/transformers-attention |
| Embedding Layer | Create PyTorch token embeddings and scale each lookup by the square root of the Transformer model dimension. | https://www.tensortonic.com/research/transformer/transformers-embedding |
| Encoder Block | Assemble a Transformer encoder block with multi-head attention, residual paths, layer normalization, and a feed-forward network. | https://www.tensortonic.com/research/transformer/transformers-encoder-block |
| Feed-Forward Network | Implement the Transformer's position-wise feed-forward network with two linear projections and a ReLU activation. | https://www.tensortonic.com/research/transformer/transformers-feed-forward |
| Layer Normalization | Implement Transformer layer normalization in NumPy using per-token mean, variance, scale, and bias. | https://www.tensortonic.com/research/transformer/transformers-layer-normalization |
| Multi-Head Attention | Build NumPy multi-head attention with learned projections, per-head scaled attention, concatenation, and output projection. | https://www.tensortonic.com/research/transformer/transformers-multi-head-attention |
| Positional Encoding | Implement sinusoidal Transformer positional encodings in NumPy with alternating sine and cosine dimensions. | https://www.tensortonic.com/research/transformer/transformers-positional-encoding |
| Tokenization | Build a word-level Transformer tokenizer with fixed special-token IDs, sorted vocabulary entries, encoding, and decoding. | https://www.tensortonic.com/research/transformer/transformers-tokenization |
| Skip-gram Pair Generation | Generate Word2Vec skip-gram training pairs by pairing each center token with words inside its context window. | https://www.tensortonic.com/research/word2vec/word2vec-skipgram-pairs |
| Frequent-Word Subsampling | Implement Word2Vec frequent-word subsampling by computing token retention probabilities from corpus frequencies. | https://www.tensortonic.com/research/word2vec/word2vec-subsampling |
| Activation Functions | Implement ReLU, sigmoid, tanh, Leaky ReLU, GELU, and Swish with their analytical derivatives. | https://www.tensortonic.com/study-plans/cracking-dl/dl-activation-functions |
| Computational Graph & Autograd | Build a minimal autograd engine that performs forward and backward passes on a computational graph. | https://www.tensortonic.com/study-plans/cracking-dl/dl-autograd |
| Batch Normalization | Implement batch normalization for training and inference, including batch statistics and running-statistic updates. | https://www.tensortonic.com/study-plans/cracking-dl/dl-batch-normalization |
| Layer Normalization | Implement Layer Normalization (Ba et al, 2016), the standard normalization technique in Transformers. | https://www.tensortonic.com/study-plans/cracking-dl/dl-layer-normalization |
| Perceptron | Train a binary perceptron from zero-initialized weights using ordered samples, step predictions, and error-correction updates. | https://www.tensortonic.com/study-plans/cracking-dl/dl-perceptron |
| Implement Grouped-Query Attention (GQA) | Implement grouped-query attention by mapping query heads onto fewer key-value heads with validated head divisibility. | https://www.tensortonic.com/study-plans/cracking-inference/inference-grouped-query-attention |
| Calculate KV Cache Memory for MHA, MQA, GQA, and MLA | Compute the total KV-cache memory, in bytes, for a full sequence under four attention variants: MHA, MQA, GQA, and MLA. | https://www.tensortonic.com/study-plans/cracking-inference/inference-kv-cache-memory |
| Calculate P50, P95, and P99 Inference Latency | Compute the 50th, 95th, and 99th percentile of a set of observed inference latencies, using linear interpolation between the two closest ranks. | https://www.tensortonic.com/study-plans/cracking-inference/inference-latency-percentiles |
| Estimate Model Memory and Minimum GPU Count | Estimate model-serving memory from weights, key-value cache, activations, overhead, and per-GPU usable capacity. | https://www.tensortonic.com/study-plans/cracking-inference/inference-model-memory-gpu-count |
| Implement Multi-Head Attention (MHA) | Split into h heads, run scaled dot-product attention per head with an optional causal mask, concatenate the heads, and apply an output projection. | https://www.tensortonic.com/study-plans/cracking-inference/inference-multi-head-attention |
| Implement Multi-Head Latent Attention (MLA) | Implement Multi-Head Latent Attention (MLA), and return both the projected attention output and the compressed latent tensor. | https://www.tensortonic.com/study-plans/cracking-inference/inference-multi-head-latent-attention |
| Implement Multi-Query Attention (MQA) | Implement multi-query attention with separate query heads, shared key-value heads, optional masking, and output projection. | https://www.tensortonic.com/study-plans/cracking-inference/inference-multi-query-attention |
| Implement Scaled Dot-Product Attention | Implement batched scaled dot-product attention for self- and cross-attention with optional masks and stable softmax. | https://www.tensortonic.com/study-plans/cracking-inference/inference-scaled-dot-product-attention |
| Implement Symmetric INT8 Quantization | Quantize finite tensors to symmetric INT8 values with an absolute-maximum scale and reconstruct dequantized outputs. | https://www.tensortonic.com/study-plans/cracking-inference/inference-symmetric-int8-quantization |
| AdaBoost from Scratch | Implement AdaBoost binary classification using decision stumps, weighted errors, adaptive sample weights, and weighted voting. | https://www.tensortonic.com/study-plans/cracking-ml/ml-adaboost |
| Agglomerative Clustering | Implement agglomerative hierarchical clustering with single, complete, and average linkage and deterministic cluster labels. | https://www.tensortonic.com/study-plans/cracking-ml/ml-agglomerative |
| Bagging Classifier | Build a bagging classifier from scratch using bootstrap-sampled CART trees and majority-vote predictions. | https://www.tensortonic.com/study-plans/cracking-ml/ml-bagging-classifier |
| Decision Tree Classifier (CART) | Implement a CART decision tree classifier with Gini impurity splits, depth limits, and majority-class leaf predictions. | https://www.tensortonic.com/study-plans/cracking-ml/ml-cart-classifier |
| Decision Tree Regressor | Implement a CART regression tree with MSE reduction splits, stopping criteria, and mean-value leaf predictions. | https://www.tensortonic.com/study-plans/cracking-ml/ml-cart-regressor |
| Categorical Encoding | Encode categorical strings with deterministic label encoding or one-hot vectors ordered by sorted category values. | https://www.tensortonic.com/study-plans/cracking-ml/ml-categorical-encoding |
| DBSCAN | Implement DBSCAN clustering with epsilon neighborhoods, minimum-point density checks, cluster expansion, and noise labels. | https://www.tensortonic.com/study-plans/cracking-ml/ml-dbscan |
| Distance Metrics | Compute Euclidean, Manhattan, cosine, Chebyshev, and Minkowski distances between numeric vectors. | https://www.tensortonic.com/study-plans/cracking-ml/ml-distance-metrics |
| Feature Scaling | Implement column-wise min-max scaling and z-score standardization while handling constant features safely. | https://www.tensortonic.com/study-plans/cracking-ml/ml-feature-scaling |
| Gaussian Naive Bayes | Implement Gaussian Naive Bayes with class priors, per-feature Gaussian likelihoods, and log-probability predictions. | https://www.tensortonic.com/study-plans/cracking-ml/ml-gaussian-naive-bayes |
| Missing Value Imputation | Impute missing numeric values with column-wise mean, median, or most-frequent statistics while preserving observed values. | https://www.tensortonic.com/study-plans/cracking-ml/ml-imputation |
| Isolation Forest | Implement Isolation Forest anomaly detection with random partition trees and path-length based anomaly scores. | https://www.tensortonic.com/study-plans/cracking-ml/ml-isolation-forest |
| K-Means Clustering | Implement K-means clustering with nearest-centroid assignments, centroid updates, convergence checks, and stable labels. | https://www.tensortonic.com/study-plans/cracking-ml/ml-kmeans |
| KNN Classifier | Implement K-nearest neighbors classification using Euclidean distance, majority voting, and deterministic tie-breaking. | https://www.tensortonic.com/study-plans/cracking-ml/ml-knn-classifier |
| Lasso Regression | Implement Lasso regression with gradient descent, an L1 subgradient penalty on weights, and an unregularized bias. | https://www.tensortonic.com/study-plans/cracking-ml/ml-lasso-regression |
| Linear Regression from Scratch | Train linear regression from scratch with mean squared error gradients for weights and bias. | https://www.tensortonic.com/study-plans/cracking-ml/ml-linear-regression-from-scratch |
| Logistic Regression from Scratch | Train binary logistic regression from scratch using sigmoid probabilities, cross-entropy gradients, and gradient descent. | https://www.tensortonic.com/study-plans/cracking-ml/ml-logistic-regression |
| PCA from Scratch | Implement PCA by centering data, eigendecomposing the covariance matrix, and projecting onto the leading components. | https://www.tensortonic.com/study-plans/cracking-ml/ml-pca |
| Random Forest from Scratch | Implement a random forest classifier with bootstrap sampling, random feature subsets at each CART split, and majority voting. | https://www.tensortonic.com/study-plans/cracking-ml/ml-random-forest |
| Regression Metrics | Compute MSE, MAE, and R-squared from scratch, including constant-target handling and rounded metric output. | https://www.tensortonic.com/study-plans/cracking-ml/ml-regression-metrics |
| Ridge Regression | Train Ridge regression with gradient descent, L2-regularized weights, and an unregularized bias term. | https://www.tensortonic.com/study-plans/cracking-ml/ml-ridge-regression |
| Softmax Regression | Train multiclass softmax regression with stable probabilities, one-hot targets, cross-entropy gradients, and gradient descent. | https://www.tensortonic.com/study-plans/cracking-ml/ml-softmax-regression |
| Bag of Words | Build an alphabetical vocabulary and document-term count matrix from a corpus of tokenized documents. | https://www.tensortonic.com/study-plans/cracking-nlp/nlp-bag-of-words |
| Named Entity Recognition | Tag tokenized sentences with dictionary-backed named entity labels while preserving sentence and token order. | https://www.tensortonic.com/study-plans/cracking-nlp/nlp-ner |
| POS Tagging | Assign part-of-speech tags using each word's most frequent training tag and a deterministic fallback for unseen words. | https://www.tensortonic.com/study-plans/cracking-nlp/nlp-pos-tagging |
| Perform Stable Exact Deduplication | Normalize document text, retain the earliest exact occurrence, and map every removed document to its retained owner. | https://www.tensortonic.com/study-plans/language-modeling-from-scratch/cs336-l14-stable-exact-deduplication |
| Implement Dot Product | Compute the algebraic dot product and geometric angle relationship for two equal-length NumPy vectors. | https://www.tensortonic.com/study-plans/math-linear-algebra/la-dot-product |
| Matrix Transpose | Transpose a rectangular NumPy matrix by swapping its row and column axes without changing element values. | https://www.tensortonic.com/study-plans/math-linear-algebra/la-matrix-transpose |
| Convexity Certificate via the Hessian | Certify convexity of a twice-differentiable objective by checking whether its Hessian is positive semidefinite. | https://www.tensortonic.com/study-plans/math-optimization/optim-convexity-certificate |
| Minimum of an Axis-Aligned Paraboloid | Compute the coordinates and objective value at the minimum of an axis-aligned convex paraboloid. | https://www.tensortonic.com/study-plans/math-optimization/optim-paraboloid-minimum |
| Minimum of a Univariate Quadratic | Compute the minimizer and minimum value of a strictly convex univariate quadratic from its coefficients. | https://www.tensortonic.com/study-plans/math-optimization/optim-quadratic-minimum |
| Aggregation Functions | Compute selected NumPy aggregation functions globally or along a requested axis using float64 values. | https://www.tensortonic.com/study-plans/numpy-basics/numpy-aggregation |
| Angle Features | Return a float64 array where row 0 contains the sine values, row 1 the cosine values, and row 2 the tangent values. | https://www.tensortonic.com/study-plans/numpy-basics/numpy-angle-features |
| Arange and Linspace | Generate a one-dimensional NumPy sequence using either step-based arange or count-based linspace semantics. | https://www.tensortonic.com/study-plans/numpy-basics/numpy-arange-linspace |
| Basic Indexing | Extract a rectangular NumPy subarray with row and column slice boundaries using standard basic indexing. | https://www.tensortonic.com/study-plans/numpy-basics/numpy-basic-indexing |
| Boolean Masking | Build three filtered views of a 2D array: an element-level boolean mask, rows kept when any element exceeds a threshold. | https://www.tensortonic.com/study-plans/numpy-basics/numpy-boolean-masking |
| Column Scaling | Scale every column of a NumPy matrix by its aligned weight through broadcasting, without explicit Python loops. | https://www.tensortonic.com/study-plans/numpy-basics/numpy-col-scaling |
| Concat and Correlate | Concatenate two 2-D arrays row-wise and return a (3, n, n) stack of Pearson correlation matrices: one for each input and one for the combined data. | https://www.tensortonic.com/study-plans/numpy-basics/numpy-concat-correlate |
| Create Arrays from Lists | Create NumPy arrays from Python lists with the requested dtype and return their values, shape, dimensions, and element count. | https://www.tensortonic.com/study-plans/numpy-basics/numpy-create-array |
| Fancy Indexing | Convert the data to float64 and return the array formed by selecting elements along that axis using integer array indexing. | https://www.tensortonic.com/study-plans/numpy-basics/numpy-fancy-indexing |
| Filter and Extract | Implement Filter and Extract, and apply a boolean mask to select values strictly greater than threshold. | https://www.tensortonic.com/study-plans/numpy-basics/numpy-filter-extract |
| Mutation Trap | Extract an independent NumPy row copy, mutate it safely, and verify that the original array remains unchanged. | https://www.tensortonic.com/study-plans/numpy-basics/numpy-mutation-trap |
| Normalized Difference | Use two 2D arrays a and b of the same shape and a scalar range [lo, hi], clip both arrays to [lo, hi], rescale each to [0, 1]. | https://www.tensortonic.com/study-plans/numpy-basics/numpy-norm-diff |
| Norm-Gated Linear Transform | Compute the linear transform Z = X @ W, then zero out every row of Z whose L2 norm is strictly below the threshold. | https://www.tensortonic.com/study-plans/numpy-basics/numpy-norm-gate |
| Normalize Columns | Standardize each NumPy matrix column by subtracting its mean and dividing by its population standard deviation. | https://www.tensortonic.com/study-plans/numpy-basics/numpy-normalize-columns |
| Outer Sum | Compute the broadcasted outer sum of two NumPy vectors without loops, supporting different lengths and numeric values. | https://www.tensortonic.com/study-plans/numpy-basics/numpy-outer-sum |
| Pairwise Differences | Implement Pairwise Differences, and compute the pairwise difference matrix without any Python loops. | https://www.tensortonic.com/study-plans/numpy-basics/numpy-pairwise-diff |
| Quantize and Frame | Apply floor, ceiling, and nearest rounding to a NumPy matrix, then add a zero-valued border around each result. | https://www.tensortonic.com/study-plans/numpy-basics/numpy-quantize-frame |
| Random Array Generation | Generate seeded float64 NumPy arrays from either a uniform or standard normal distribution. | https://www.tensortonic.com/study-plans/numpy-basics/numpy-random-arrays |
| Reshaping Arrays | Transform a float64 NumPy array with flattening, transposition, or a validated target shape. | https://www.tensortonic.com/study-plans/numpy-basics/numpy-reshape |
| Row Extremes | Implement Row Extremes, using np.argmax(axis=1) to find the column index of the maximum value in each row. | https://www.tensortonic.com/study-plans/numpy-basics/numpy-row-extremes |
| Row Scaling | Scale every row of a NumPy matrix by its aligned weight through broadcasting, without explicit Python loops. | https://www.tensortonic.com/study-plans/numpy-basics/numpy-row-scaling |
| Sort and Argsort | Return NumPy values sorted along a selected axis together with the indices that produce the same ordering. | https://www.tensortonic.com/study-plans/numpy-basics/numpy-sort-argsort |
| Tile and Diff | Tile a 2-D array vertically and return the tiled result alongside its row-wise finite differences, packed as a (2, m·reps, n) float64 array. | https://www.tensortonic.com/study-plans/numpy-basics/numpy-tile-diff |
| Winsorize | Winsorization clips extreme values in each column to percentile-based bounds, a standard technique for suppressing outliers in ML preprocessing. | https://www.tensortonic.com/study-plans/numpy-basics/numpy-winsorize |
| Zeros and Ones | Create a two-dimensional float64 NumPy array of a requested shape filled entirely with zeros or ones. | https://www.tensortonic.com/study-plans/numpy-basics/numpy-zeros-ones |
| Boolean Indexing | Filter pandas rows by a numeric column threshold and return the matching records with their original column order. | https://www.tensortonic.com/study-plans/pandas-basics/pandas-boolean-indexing |
| Change Data Types | Create a DataFrame, convert the specified column to the target type, and return the dtypes before and after conversion. | https://www.tensortonic.com/study-plans/pandas-basics/pandas-change-dtypes |
| Column Selection | Create a pandas DataFrame from dictionary data and extract one named column as an ordered list. | https://www.tensortonic.com/study-plans/pandas-basics/pandas-column-selection |
| Data Types Overview | Create a pandas DataFrame and report each column dtype together with counts for every unique dtype. | https://www.tensortonic.com/study-plans/pandas-basics/pandas-data-types |
| Drop Duplicates | Create a DataFrame, remove duplicate rows, and return the cleaned result along with counts of rows before and after deduplication. | https://www.tensortonic.com/study-plans/pandas-basics/pandas-drop-duplicates |
| GroupBy Basics | Create a DataFrame and compute the sum, mean, and count of the value column for each group. | https://www.tensortonic.com/study-plans/pandas-basics/pandas-groupby-basics |
| Handle Missing Values | Create a pandas DataFrame, count missing entries per column, and replace every null with a supplied fill value. | https://www.tensortonic.com/study-plans/pandas-basics/pandas-handle-missing |
| Head and Tail Operations | Create a pandas DataFrame and return the requested first and last rows as record-oriented dictionaries. | https://www.tensortonic.com/study-plans/pandas-basics/pandas-head-tail |
| Inspect DataFrame Shape | Create a DataFrame and return its structural properties: row count, column count, column names, data types, and total number of values. | https://www.tensortonic.com/study-plans/pandas-basics/pandas-inspect-shape |
| Loc vs iLoc | Create a DataFrame and use positional indexing to extract: the single element, the full row, and the full column. | https://www.tensortonic.com/study-plans/pandas-basics/pandas-loc-iloc |
| Multi-Column Selection | Create a pandas DataFrame and select an ordered subset of named columns without changing row order. | https://www.tensortonic.com/study-plans/pandas-basics/pandas-multi-column-selection |
| Create DataFrame from Dict | Create a pandas DataFrame from dictionary data and report its records, shape, and ordered column names. | https://www.tensortonic.com/study-plans/pandas-basics/pandas-read-csv |
| Rename Columns | Rename selected pandas DataFrame columns from an old-to-new mapping and return the updated records. | https://www.tensortonic.com/study-plans/pandas-basics/pandas-rename-columns |
| Replace Values | Create a DataFrame, replace all occurrences of the old value with the new value in the specified column, and count how many replacements were made. | https://www.tensortonic.com/study-plans/pandas-basics/pandas-replace-values |
| Resetting Index | Set a pandas column as the index, then restore the default integer index while retaining the original values. | https://www.tensortonic.com/study-plans/pandas-basics/pandas-resetting-index |
| Setting Index | Set a named pandas DataFrame column as the index and report the resulting records and index metadata. | https://www.tensortonic.com/study-plans/pandas-basics/pandas-setting-index |
| Activation Functions | Implement four common activation functions from scratch using basic PyTorch tensor operations (no torch.nn module). | https://www.tensortonic.com/study-plans/pytorch-basics/pytorch-activation-function-from-scratch |
| Attention Mechanism from Scratch | Implement the scaled dot-product attention mechanism, a core building block of the Transformer architecture. | https://www.tensortonic.com/study-plans/pytorch-basics/pytorch-attention-from-scratch |
| Balanced DataLoader | Build a PyTorch DataLoader that balances class sampling with per-example weights derived from label frequencies. | https://www.tensortonic.com/study-plans/pytorch-basics/pytorch-balanced-dataloader |
| Basic Autograd | Use PyTorch autograd to evaluate a scalar function and return its derivative at every supplied input value. | https://www.tensortonic.com/study-plans/pytorch-basics/pytorch-basic-autograd |
| Batch Normalization | Normalize each feature across the batch, then scale and shift using learnable parameters. | https://www.tensortonic.com/study-plans/pytorch-basics/pytorch-batch-normalization |
| Beam Search Decoding | Beam search is a heuristic search algorithm used in sequence generation tasks such as machine translation, text generation, and speech recognition. | https://www.tensortonic.com/study-plans/pytorch-basics/pytorch-beam-search |
| Simple Neural Network | Implement a class SimpleNet subclassing nn.Module with two linear layers and ReLU between them. | https://www.tensortonic.com/study-plans/pytorch-basics/pytorch-build-simple-nn-from-scratch |
| Conv2d from Scratch | Implement a PyTorch Conv2d module from tensor operations with configurable channels, kernel, stride, padding, and bias. | https://www.tensortonic.com/study-plans/pytorch-basics/pytorch-conv2d-from-scratch |
| Custom Dataset Class | Implement a PyTorch Dataset over row records with indexed feature tensors and labels. | https://www.tensortonic.com/study-plans/pytorch-basics/pytorch-custom-dataclass |
| Custom Linear Layer | Implement a custom linear layer that computes the affine transformation without using any built-in linear layer. | https://www.tensortonic.com/study-plans/pytorch-basics/pytorch-custom-linear-layer |
| Custom SGD with Momentum | Implement momentum SGD by subclassing the PyTorch optimizer interface and maintaining per-parameter velocity. | https://www.tensortonic.com/study-plans/pytorch-basics/pytorch-custom-optimizer |
| Dropout from Scratch | Implement PyTorch inverted dropout from a supplied mask during training while returning inputs unchanged in evaluation mode. | https://www.tensortonic.com/study-plans/pytorch-basics/pytorch-dropout-from-scratch |
| Early Stopping | Train a PyTorch model with validation monitoring and stop after the configured number of unimproved epochs. | https://www.tensortonic.com/study-plans/pytorch-basics/pytorch-early-stopping |
| Gradient Accumulation | Simulate gradient accumulation over multiple micro-batches, and return the final weights and last averaged gradient. | https://www.tensortonic.com/study-plans/pytorch-basics/pytorch-gradient-accumulation |
| Loss Functions | Implement three common loss functions from scratch using PyTorch tensor operations: mean squared error, cross-entropy, and Huber loss. | https://www.tensortonic.com/study-plans/pytorch-basics/pytorch-loss-functions |
| LSTM Cell from Scratch | Implement a single LSTM (Long Short-Term Memory) cell that processes one time step of input. | https://www.tensortonic.com/study-plans/pytorch-basics/pytorch-lstm-cell-from-scratch |
| Manual Weight Update | Perform a PyTorch training step with manual parameter updates after backpropagation, without an optimizer object. | https://www.tensortonic.com/study-plans/pytorch-basics/pytorch-manual-weight-update |
| Masked Causal Attention | Implement scaled dot-product attention with a causal mask that prevents each position from attending to future positions. | https://www.tensortonic.com/study-plans/pytorch-basics/pytorch-masked-causal-attention |
| Mini Training Loop | Run one complete PyTorch training epoch over a DataLoader and return the sample-weighted mean loss. | https://www.tensortonic.com/study-plans/pytorch-basics/pytorch-mini-training |
| Multi-Head Attention | Implement PyTorch multi-head attention with head splitting, scaled softmax attention, concatenation, and output projection. | https://www.tensortonic.com/study-plans/pytorch-basics/pytorch-multi-head-attention |
| Optimizer Scheduler | Train with a PyTorch optimizer and StepLR schedule, recording the learning rate applied at each epoch. | https://www.tensortonic.com/study-plans/pytorch-basics/pytorch-optimizer-scheduler |
| Residual Block | Implement a PyTorch residual block with two padded convolutions, batch normalization, ReLU, and an identity shortcut. | https://www.tensortonic.com/study-plans/pytorch-basics/pytorch-residual-block |
| RNN Cell from Scratch | Implement a vanilla PyTorch RNN cell that combines current inputs and previous hidden states with a tanh update. | https://www.tensortonic.com/study-plans/pytorch-basics/pytorch-rnn-cell-from-scratch |
| Softmax from Scratch | Implement numerically stable batched softmax in PyTorch by shifting logits before exponentiation and normalization. | https://www.tensortonic.com/study-plans/pytorch-basics/pytorch-softmax-from-scratch |
| Tensor Operations | Perform common element-wise and matrix tensor operations: add, multiply, matmul, power, and max. | https://www.tensortonic.com/study-plans/pytorch-basics/pytorch-tensor-arithmetic |
| Tensor Factory | Create PyTorch tensors with zeros, ones, or a constant fill value using the requested shape and dtype. | https://www.tensortonic.com/study-plans/pytorch-basics/pytorch-tensor-creation |
| Tensor Shape Manipulation | Reshape tensors using three common PyTorch operations: flatten to collapse into 1D, squeeze to remove size-1 dimensions. | https://www.tensortonic.com/study-plans/pytorch-basics/pytorch-tensor-reshape |
| Transform Pipeline | Implement a callable class that converts a raw image tensor into a normalized, channel-first tensor ready for a neural network. | https://www.tensortonic.com/study-plans/pytorch-basics/pytorch-transforms-pipeline |
| Weight Initialization | Implement a function that initializes a weight tensor using one of four standard initialization methods. | https://www.tensortonic.com/study-plans/pytorch-basics/pytorch-weight-initialization |
| Basic SELECT | Write a SQL SELECT query that aliases product names and calculates inventory value from unit price and stock quantity. | https://www.tensortonic.com/study-plans/sql-basics/sql/sql-basic-select |
| Fused Multiply-Add | Implement a Triton fused multiply-add kernel with contiguous tiles, hardware FMA, and masked tail handling. | https://www.tensortonic.com/study-plans/triton-basics/triton/triton-fused-multiply-add |
| GELU | Implement exact GELU activation in Triton with device error-function math and masked contiguous tiles. | https://www.tensortonic.com/study-plans/triton-basics/triton/triton-gelu |
| Vector Max Reduction | Compute a vector maximum with one Triton reduction program and masked tail lanes that cannot win comparisons. | https://www.tensortonic.com/study-plans/triton-basics/triton/triton-max |
| Single-Pass Mean and Variance | Compute population mean and variance in Triton with single-pass statistics, atomic partials, and masked tails. | https://www.tensortonic.com/study-plans/triton-basics/triton/triton-mean-variance |
| ReLU | Implement ReLU activation in Triton with contiguous program tiles, branch-free rectification, and masked tails. | https://www.tensortonic.com/study-plans/triton-basics/triton/triton-relu |
| SiLU | Implement fused SiLU or Swish activation in Triton with contiguous tiles, sigmoid weighting, and masked tails. | https://www.tensortonic.com/study-plans/triton-basics/triton/triton-silu |
| Vector Sum Reduction | Implement tiled vector sum reduction in Triton with register partials, atomic accumulation, and masked tail lanes. | https://www.tensortonic.com/study-plans/triton-basics/triton/triton-sum |
| Vector Addition | Implement elementwise vector addition in Triton with contiguous program tiles and safe masking for partial tails. | https://www.tensortonic.com/study-plans/triton-basics/triton/triton-vector-addition |

View my verified ML profile: [TensorTonic profile](https://www.tensortonic.com/profile/uks007)
<!-- tensortonic:end -->
