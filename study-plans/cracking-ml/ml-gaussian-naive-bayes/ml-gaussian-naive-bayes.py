import numpy as np

class GaussianNaiveBayesClassifier:
    def __init__(self,epsilon=1e-9):
        self.classes = None
        self.class_priors = None
        self.feature_means = None
        self.feature_vars = None
        self.epsilon = epsilon

    def train(self, X_train, y_train):
        X_train = np.asarray(X_train, dtype=np.float64)
        y_train = np.asarray(y_train)
        n_samples, n_features = X_train.shape

        self.classes = np.unique(y_train)
        self.class_priors = {}
        self.feature_means = {}
        self.feature_vars = {}
        # For each class and each feature, estimate the mean and variance from (X_train, y_train)
        for cls in self.classes:
            X_cls = X_train[y_train == cls]
            self.class_priors[cls] = len(X_cls) / n_samples
            self.feature_means[cls] = np.mean(X_cls,axis=0)
            # small epsilon for numerical stability
            self.feature_vars[cls] = (np.var(X_cls, axis=0) + self.epsilon)

    def get_log_gaussian_pdf(self, x, mean, var):
        return (-0.5 * np.log(2 * np.pi * var)- ((x - mean) ** 2) / (2 * var))

    def test(self, X_test):
        X_test = np.asarray(X_test, dtype=np.float64)
        predictions = []
        for x in X_test:
            # For each test point, compute log-posterior for each class using log-sum of Gaussian log-likelihoods plus log-prior
            class_scores = []
            for cls in self.classes:
                log_prior = np.log(self.class_priors[cls])
                log_likelihood = np.sum(
                    self.get_log_gaussian_pdf(
                        x,self.feature_means[cls],
                        self.feature_vars[cls]
                    )
                )
                class_scores.append(log_prior + log_likelihood)
            # Predict the class with the highest log-posterior
            predictions.append(self.classes[np.argmax(class_scores)])

        return predictions
    
def gaussian_nb(X_train, y_train, X_test):
    """
    Returns: A list of predicted integer labels for each test point
    """
    classifer = GaussianNaiveBayesClassifier()
    classifer.train(X_train, y_train)
    return classifer.test(X_test)
