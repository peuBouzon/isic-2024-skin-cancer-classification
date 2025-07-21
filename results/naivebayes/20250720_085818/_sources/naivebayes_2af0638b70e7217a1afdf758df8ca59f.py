import torch
import numpy as np
from torch import nn
from imblearn.under_sampling import RandomUnderSampler

class NaiveBayes(nn.Module):
    def __init__(self, vision_model:nn.Module, 
                 num_class,
                 eps=1e-6,
                 laplacian_smoothing=1):
        super().__init__()
        self.n_classes = num_class
        self.laplacian_smoothing = laplacian_smoothing
        self.eps = eps
        self.vision_model = vision_model
        self.is_fit = False
        self.weights = None
        self.numerical_features = None
        self.categorical_features = None
        self.register_buffer('means', None)
        self.register_buffer('variances', None)
        self.register_buffer(f'log_frequency_table', None)

    def fit(self, metadata, labels, categorical_features, numerical_features, weights=None):
        self.numerical_features = numerical_features
        self.categorical_features = categorical_features

        self.means = torch.zeros((self.n_classes, len(numerical_features)))
        self.variances = torch.zeros((self.n_classes, len(numerical_features)))
        self.log_frequency_table = torch.zeros((self.n_classes, len(categorical_features)))

        self.weights = torch.ones((len(np.unique(labels)))) if weights is None else weights
        if numerical_features is not None and len(numerical_features) > 0:
            self.estimate_mean_var(metadata[:, numerical_features], labels)

        if categorical_features is not None and len(categorical_features) > 0:
            self.fill_log_frequency_table(metadata[:, categorical_features], labels)

        self.is_fit = True

    def estimate_mean_var(self, features, labels):
        for label in range(self.n_classes):
            x = features[(labels == label)]
            self.means[label, :] = torch.nanmean(x, dim=0)

            # fill nan values with mean to calculate variance:
            x_mean_fill = torch.where(torch.isnan(x), self.means[label, :].unsqueeze(0), x)
            self.variances[label, :] = torch.var(x_mean_fill, dim=0, unbiased=True) + self.eps

    def fill_log_frequency_table(self, features:torch.Tensor, labels):
        for label in range(self.n_classes):
            self.log_frequency_table[label, :] = (features[(labels == label)].sum(axis=0) + self.laplacian_smoothing) * self.weights[label]
        
        self.log_frequency_table = self.log_frequency_table / self.log_frequency_table.sum(axis=0)
        self.log_frequency_table = torch.log(self.log_frequency_table)

    def get_numerical_features_log_probs(self, batch:torch.Tensor):
        # log(normal_distribution) = -log(sqrt(2*pi*var)) -(x-u)²/2*var
        normalization_term = -torch.log(torch.sqrt(2*torch.pi*self.variances))
        log_probs = normalization_term - 0.5 * (batch[:, self.numerical_features].unsqueeze(dim=1) - self.means) ** 2 / self.variances
        # sum the log probs from features that are not NaN
        mask = ~torch.isnan(batch[:, self.numerical_features])
        mask_expanded = mask.unsqueeze(1).expand(-1, log_probs.size(1), -1)
        return torch.where(mask_expanded, log_probs, torch.zeros_like(log_probs)).sum(dim=2).float()

    def get_categorical_features_log_probs(self, batch:torch.Tensor):
        return torch.matmul((batch[:, self.categorical_features] == 1).float(), self.log_frequency_table.t())

    def forward(self, meta:torch.Tensor):
        if self.is_fit:
            #F.log_softmax(self.vision_model(img), dim=1)
            if self.categorical_features is not None and self.numerical_features is not None:
                return self.get_categorical_features_log_probs(meta) + self.get_numerical_features_log_probs(meta)
            if self.numerical_features is not None:
                return self.get_numerical_features_log_probs(meta)
            if self.categorical_features is not None:
                return self.get_categorical_features_log_probs(meta)
            return None
        else:
            raise Exception('The Naive Bayes was not fit.')
        


class NaiveBayesEnsemble:
    """
    An ensemble of NaiveBayes classifiers, each trained on a different
    randomly undersampled subset of the data (bagging).
    """
    def __init__(self, n_estimators=2, sampling_strategy=0.01, random_state=42):
        self.n_estimators = n_estimators
        self.sampling_strategy = sampling_strategy
        self.random_state = random_state
        self.models = []
        self.feature_indices = {}

    def fit(self, X, y, categorical_features, numerical_features, weights):
        """
        Fits n_estimators NaiveBayes models on different undersampled splits.
        """
        self.feature_indices = {
            'categorical': categorical_features,
            'numerical': numerical_features
        }
        
        # Ensure X and y are numpy arrays for the sampler
        if isinstance(X, torch.Tensor):
            X = X.cpu().numpy()
        if isinstance(y, torch.Tensor):
            y = y.cpu().numpy()

        for i in range(self.n_estimators):
            # Create a new sampler for each model to get a different random subset
            sampler = RandomUnderSampler(
                sampling_strategy=self.sampling_strategy,
                random_state=self.random_state + i
            )
            
            # Resample the data
            X_res, y_res = sampler.fit_resample(X, y)
            X_res, y_res = torch.tensor(X_res, dtype=torch.float32), torch.tensor(y_res, dtype=torch.long)
            
            # Create and train a new NaiveBayes model on the subset
            model = NaiveBayes(None, num_class=2, eps=1e-6, laplacian_smoothing=1)
            model.fit(X_res, y_res, categorical_features, numerical_features, weights=weights)
            self.models.append(model)

    def __call__(self, X_tensor):
        """
        Performs soft voting by averaging the probabilities from all models.
        """
        with torch.no_grad():
            # Collect probabilities from each model in the ensemble
            all_probs = [model(X_tensor) for model in self.models]
            
            # Stack and average the probabilities
            stacked_probs = torch.stack(all_probs)
            avg_probs = torch.mean(stacked_probs, dim=0)
            
        return avg_probs