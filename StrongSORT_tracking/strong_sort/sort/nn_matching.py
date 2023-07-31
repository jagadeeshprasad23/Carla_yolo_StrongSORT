import numpy as np
import torch
from torchreid.reid.metrics.distance import compute_distance_matrix

def _pdist(a, b):
    
    a, b = np.asarray(a), np.asarray(b)
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    a2, b2 = np.square(a).sum(axis=1), np.square(b).sum(axis=1)
    r2 = -2. * np.dot(a, b.T) + a2[:, None] + b2[None, :]
    r2 = np.clip(r2, 0., float(np.inf))
    return r2

def _cosine_distance(a, b, data_is_normalized=False):
    
    if not data_is_normalized:
        a = np.asarray(a) / np.linalg.norm(a, axis=1, keepdims=True)
        b = np.asarray(b) / np.linalg.norm(b, axis=1, keepdims=True)
    return 1. - np.dot(a, b.T)

def _nn_euclidean_distance(x, y):
    
    x_ = torch.from_numpy(np.asarray(x) / np.linalg.norm(x, axis=1, keepdims=True))
    y_ = torch.from_numpy(np.asarray(y) / np.linalg.norm(y, axis=1, keepdims=True))
    distances = compute_distance_matrix(x_, y_, metric='euclidean')
    return np.maximum(0.0, torch.min(distances, axis=0)[0].numpy())

def _nn_cosine_distance(x, y):
    
    x_ = torch.from_numpy(np.asarray(x))
    y_ = torch.from_numpy(np.asarray(y))
    distances = compute_distance_matrix(x_, y_, metric='cosine')
    distances = distances.cpu().detach().numpy()
    return distances.min(axis=0)

class NearestNeighborDistanceMetric(object):

    def __init__(self, metric, matching_threshold, budget=None):
        if metric == "euclidean":
            self._metric = _nn_euclidean_distance
        elif metric == "cosine":
            self._metric = _nn_cosine_distance
        else:
            raise ValueError(
                "Invalid metric; must be either 'euclidean' or 'cosine'")
        self.matching_threshold = matching_threshold
        self.budget = budget
        self.samples = {}

    def partial_fit(self, features, targets, active_targets):
        
        for feature, target in zip(features, targets):
            self.samples.setdefault(target, []).append(feature)
            if self.budget is not None:
                self.samples[target] = self.samples[target][-self.budget:]
        self.samples = {k: self.samples[k] for k in active_targets}

    def distance(self, features, targets):
        
        cost_matrix = np.zeros((len(targets), len(features)))
        for i, target in enumerate(targets):
            cost_matrix[i, :] = self._metric(self.samples[target], features)
        return cost_matrix