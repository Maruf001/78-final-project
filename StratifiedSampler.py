from torch.utils.data import DataLoader, Dataset, Sampler, BatchSampler
import numpy as np
from collections import defaultdict


class StratifiedSampler(Sampler):
    def __init__(self, dataset, batch_size):
        self.dataset = dataset
        self.batch_size = batch_size

        self.labels_to_indices = defaultdict(list)
        for idx, (_, label) in enumerate(dataset):
            self.labels_to_indices[label].append(idx)

        self.unique_labels = list(self.labels_to_indices.keys())
        self.num_classes = len(self.unique_labels)
        assert (
            batch_size % self.num_classes == 0
        ), "Batch size should be a multiple of the number of classes"

        self.class_batch_size = batch_size // self.num_classes
        self._reset()

    def _reset(self):
        self.available_indices = {
            label: np.random.permutation(indices).tolist()
            for label, indices in self.labels_to_indices.items()
        }

    def __iter__(self):
        for i in range(len(self.dataset) // self.batch_size):
            batch_indices = []
            for label in self.unique_labels:
                start_idx = i * self.class_batch_size
                end_idx = (i + 1) * self.class_batch_size
                indices = self.available_indices[label][start_idx:end_idx]
                batch_indices.extend(indices)
            yield batch_indices

    def __len__(self):
        return len(self.dataset) // self.batch_size
