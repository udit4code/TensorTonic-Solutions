import torch
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

def create_balanced_loader(features, labels, batch_size):
    """
    Returns:
        A DataLoader that oversamples underrepresented classes using
        WeightedRandomSampler.
    """

    # Combine the feature and label tensors into a single dataset.
    dataset = TensorDataset(features, labels)

    # Count how many samples belong to each class.
    # Example:
    # labels = [0,0,0,1,1]
    #
    # class_counts =
    # tensor([3,2])
    class_counts = torch.bincount(labels)

    # Compute a weight for each class.
    # Classes with fewer samples receive larger weights so that they
    # are sampled more frequently.
    #
    # Example:
    #
    # class_counts = [900,100]
    #
    # class_weights =
    # [1/900, 1/100]
    class_weights = 1.0 / class_counts.float()

    # Assign every sample the weight corresponding to its class.
    # Example:
    # labels =
    # [0,0,1,1,0]
    # sample_weights =
    # [1/900,1/900,1/100,1/100,1/900]
    sample_weights = class_weights[labels]
    # Create a sampler that draws samples according to the computed sample weights.
    # Minority-class samples have larger weights and are therefore more likely to appear in each mini-batch.
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )
    # Create a DataLoader that uses the weighted sampler instead of the usual random shuffling.
    # Since the sampler already determines which samples are drawn, do NOT specify shuffle=True.
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler
    )

    return loader