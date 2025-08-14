"""SAE-specific activation handling for interpretability."""

import math
from dataclasses import dataclass, field
from typing import List, Optional, Union, Tuple
import torch as t
import urllib.request
import blobfile as bf
import boostedblob as bbb
from neuron_explainer.fast_dataclasses import FastDataclass, loads, register_dataclass
from neuron_explainer.azure import standardize_azure_url


@register_dataclass
@dataclass
class SAEActivationRecord(FastDataclass):
    """Collated lists of tokens and their SAE feature activations."""
    
    tokens: List[str]
    """Tokens in the text sequence, represented as strings."""
    feature_activations: List[float]
    """SAE feature activation values for each token in the text sequence."""
    feature_indices: List[int]
    """Indices of the active SAE features for each token."""
    layer_index: int
    """The layer index where SAE features were extracted."""


@register_dataclass
@dataclass
class SAEFeatureId(FastDataclass):
    """Identifier for a SAE feature in the model."""
    
    layer_index: int
    """The index of layer the SAE is applied to."""
    feature_index: int
    """The feature's index within the SAE (0 to n_features-1)."""


@register_dataclass
@dataclass
class SAEFeatureRecord(FastDataclass):
    """SAE feature-indexed activation data with summary stats and notable activation records."""
    
    feature_id: SAEFeatureId
    """Identifier for the SAE feature."""
    
    random_sample: list[SAEActivationRecord] = field(default_factory=list)
    """Random activation records for this SAE feature."""
    
    random_sample_by_quantile: Optional[list[list[SAEActivationRecord]]] = None
    """Random samples of activation records in each of the specified quantiles."""
    
    quantile_boundaries: Optional[list[float]] = None
    """Boundaries of the quantiles used to generate the random_sample_by_quantile field."""
    
    # Moments of activations
    mean: Optional[float] = math.nan
    variance: Optional[float] = math.nan
    skewness: Optional[float] = math.nan
    kurtosis: Optional[float] = math.nan
    sparsity: Optional[float] = math.nan  # Fraction of tokens where feature is active
    
    most_positive_activation_records: list[SAEActivationRecord] = field(default_factory=list)
    """Activation records with the highest SAE feature activations."""
    
    @property
    def max_activation(self) -> float:
        """Return the maximum activation value over all top-activating activation records."""
        if not self.most_positive_activation_records:
            return 0.0
        return max([max(ar.feature_activations) for ar in self.most_positive_activation_records])
    
    def _get_slices_for_splits(
        self,
        splits: list[str],
        num_activation_records_per_split: int,
    ) -> dict[str, slice]:
        """Get equal-sized interleaved subsets for each split."""
        stride = len(splits)
        num_activation_records_for_even_splits = num_activation_records_per_split * stride
        slices_by_split = {
            split: slice(split_index, num_activation_records_for_even_splits, stride)
            for split_index, split in enumerate(splits)
        }
        return slices_by_split
    
    def train_activation_records(
        self,
        n_examples_per_split: Optional[int] = None,
    ) -> list[SAEActivationRecord]:
        """Train split for generating explanations."""
        splits = ["train", "calibration", "valid", "test"]
        if n_examples_per_split is None:
            n_examples_per_split = len(self.most_positive_activation_records) // len(splits)
        slices = self._get_slices_for_splits(splits, n_examples_per_split)
        return self.most_positive_activation_records[slices["train"]]
    
    def valid_activation_records(
        self,
        n_examples_per_split: Optional[int] = None,
    ) -> list[SAEActivationRecord]:
        """Validation split for evaluating explanations."""
        splits = ["train", "calibration", "valid", "test"]
        if n_examples_per_split is None:
            n_examples_per_split = len(self.most_positive_activation_records) // len(splits)
        top_slices = self._get_slices_for_splits(splits, n_examples_per_split)
        
        random_splits = ["calibration", "valid", "test"]
        if n_examples_per_split is None:
            n_examples_per_split = len(self.random_sample) // len(random_splits)
        random_slices = self._get_slices_for_splits(random_splits, n_examples_per_split)
        
        return (
            self.most_positive_activation_records[top_slices["valid"]]
            + self.random_sample[random_slices["valid"]]
        )


class SAEActivationExtractor:
    """Extract SAE feature activations from a model using the Featurized interface."""
    
    def __init__(self, featurized_model):
        """
        Initialize with a Featurized model instance.
        
        Args:
            featurized_model: Instance of Featurized class from featurized.py
        """
        self.model = featurized_model
    
    def extract_feature_activations(
        self,
        text: str,
        layer_idx: int,
        top_k: int = 10
    ) -> Tuple[List[str], t.Tensor, t.Tensor]:
        """
        Extract SAE feature activations for a given text at a specific layer.
        
        Args:
            text: Input text to analyze
            layer_idx: Layer index to extract features from
            top_k: Number of top features to keep per token
            
        Returns:
            Tuple of (tokens, top_acts, top_indices)
        """
        # Tokenize the input
        inputs = self.model.tokenize(text)
        
        # Get SAE features for the specified layer
        features = self.model.embed_layer(layer_idx, inputs)
        
        # Get tokens as strings
        token_ids = inputs['input_ids'][0].cpu().tolist()
        tokens = [self.model.tokenizer.decode([tid]) for tid in token_ids]
        
        # features should have top_acts and top_indices attributes
        return tokens, features.top_acts, features.top_indices
    
    def create_activation_record(
        self,
        text: str,
        layer_idx: int,
        feature_idx: int,
        top_k: int = 10
    ) -> SAEActivationRecord:
        """
        Create a SAEActivationRecord for a specific feature.
        
        Args:
            text: Input text
            layer_idx: Layer index
            feature_idx: Specific SAE feature index to track
            top_k: Number of top features per token
            
        Returns:
            SAEActivationRecord with activations for the specified feature
        """
        tokens, top_acts, top_indices = self.extract_feature_activations(text, layer_idx, top_k)
        
        # Extract activations for the specific feature
        feature_activations = []
        feature_indices_list = []
        
        for token_idx in range(len(tokens)):
            # Check if the feature is in the top-k for this token
            token_indices = top_indices[token_idx].cpu().tolist()
            token_acts = top_acts[token_idx].cpu().tolist()
            
            if feature_idx in token_indices:
                idx_pos = token_indices.index(feature_idx)
                feature_activations.append(token_acts[idx_pos])
            else:
                feature_activations.append(0.0)
            
            feature_indices_list.append(feature_idx)
        
        return SAEActivationRecord(
            tokens=tokens,
            feature_activations=feature_activations,
            feature_indices=feature_indices_list,
            layer_index=layer_idx
        )
    
    def get_top_activating_features(
        self,
        text: str,
        layer_idx: int,
        n_features: int = 5
    ) -> List[Tuple[int, float]]:
        """
        Get the top activating SAE features for a text.
        
        Args:
            text: Input text
            layer_idx: Layer index
            n_features: Number of top features to return
            
        Returns:
            List of (feature_index, max_activation) tuples
        """
        _, top_acts, top_indices = self.extract_feature_activations(text, layer_idx)
        
        # Aggregate max activations per feature across all tokens
        feature_max_acts = {}
        for token_idx in range(top_acts.shape[0]):
            indices = top_indices[token_idx].cpu().tolist()
            acts = top_acts[token_idx].cpu().tolist()
            
            for idx, act in zip(indices, acts):
                if idx not in feature_max_acts or act > feature_max_acts[idx]:
                    feature_max_acts[idx] = act
        
        # Sort by activation strength
        sorted_features = sorted(feature_max_acts.items(), key=lambda x: x[1], reverse=True)
        return sorted_features[:n_features]


def load_sae_feature(
    layer_index: Union[str, int],
    feature_index: Union[str, int],
    dataset_path: str = "sae_features"
) -> SAEFeatureRecord:
    """Load the SAEFeatureRecord for the specified feature."""
    file_path = bf.join(dataset_path, str(layer_index), f"{feature_index}.json")
    if bf.exists(file_path):
        with bf.BlobFile(file_path, "r") as f:
            feature_record = loads(f.read())
            if not isinstance(feature_record, SAEFeatureRecord):
                raise ValueError(
                    f"Stored data incompatible with current version of SAEFeatureRecord dataclass."
                )
            return feature_record
    else:
        # Return empty record if not found
        return SAEFeatureRecord(
            feature_id=SAEFeatureId(
                layer_index=int(layer_index),
                feature_index=int(feature_index)
            )
        )


@bbb.ensure_session
async def load_sae_feature_async(
    layer_index: Union[str, int],
    feature_index: Union[str, int],
    dataset_path: str = "sae_features"
) -> SAEFeatureRecord:
    """Async version of load_sae_feature."""
    file = bf.join(dataset_path, str(layer_index), f"{feature_index}.json")
    if await bbb.exists(file):
        raw_contents = await bbb.read.read_single(file)
        feature_record = loads(raw_contents.decode("utf-8"))
        if not isinstance(feature_record, SAEFeatureRecord):
            raise ValueError(
                f"Stored data incompatible with current version of SAEFeatureRecord dataclass."
            )
        return feature_record
    else:
        return SAEFeatureRecord(
            feature_id=SAEFeatureId(
                layer_index=int(layer_index),
                feature_index=int(feature_index)
            )
        )