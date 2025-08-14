"""Utilities for formatting SAE activation records into prompts."""

import math
from typing import Optional, Sequence

from neuron_explainer.activations.sae_activations import SAEActivationRecord

UNKNOWN_ACTIVATION_STRING = "unknown"


def relu(x: float) -> float:
    return max(0.0, x)


def calculate_max_sae_activation(activation_records: Sequence[SAEActivationRecord]) -> float:
    """Return the maximum SAE feature activation value across all the activation records."""
    flattened = [
        max(relu(x) for x in activation_record.feature_activations)
        for activation_record in activation_records
    ]
    if not flattened:
        return 0.0
    return max(flattened)


def normalize_sae_activations(activation_values: list[float], max_activation: float) -> list[int]:
    """Convert raw SAE feature activations to integers on the range [0, 10]."""
    if max_activation <= 0:
        return [0 for x in activation_values]
    # Apply relu and normalize to [0, 10] scale
    return [min(10, math.floor(10 * relu(x) / max_activation)) for x in activation_values]


def _format_sae_activation_record(
    activation_record: SAEActivationRecord,
    max_activation: float,
    omit_zeros: bool,
    hide_activations: bool = False,
    start_index: int = 0,
) -> str:
    """Format SAE feature activations into a string, suitable for use in prompts."""
    tokens = activation_record.tokens
    normalized_activations = normalize_sae_activations(
        activation_record.feature_activations, max_activation
    )
    
    if omit_zeros:
        assert (not hide_activations) and start_index == 0, "Can't hide activations and omit zeros"
        tokens = [
            token for token, activation in zip(tokens, normalized_activations) if activation > 0
        ]
        normalized_activations = [x for x in normalized_activations if x > 0]
    
    entries = []
    assert len(tokens) == len(normalized_activations)
    for index, token, activation in zip(range(len(tokens)), tokens, normalized_activations):
        activation_string = str(int(activation))
        if hide_activations or index < start_index:
            activation_string = UNKNOWN_ACTIVATION_STRING
        entries.append(f"{token}\t{activation_string}")
    return "\n".join(entries)


def format_sae_activation_records(
    activation_records: Sequence[SAEActivationRecord],
    max_activation: float,
    *,
    omit_zeros: bool = False,
    start_indices: Optional[list[int]] = None,
    hide_activations: bool = False,
    include_feature_info: bool = True,
) -> str:
    """
    Format a list of SAE activation records into a string.
    
    Args:
        activation_records: List of SAE activation records to format
        max_activation: Maximum activation value for normalization
        omit_zeros: Whether to omit tokens with zero activation
        start_indices: List of starting indices for each record
        hide_activations: Whether to hide activation values
        include_feature_info: Whether to include feature and layer info in output
    """
    formatted_records = []
    
    for i, activation_record in enumerate(activation_records):
        if include_feature_info and hasattr(activation_record, 'feature_indices'):
            # Add header with feature info
            feature_idx = activation_record.feature_indices[0] if activation_record.feature_indices else -1
            layer_idx = activation_record.layer_index
            header = f"# Layer {layer_idx}, Feature {feature_idx}\n"
        else:
            header = ""
        
        formatted = _format_sae_activation_record(
            activation_record,
            max_activation,
            omit_zeros=omit_zeros,
            hide_activations=hide_activations,
            start_index=0 if start_indices is None else start_indices[i],
        )
        formatted_records.append(header + formatted)
    
    return (
        "\n<start>\n"
        + "\n<end>\n<start>\n".join(formatted_records)
        + "\n<end>\n"
    )


def _format_tokens_for_sae_simulation(tokens: Sequence[str]) -> str:
    """
    Format tokens into a string with each token marked as having an "unknown" activation,
    suitable for use in SAE feature simulation prompts.
    """
    entries = []
    for token in tokens:
        entries.append(f"{token}\t{UNKNOWN_ACTIVATION_STRING}")
    return "\n".join(entries)


def format_sequences_for_sae_simulation(
    all_tokens: Sequence[Sequence[str]],
) -> str:
    """
    Format a list of lists of tokens into a string with each token marked as having an "unknown"
    activation, suitable for use in SAE feature simulation prompts.
    """
    return (
        "\n<start>\n"
        + "\n<end>\n<start>\n".join(
            [_format_tokens_for_sae_simulation(tokens) for tokens in all_tokens]
        )
        + "\n<end>\n"
    )


def non_zero_sae_activation_proportion(
    activation_records: Sequence[SAEActivationRecord], max_activation: float
) -> float:
    """Return the proportion of SAE feature activation values that aren't zero (sparsity metric)."""
    total_activations_count = sum(
        [len(activation_record.feature_activations) for activation_record in activation_records]
    )
    if total_activations_count == 0:
        return 0.0
    
    normalized_activations = [
        normalize_sae_activations(activation_record.feature_activations, max_activation)
        for activation_record in activation_records
    ]
    non_zero_activations_count = sum(
        [len([x for x in activations if x != 0]) for activations in normalized_activations]
    )
    return non_zero_activations_count / total_activations_count


def get_activation_statistics(activation_records: Sequence[SAEActivationRecord]) -> dict:
    """
    Calculate statistics for SAE feature activations.
    
    Returns:
        Dictionary with mean, variance, max, sparsity statistics
    """
    all_activations = []
    for record in activation_records:
        all_activations.extend(record.feature_activations)
    
    if not all_activations:
        return {
            "mean": 0.0,
            "variance": 0.0,
            "max": 0.0,
            "sparsity": 0.0,
        }
    
    import numpy as np
    activations_array = np.array(all_activations)
    
    return {
        "mean": float(np.mean(activations_array)),
        "variance": float(np.var(activations_array)),
        "max": float(np.max(activations_array)),
        "sparsity": float(np.mean(activations_array > 0)),  # Fraction of non-zero activations
    }