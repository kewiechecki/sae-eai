#!/usr/bin/env python3
"""Test script for SAE feature interpretability."""

import sys
import os

import torch as t
from featurized import Featurized
from neuron_explainer.activations.sae_activations import (
    SAEActivationExtractor,
    SAEFeatureRecord,
    SAEFeatureId,
    SAEActivationRecord,
)
from neuron_explainer.activations.sae_activation_records import (
    format_sae_activation_records,
    calculate_max_sae_activation,
    get_activation_statistics,
)


def test_basic_extraction():
    """Test basic SAE feature extraction."""
    print("Testing basic SAE feature extraction...")
    
    # Initialize the model
    model_path = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    sae_path = "EleutherAI/sae-DeepSeek-R1-Distill-Qwen-1.5B-65k"
    
    print(f"Loading model: {model_path}")
    print(f"Loading SAE: {sae_path}")
    
    featurized = Featurized(model_path, sae_path)
    extractor = SAEActivationExtractor(featurized)
    
    # Test texts
    test_texts = [
        "The patient presents with fever and headache.",
        "Machine learning models can be interpretable.",
        "Python is a programming language.",
    ]
    
    layer_idx = 5  # Middle layer
    
    for text in test_texts:
        print(f"\n\nAnalyzing: '{text}'")
        print("-" * 50)
        
        # Get top activating features
        top_features = extractor.get_top_activating_features(
            text, layer_idx, n_features=5
        )
        
        print(f"Top 5 activating SAE features at layer {layer_idx}:")
        for idx, (feature_idx, max_act) in enumerate(top_features, 1):
            print(f"  {idx}. Feature {feature_idx}: max activation = {max_act:.4f}")
        
        # Get detailed activations for the top feature
        if top_features:
            top_feature_idx = top_features[0][0]
            print(f"\nDetailed activations for top feature {top_feature_idx}:")
            
            activation_record = extractor.create_activation_record(
                text, layer_idx, top_feature_idx, top_k=10
            )
            
            # Show token-by-token activations
            for token, activation in zip(activation_record.tokens[:10], 
                                        activation_record.feature_activations[:10]):
                if activation > 0:
                    print(f"  Token: '{token}' -> activation: {activation:.4f}")


def test_activation_records():
    """Test SAE activation record formatting."""
    print("\n\nTesting SAE activation record formatting...")
    print("=" * 50)
    
    # Create sample activation records
    record1 = SAEActivationRecord(
        tokens=["The", "cat", "sat", "on", "the", "mat"],
        feature_activations=[0.0, 2.5, 1.2, 0.0, 0.0, 3.1],
        feature_indices=[42] * 6,
        layer_index=5
    )
    
    record2 = SAEActivationRecord(
        tokens=["Machine", "learning", "is", "powerful"],
        feature_activations=[1.5, 4.2, 0.0, 2.8],
        feature_indices=[42] * 4,
        layer_index=5
    )
    
    records = [record1, record2]
    
    # Calculate max activation
    max_activation = calculate_max_sae_activation(records)
    print(f"Max activation across records: {max_activation:.4f}")
    
    # Format records
    formatted = format_sae_activation_records(
        records,
        max_activation,
        omit_zeros=False,
        include_feature_info=True
    )
    
    print("\nFormatted activation records:")
    print(formatted)
    
    # Get statistics
    stats = get_activation_statistics(records)
    print("\nActivation statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value:.4f}")


def test_feature_comparison():
    """Test comparing multiple SAE features."""
    print("\n\nTesting SAE feature comparison...")
    print("=" * 50)
    
    model_path = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    sae_path = "EleutherAI/sae-DeepSeek-R1-Distill-Qwen-1.5B-65k"
    
    featurized = Featurized(model_path, sae_path)
    extractor = SAEActivationExtractor(featurized)
    
    text = "The patient has symptoms of fever, cough, and fatigue."
    layer_idx = 5
    
    # Get top 3 features
    top_features = extractor.get_top_activating_features(
        text, layer_idx, n_features=3
    )
    
    print(f"Comparing top 3 features for text: '{text}'")
    print(f"Layer: {layer_idx}")
    print("-" * 50)
    
    for feature_idx, max_act in top_features:
        print(f"\nFeature {feature_idx} (max activation: {max_act:.4f}):")
        
        # Get activation record
        record = extractor.create_activation_record(
            text, layer_idx, feature_idx, top_k=10
        )
        
        # Show active tokens
        active_tokens = []
        for token, activation in zip(record.tokens, record.feature_activations):
            if activation > 0:
                active_tokens.append(f"{token}({activation:.2f})")
        
        if active_tokens:
            print(f"  Active tokens: {', '.join(active_tokens)}")
        else:
            print(f"  No active tokens")


def main():
    """Run all tests."""
    print("SAE Feature Interpretability Test Suite")
    print("=" * 60)
    
    try:
        test_basic_extraction()
        test_activation_records()
        test_feature_comparison()
        
        print("\n\n" + "=" * 60)
        print("All tests completed successfully!")
        
    except Exception as e:
        print(f"\n\nError during testing: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
