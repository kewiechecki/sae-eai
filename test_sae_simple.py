#!/usr/bin/env python3
"""Simple test script for SAE feature extraction without external dependencies."""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch as t
from featurized import Featurized


def test_sae_feature_extraction():
    """Test basic SAE feature extraction using existing Featurized class."""
    print("Testing SAE Feature Extraction")
    print("=" * 60)
    
    # Force CPU usage to avoid CUDA OOM
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    
    # Initialize the model
    model_path = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    sae_path = "EleutherAI/sae-DeepSeek-R1-Distill-Qwen-1.5B-65k"
    
    print(f"Loading model: {model_path}")
    print(f"Loading SAE: {sae_path}")
    print("Using CPU to avoid memory issues...")
    
    featurized = Featurized(model_path, sae_path)
    
    # Test texts
    test_texts = [
        "The patient presents with fever and headache.",
        "Machine learning models can be interpretable.",
        "Python is a programming language.",
    ]
    
    # Test different layers
    test_layers = [5, 7, 10]
    
    for text in test_texts:
        print(f"\n\nAnalyzing: '{text}'")
        print("-" * 50)
        
        # Tokenize
        inputs = featurized.tokenize(text)
        token_ids = inputs['input_ids'][0].cpu().tolist()
        tokens = [featurized.tokenizer.decode([tid]) for tid in token_ids]
        print(f"Tokens ({len(tokens)}): {tokens}")
        
        for layer_idx in test_layers:
            print(f"\n  Layer {layer_idx}:")
            
            try:
                # Get SAE features
                features = featurized.embed_layer(layer_idx, inputs)
                
                # Features should have top_acts and top_indices
                if hasattr(features, 'top_acts') and hasattr(features, 'top_indices'):
                    top_acts = features.top_acts
                    top_indices = features.top_indices
                    
                    # Get top features across all tokens
                    all_indices = top_indices.flatten().cpu().tolist()
                    all_acts = top_acts.flatten().cpu().tolist()
                    
                    # Find unique features and their max activations
                    feature_max = {}
                    for idx, act in zip(all_indices, all_acts):
                        if idx not in feature_max or act > feature_max[idx]:
                            feature_max[idx] = act
                    
                    # Sort by activation
                    sorted_features = sorted(feature_max.items(), 
                                           key=lambda x: x[1], reverse=True)[:5]
                    
                    print(f"    Top 5 features:")
                    for feature_idx, max_act in sorted_features:
                        print(f"      Feature {feature_idx}: {max_act:.4f}")
                    
                    # Show per-token top features for first few tokens
                    print(f"    Per-token top features (first 3 tokens):")
                    for i in range(min(3, len(tokens))):
                        token_top_idx = top_indices[i][:3].cpu().tolist()
                        token_top_act = top_acts[i][:3].cpu().tolist()
                        
                        features_str = ", ".join([f"{idx}({act:.2f})" 
                                                 for idx, act in zip(token_top_idx, token_top_act)])
                        print(f"      '{tokens[i]}': {features_str}")
                else:
                    print(f"    Features object missing expected attributes")
                    
            except Exception as e:
                print(f"    Error processing layer {layer_idx}: {e}")


def test_feature_reconstruction():
    """Test reconstructing activations from SAE features."""
    print("\n\nTesting SAE Feature Reconstruction")
    print("=" * 60)
    
    model_path = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    sae_path = "EleutherAI/sae-DeepSeek-R1-Distill-Qwen-1.5B-65k"
    
    featurized = Featurized(model_path, sae_path)
    
    text = "The quick brown fox jumps over the lazy dog."
    layer_idx = 5
    
    print(f"Text: '{text}'")
    print(f"Layer: {layer_idx}")
    print("-" * 50)
    
    # Tokenize
    inputs = featurized.tokenize(text)
    
    # Get SAE features
    features = featurized.embed_layer(layer_idx, inputs)
    print(f"Extracted SAE features shape: top_acts={features.top_acts.shape}, top_indices={features.top_indices.shape}")
    
    # Try to reconstruct and get logits
    try:
        logits = featurized.unembed_layer(layer_idx, inputs, features)
        print(f"Reconstructed logits shape: {logits.shape}")
        
        # Get predictions
        predictions = t.argmax(logits[0, -1, :]).item()
        predicted_token = featurized.tokenizer.decode([predictions])
        print(f"Predicted next token: '{predicted_token}'")
        
    except Exception as e:
        print(f"Error during reconstruction: {e}")


def test_sparsity_analysis():
    """Analyze sparsity of SAE features."""
    print("\n\nTesting SAE Feature Sparsity")
    print("=" * 60)
    
    model_path = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    sae_path = "EleutherAI/sae-DeepSeek-R1-Distill-Qwen-1.5B-65k"
    
    featurized = Featurized(model_path, sae_path)
    
    texts = [
        "Medical diagnosis requires careful observation.",
        "The patient's symptoms indicate infection.",
        "Treatment options include antibiotics.",
    ]
    
    layer_idx = 7
    
    print(f"Analyzing sparsity at layer {layer_idx}")
    print("-" * 50)
    
    all_active_features = set()
    total_activations = 0
    non_zero_activations = 0
    
    for text in texts:
        inputs = featurized.tokenize(text)
        features = featurized.embed_layer(layer_idx, inputs)
        
        if hasattr(features, 'top_acts') and hasattr(features, 'top_indices'):
            # Count non-zero activations
            acts = features.top_acts.cpu()
            indices = features.top_indices.cpu()
            
            non_zero_mask = acts > 0
            non_zero_count = non_zero_mask.sum().item()
            total_count = acts.numel()
            
            total_activations += total_count
            non_zero_activations += non_zero_count
            
            # Collect unique active features
            active_indices = indices[non_zero_mask].tolist()
            all_active_features.update(active_indices)
    
    if total_activations > 0:
        sparsity = 1.0 - (non_zero_activations / total_activations)
        print(f"Overall sparsity: {sparsity:.2%}")
        print(f"Active features: {non_zero_activations}/{total_activations}")
        print(f"Unique active features across texts: {len(all_active_features)}")
        
        # Show distribution of active features
        if all_active_features:
            sorted_features = sorted(all_active_features)
            print(f"Feature index range: {min(sorted_features)} - {max(sorted_features)}")


def main():
    """Run all tests."""
    print("SAE Feature Analysis Test Suite")
    print("=" * 60)
    
    try:
        test_sae_feature_extraction()
        test_feature_reconstruction()
        test_sparsity_analysis()
        
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