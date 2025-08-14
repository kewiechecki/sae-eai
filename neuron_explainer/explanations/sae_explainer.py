"""Uses API calls to generate explanations of SAE feature behavior."""

from __future__ import annotations

import logging
import re
from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Optional, Sequence, Union

from neuron_explainer.activations.sae_activation_records import (
    calculate_max_sae_activation,
    format_sae_activation_records,
    non_zero_sae_activation_proportion,
)
from neuron_explainer.activations.sae_activations import SAEActivationRecord
from neuron_explainer.api_client import ApiClient
from neuron_explainer.explanations.few_shot_examples import FewShotExampleSet
from neuron_explainer.explanations.prompt_builder import (
    HarmonyMessage,
    PromptBuilder,
    PromptFormat,
    Role,
)

logger = logging.getLogger(__name__)


# Explanation prefix specific to SAE features
SAE_EXPLANATION_PREFIX = "this SAE feature activates for"


def _split_numbered_list(text: str) -> list[str]:
    """Split a numbered list into a list of strings."""
    lines = re.split(r"\n\d+\.", text)
    # Strip the leading whitespace from each line.
    return [line.lstrip() for line in lines]


def _remove_final_period(text: str) -> str:
    """Strip a final period or period-space from a string."""
    if text.endswith("."):
        return text[:-1]
    elif text.endswith(". "):
        return text[:-2]
    return text


class ContextSize(int, Enum):
    TWO_K = 2049
    FOUR_K = 4097
    EIGHT_K = 8193
    SIXTEEN_K = 16385

    @classmethod
    def from_int(cls, i: int) -> ContextSize:
        for context_size in cls:
            if context_size.value == i:
                return context_size
        raise ValueError(f"{i} is not a valid ContextSize")


HARMONY_V4_MODELS = ["gpt-3.5-turbo", "gpt-4", "gpt-4-1106-preview"]


class SAEFeatureExplainer(ABC):
    """
    Abstract base class for SAE Feature Explainer classes that generate explanations 
    from SAE feature activation data.
    """

    def __init__(
        self,
        model_name: str,
        prompt_format: PromptFormat = PromptFormat.HARMONY_V4,
        context_size: ContextSize = ContextSize.FOUR_K,
        max_concurrent: Optional[int] = 10,
        cache: bool = False,
    ):
        if prompt_format == PromptFormat.HARMONY_V4:
            assert model_name in HARMONY_V4_MODELS
        elif prompt_format in [PromptFormat.NONE, PromptFormat.INSTRUCTION_FOLLOWING]:
            assert model_name not in HARMONY_V4_MODELS
        else:
            raise ValueError(f"Unhandled prompt format {prompt_format}")

        self.model_name = model_name
        self.prompt_format = prompt_format
        self.context_size = context_size
        self.client = ApiClient(model_name=model_name, max_concurrent=max_concurrent, cache=cache)

    async def generate_explanations(
        self,
        *,
        num_samples: int = 5,
        max_tokens: int = 60,
        temperature: float = 1.0,
        top_p: float = 1.0,
        **prompt_kwargs: Any,
    ) -> list[Any]:
        """Generate explanations based on SAE feature activation data."""
        prompt = self.make_explanation_prompt(max_tokens_for_completion=max_tokens, **prompt_kwargs)

        generate_kwargs: dict[str, Any] = {
            "n": num_samples,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
        }

        if self.prompt_format == PromptFormat.HARMONY_V4:
            assert isinstance(prompt, list)
            assert isinstance(prompt[0], dict)  # Really a HarmonyMessage
            generate_kwargs["messages"] = prompt
        else:
            assert isinstance(prompt, str)
            generate_kwargs["prompt"] = prompt

        response = await self.client.make_request(**generate_kwargs)
        logger.debug("response in generate_explanations is %s", response)

        if self.prompt_format == PromptFormat.HARMONY_V4:
            explanations = [x["message"]["content"] for x in response["choices"]]
        elif self.prompt_format in [PromptFormat.NONE, PromptFormat.INSTRUCTION_FOLLOWING]:
            explanations = [x["text"] for x in response["choices"]]
        else:
            raise ValueError(f"Unhandled prompt format {self.prompt_format}")

        return self.postprocess_explanations(explanations, prompt_kwargs)

    @abstractmethod
    def make_explanation_prompt(self, **kwargs: Any) -> Union[str, list[HarmonyMessage]]:
        """
        Create a prompt to send to the API to generate one or more explanations.

        A prompt can be a simple string, or a list of HarmonyMessages, depending on the PromptFormat
        used by this instance.
        """
        ...

    def postprocess_explanations(
        self, completions: list[str], prompt_kwargs: dict[str, Any]
    ) -> list[Any]:
        """Postprocess the completions returned by the API into a list of explanations."""
        return completions  # no-op by default


class TokenActivationPairExplainer(SAEFeatureExplainer):
    """
    Generate explanations for SAE features by showing token-activation pairs.
    """

    def make_explanation_prompt(
        self,
        activation_records: Sequence[SAEActivationRecord],
        max_tokens_for_completion: int,
        **kwargs: Any,
    ) -> Union[str, list[HarmonyMessage]]:
        """
        Create a prompt for explaining a SAE feature based on its activation patterns.
        
        Args:
            activation_records: List of SAE activation records
            max_tokens_for_completion: Maximum tokens for the completion
        """
        max_activation = calculate_max_sae_activation(activation_records)
        sparsity = non_zero_sae_activation_proportion(activation_records, max_activation)
        
        # Build the prompt
        prompt_builder = PromptBuilder()
        
        # Add system message for Harmony V4 format
        if self.prompt_format == PromptFormat.HARMONY_V4:
            system_message = (
                "You are an expert at analyzing SAE (Sparse Autoencoder) features in neural networks. "
                "Your task is to explain what pattern or concept a particular SAE feature represents "
                "based on its activation patterns across different text samples."
            )
            prompt_builder.add_message(Role.SYSTEM, system_message)
        
        # Format activation records
        formatted_activations = format_sae_activation_records(
            activation_records,
            max_activation,
            omit_zeros=False,
            include_feature_info=True
        )
        
        # Build the main prompt
        user_message = f"""
Below are text samples with SAE feature activation scores (0-10 scale, where 10 is maximum activation).
The feature has a sparsity of {sparsity:.2%} (proportion of tokens where it activates).

{formatted_activations}

Based on these activation patterns, provide a concise explanation of what this SAE feature detects or represents.
Start your explanation with: "{SAE_EXPLANATION_PREFIX}"

Focus on identifying:
1. Common patterns or themes in high-activation contexts
2. Linguistic or semantic features being detected
3. Any structural patterns (syntax, grammar, etc.)
"""
        
        if self.prompt_format == PromptFormat.HARMONY_V4:
            prompt_builder.add_message(Role.USER, user_message)
            return prompt_builder.build(self.prompt_format)
        else:
            return user_message

    def postprocess_explanations(
        self, completions: list[str], prompt_kwargs: dict[str, Any]
    ) -> list[str]:
        """Clean up the generated explanations."""
        processed = []
        for completion in completions:
            # Remove the prefix if it's included
            if completion.startswith(SAE_EXPLANATION_PREFIX):
                completion = completion[len(SAE_EXPLANATION_PREFIX):].strip()
            
            # Clean up formatting
            completion = _remove_final_period(completion)
            processed.append(completion)
        
        return processed


class ListExplainer(SAEFeatureExplainer):
    """
    Generate multiple hypotheses about what a SAE feature represents.
    """

    def make_explanation_prompt(
        self,
        activation_records: Sequence[SAEActivationRecord],
        max_tokens_for_completion: int,
        num_hypotheses: int = 5,
        **kwargs: Any,
    ) -> Union[str, list[HarmonyMessage]]:
        """
        Create a prompt for generating multiple hypotheses about a SAE feature.
        
        Args:
            activation_records: List of SAE activation records
            max_tokens_for_completion: Maximum tokens for the completion
            num_hypotheses: Number of hypotheses to generate
        """
        max_activation = calculate_max_sae_activation(activation_records)
        
        # Build the prompt
        prompt_builder = PromptBuilder()
        
        # Add system message for Harmony V4 format
        if self.prompt_format == PromptFormat.HARMONY_V4:
            system_message = (
                "You are an expert at analyzing SAE (Sparse Autoencoder) features. "
                "Generate multiple hypotheses about what a SAE feature might represent."
            )
            prompt_builder.add_message(Role.SYSTEM, system_message)
        
        # Format activation records with less detail for list generation
        formatted_activations = format_sae_activation_records(
            activation_records[:10],  # Limit to first 10 for brevity
            max_activation,
            omit_zeros=True,  # Only show non-zero activations
            include_feature_info=False
        )
        
        # Build the main prompt
        user_message = f"""
Analyze these SAE feature activations and generate {num_hypotheses} distinct hypotheses about what this feature detects:

{formatted_activations}

Provide {num_hypotheses} numbered hypotheses, each on a new line. Each hypothesis should be concise (10-20 words).
Focus on different aspects: semantic meaning, syntactic patterns, contextual cues, etc.

Format:
1. [First hypothesis]
2. [Second hypothesis]
...
"""
        
        if self.prompt_format == PromptFormat.HARMONY_V4:
            prompt_builder.add_message(Role.USER, user_message)
            return prompt_builder.build(self.prompt_format)
        else:
            return user_message

    def postprocess_explanations(
        self, completions: list[str], prompt_kwargs: dict[str, Any]
    ) -> list[list[str]]:
        """Parse the numbered lists into individual hypotheses."""
        all_hypotheses = []
        for completion in completions:
            hypotheses = _split_numbered_list(completion)
            # Filter out empty strings
            hypotheses = [h.strip() for h in hypotheses if h.strip()]
            all_hypotheses.append(hypotheses)
        
        return all_hypotheses


class SAEFeatureComparator(SAEFeatureExplainer):
    """
    Compare and contrast multiple SAE features to understand their relationships.
    """

    def make_explanation_prompt(
        self,
        feature_records: dict[int, Sequence[SAEActivationRecord]],
        max_tokens_for_completion: int,
        **kwargs: Any,
    ) -> Union[str, list[HarmonyMessage]]:
        """
        Create a prompt for comparing multiple SAE features.
        
        Args:
            feature_records: Dictionary mapping feature indices to their activation records
            max_tokens_for_completion: Maximum tokens for the completion
        """
        prompt_builder = PromptBuilder()
        
        # Add system message for Harmony V4 format
        if self.prompt_format == PromptFormat.HARMONY_V4:
            system_message = (
                "You are an expert at analyzing and comparing SAE features. "
                "Identify similarities, differences, and relationships between features."
            )
            prompt_builder.add_message(Role.SYSTEM, system_message)
        
        # Format each feature's activations
        features_text = []
        for feature_idx, records in feature_records.items():
            max_activation = calculate_max_sae_activation(records)
            formatted = format_sae_activation_records(
                records[:5],  # Limit samples per feature
                max_activation,
                omit_zeros=True,
                include_feature_info=False
            )
            features_text.append(f"### Feature {feature_idx}:\n{formatted}")
        
        # Build the main prompt
        user_message = f"""
Compare these SAE features based on their activation patterns:

{chr(10).join(features_text)}

Provide:
1. What each feature appears to detect
2. Key similarities between features
3. Important differences
4. Potential relationships (e.g., hierarchical, complementary, overlapping)

Keep your analysis concise and focused on the most notable patterns.
"""
        
        if self.prompt_format == PromptFormat.HARMONY_V4:
            prompt_builder.add_message(Role.USER, user_message)
            return prompt_builder.build(self.prompt_format)
        else:
            return user_message