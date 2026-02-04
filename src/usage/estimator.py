"""
2api.ai - Token Estimation

Estimates token counts for requests before sending to providers.

Provides:
- Approximate token counting for text
- Message token estimation
- Image token estimation
- Tool token estimation
"""

import json
import math
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

from ..core.models import Message, Role


@dataclass
class TokenEstimate:
    """
    Token count estimate for a request.

    Includes breakdown by component type.
    """
    total_tokens: int
    text_tokens: int = 0
    message_overhead: int = 0
    image_tokens: int = 0
    tool_tokens: int = 0
    system_tokens: int = 0

    # Confidence level
    confidence: float = 0.9  # 0.0-1.0

    def with_margin(self, margin: float = 0.1) -> int:
        """Get estimate with safety margin."""
        return int(self.total_tokens * (1 + margin))


class TokenEstimator:
    """
    Estimates token counts for different content types.

    Uses heuristics since we don't have actual tokenizers.
    For production, consider using tiktoken for OpenAI models.
    """

    # Average characters per token by model family
    CHARS_PER_TOKEN = {
        "openai": 4.0,      # GPT models average ~4 chars/token
        "anthropic": 3.5,   # Claude slightly more efficient
        "google": 4.0,      # Gemini similar to GPT
        "default": 4.0
    }

    # Overhead tokens per message
    MESSAGE_OVERHEAD = {
        "openai": 4,        # <|start|>role<|end|>content<|end|>
        "anthropic": 3,
        "google": 3,
        "default": 4
    }

    # Base tokens for various components
    TOOL_DEFINITION_BASE = 10  # Base tokens per tool
    TOOL_PARAM_TOKENS = 5      # Per parameter
    FUNCTION_CALL_OVERHEAD = 10

    # Image token counts (OpenAI formula)
    IMAGE_TOKENS = {
        "low": 85,
        "high": 170,  # Base for high detail
        "high_tile": 170  # Per 512x512 tile
    }

    def __init__(self, provider: str = "default"):
        """
        Initialize estimator for a specific provider.

        Args:
            provider: Provider name for tuned estimates
        """
        self.provider = provider
        self.chars_per_token = self.CHARS_PER_TOKEN.get(
            provider,
            self.CHARS_PER_TOKEN["default"]
        )
        self.message_overhead = self.MESSAGE_OVERHEAD.get(
            provider,
            self.MESSAGE_OVERHEAD["default"]
        )

    def estimate_text_tokens(self, text: str) -> int:
        """
        Estimate tokens for plain text.

        Uses character-based heuristic with adjustments for:
        - Whitespace
        - Numbers
        - Special characters
        """
        if not text:
            return 0

        # Basic character count estimate
        base_tokens = len(text) / self.chars_per_token

        # Adjust for whitespace (tends to reduce tokens)
        whitespace_ratio = len(re.findall(r'\s', text)) / max(len(text), 1)
        whitespace_adjustment = 1 - (whitespace_ratio * 0.1)

        # Adjust for numbers (more tokens)
        number_ratio = len(re.findall(r'\d', text)) / max(len(text), 1)
        number_adjustment = 1 + (number_ratio * 0.2)

        # Adjust for special characters
        special_ratio = len(re.findall(r'[^\w\s]', text)) / max(len(text), 1)
        special_adjustment = 1 + (special_ratio * 0.1)

        adjusted_tokens = base_tokens * whitespace_adjustment * number_adjustment * special_adjustment

        return max(1, int(math.ceil(adjusted_tokens)))

    def estimate_json_tokens(self, data: Any) -> int:
        """Estimate tokens for JSON-serializable data."""
        try:
            json_str = json.dumps(data, separators=(',', ':'))
            # JSON tends to have more overhead
            base_tokens = self.estimate_text_tokens(json_str)
            return int(base_tokens * 1.1)  # 10% overhead for structure
        except (TypeError, ValueError):
            return 0

    def estimate_message_tokens(self, message: Message) -> int:
        """
        Estimate tokens for a single message.

        Accounts for:
        - Role tokens
        - Content tokens
        - Tool call tokens
        - Structural overhead
        """
        tokens = self.message_overhead

        # Role token
        tokens += 1

        # Content
        if message.content:
            if isinstance(message.content, str):
                tokens += self.estimate_text_tokens(message.content)
            elif isinstance(message.content, list):
                # Multimodal content
                for part in message.content:
                    if hasattr(part, 'text'):
                        tokens += self.estimate_text_tokens(part.text)
                    elif hasattr(part, 'image_url'):
                        tokens += self.estimate_image_tokens(
                            getattr(part.image_url, 'detail', 'auto')
                        )

        # Name field
        if message.name:
            tokens += self.estimate_text_tokens(message.name) + 1

        # Tool call ID
        if message.tool_call_id:
            tokens += self.estimate_text_tokens(message.tool_call_id) + 1

        # Tool calls
        if message.tool_calls:
            for tc in message.tool_calls:
                tokens += self.FUNCTION_CALL_OVERHEAD
                tokens += self.estimate_text_tokens(tc.function.name)
                tokens += self.estimate_text_tokens(tc.function.arguments)

        return tokens

    def estimate_messages_tokens(self, messages: List[Message]) -> TokenEstimate:
        """
        Estimate tokens for a list of messages.

        Returns detailed breakdown.
        """
        text_tokens = 0
        message_overhead = 0
        image_tokens = 0
        system_tokens = 0

        for msg in messages:
            msg_tokens = self.estimate_message_tokens(msg)

            if msg.role == Role.SYSTEM:
                system_tokens += msg_tokens
            else:
                # Extract image tokens if multimodal
                if isinstance(msg.content, list):
                    for part in msg.content:
                        if hasattr(part, 'image_url'):
                            detail = getattr(part.image_url, 'detail', 'auto')
                            img_toks = self.estimate_image_tokens(detail)
                            image_tokens += img_toks
                            msg_tokens -= img_toks

                text_tokens += msg_tokens - self.message_overhead
                message_overhead += self.message_overhead

        return TokenEstimate(
            total_tokens=text_tokens + message_overhead + image_tokens + system_tokens,
            text_tokens=text_tokens,
            message_overhead=message_overhead,
            image_tokens=image_tokens,
            system_tokens=system_tokens,
            confidence=0.85
        )

    def estimate_image_tokens(
        self,
        detail: str = "auto",
        width: Optional[int] = None,
        height: Optional[int] = None
    ) -> int:
        """
        Estimate tokens for an image.

        Uses OpenAI's image token formula as reference.

        Args:
            detail: "low", "high", or "auto"
            width: Image width (optional, for high detail)
            height: Image height (optional, for high detail)
        """
        if detail == "low":
            return self.IMAGE_TOKENS["low"]

        if detail == "high" or detail == "auto":
            if width and height:
                # Calculate tiles for high detail
                # OpenAI: Scale to fit within 2048x2048, then tile in 512x512
                scale = min(2048 / width, 2048 / height, 1)
                scaled_w = int(width * scale)
                scaled_h = int(height * scale)

                # Fit to 768 on shortest side
                shortest = min(scaled_w, scaled_h)
                if shortest > 768:
                    scale2 = 768 / shortest
                    scaled_w = int(scaled_w * scale2)
                    scaled_h = int(scaled_h * scale2)

                # Count 512x512 tiles
                tiles_w = math.ceil(scaled_w / 512)
                tiles_h = math.ceil(scaled_h / 512)
                num_tiles = tiles_w * tiles_h

                return self.IMAGE_TOKENS["high"] + (num_tiles * self.IMAGE_TOKENS["high_tile"])

            # Default high detail without dimensions
            return self.IMAGE_TOKENS["high"] + (4 * self.IMAGE_TOKENS["high_tile"])  # Assume 2x2 tiles

        return self.IMAGE_TOKENS["low"]

    def estimate_tools_tokens(self, tools: List[Dict[str, Any]]) -> int:
        """
        Estimate tokens for tool definitions.

        Args:
            tools: List of tool definitions (OpenAI format)
        """
        tokens = 0

        for tool in tools:
            tokens += self.TOOL_DEFINITION_BASE

            if "function" in tool:
                func = tool["function"]

                # Function name
                if "name" in func:
                    tokens += self.estimate_text_tokens(func["name"])

                # Description
                if "description" in func:
                    tokens += self.estimate_text_tokens(func["description"])

                # Parameters
                if "parameters" in func:
                    params = func["parameters"]
                    if "properties" in params:
                        for prop_name, prop_schema in params["properties"].items():
                            tokens += self.TOOL_PARAM_TOKENS
                            tokens += self.estimate_text_tokens(prop_name)
                            if "description" in prop_schema:
                                tokens += self.estimate_text_tokens(prop_schema["description"])

        return tokens

    def estimate_request_tokens(
        self,
        messages: List[Message],
        tools: Optional[List[Dict[str, Any]]] = None,
        system_prompt: Optional[str] = None
    ) -> TokenEstimate:
        """
        Estimate total input tokens for a request.

        Args:
            messages: List of messages
            tools: Optional tool definitions
            system_prompt: Optional separate system prompt

        Returns:
            Detailed token estimate
        """
        estimate = self.estimate_messages_tokens(messages)

        # Add tool tokens
        if tools:
            tool_tokens = self.estimate_tools_tokens(tools)
            estimate.tool_tokens = tool_tokens
            estimate.total_tokens += tool_tokens

        # Add separate system prompt if provided
        if system_prompt:
            sys_tokens = self.estimate_text_tokens(system_prompt)
            estimate.system_tokens += sys_tokens
            estimate.total_tokens += sys_tokens

        return estimate

    def estimate_output_tokens(
        self,
        max_tokens: Optional[int] = None,
        typical_ratio: float = 0.5
    ) -> int:
        """
        Estimate expected output tokens.

        Args:
            max_tokens: Maximum tokens allowed
            typical_ratio: Typical output/max ratio

        Returns:
            Estimated output tokens
        """
        if max_tokens:
            return int(max_tokens * typical_ratio)
        return 500  # Default estimate


# Convenience functions

def estimate_tokens(
    text: str,
    provider: str = "default"
) -> int:
    """Estimate tokens for text."""
    estimator = TokenEstimator(provider)
    return estimator.estimate_text_tokens(text)


def estimate_message_tokens(
    messages: List[Message],
    provider: str = "default"
) -> TokenEstimate:
    """Estimate tokens for messages."""
    estimator = TokenEstimator(provider)
    return estimator.estimate_messages_tokens(messages)


def estimate_request_cost(
    messages: List[Message],
    model_id: str,
    tools: Optional[List[Dict[str, Any]]] = None,
    max_output_tokens: int = 1000
) -> Dict[str, Any]:
    """
    Estimate cost for a request before sending.

    Returns dict with estimated tokens and cost.
    """
    from .pricing import calculate_cost

    # Determine provider from model_id
    provider = "default"
    if "/" in model_id:
        provider = model_id.split("/")[0]

    estimator = TokenEstimator(provider)

    # Estimate input tokens
    input_estimate = estimator.estimate_request_tokens(messages, tools)

    # Estimate output tokens
    output_tokens = estimator.estimate_output_tokens(max_output_tokens)

    # Calculate cost
    estimated_cost = calculate_cost(
        model_id=model_id,
        input_tokens=input_estimate.total_tokens,
        output_tokens=output_tokens
    )

    return {
        "model_id": model_id,
        "estimated_input_tokens": input_estimate.total_tokens,
        "estimated_output_tokens": output_tokens,
        "estimated_total_tokens": input_estimate.total_tokens + output_tokens,
        "estimated_cost_usd": estimated_cost,
        "token_breakdown": {
            "text": input_estimate.text_tokens,
            "message_overhead": input_estimate.message_overhead,
            "images": input_estimate.image_tokens,
            "tools": input_estimate.tool_tokens,
            "system": input_estimate.system_tokens
        },
        "confidence": input_estimate.confidence
    }
