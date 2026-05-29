# Embedding functions for prompt optimization

import os
from typing import List

import torch
import torch.nn.functional as F


# =============================================================================
# EMBEDDING CONFIGURATION
# =============================================================================

class EmbeddingConfig:
    """Embedding model configuration."""
    MODE = "openrouter"  # "local" or "openrouter"
    LOCAL_MODEL = "sentence-transformers/all-mpnet-base-v2"
    OPENROUTER_MODEL = "qwen/qwen3-embedding-8b"
    OPENROUTER_DIMENSIONS = 1024  # MRL supports 32-4096

    @classmethod
    def get_api_key(cls) -> str:
        """Get API key from env var or config file (priority: OPENROUTER_API_KEY > yaml value > LLM_API_KEY)."""
        env_key = os.environ.get("OPENROUTER_API_KEY", "")
        if env_key:
            return env_key
        try:
            import yaml
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            config_path = os.path.join(base_dir, "config", "config2.yaml")
            if os.path.exists(config_path):
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = yaml.safe_load(f)
                if 'models' in config:
                    for model_cfg in config['models'].values():
                        api_key = model_cfg.get('api_key')
                        if api_key and 'openrouter' in model_cfg.get('base_url', ''):
                            if not (api_key.startswith("YOUR_") and api_key.endswith("_HERE")):
                                return api_key
        except Exception:
            pass
        return os.environ.get("LLM_API_KEY", "")

    @classmethod
    def use_openrouter(cls) -> bool:
        return cls.MODE == "openrouter"


# Backward-compatible constants
EMBEDDING_MODE = EmbeddingConfig.MODE
SENTENCE_MODEL_NAME = EmbeddingConfig.LOCAL_MODEL
OPENROUTER_EMBEDDING_MODEL = EmbeddingConfig.OPENROUTER_MODEL
OPENROUTER_EMBEDDING_DIMENSIONS = EmbeddingConfig.OPENROUTER_DIMENSIONS
OPENROUTER_API_KEY = EmbeddingConfig.get_api_key()


# =============================================================================
# EMBEDDING FUNCTIONS
# =============================================================================

def mean_pooling(model_output, attention_mask):
    """Mean pooling for sentence embeddings."""
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


def get_openrouter_embeddings(texts: List[str], model: str = None, api_key: str = None,
                              dimensions: int = None, batch_size: int = 20) -> torch.Tensor:
    """Fetch text embeddings via the OpenRouter API.

    Args:
        texts: List of texts to embed.
        model: OpenRouter embedding model name.
        api_key: OpenRouter API key.
        dimensions: Embedding dimensions (MRL supports custom dimensions).
        batch_size: Number of texts per batch (to stay within API limits).

    Returns:
        Normalized embeddings tensor of shape (len(texts), embedding_dim).
    """
    import requests

    model = model or OPENROUTER_EMBEDDING_MODEL
    api_key = api_key or OPENROUTER_API_KEY
    dimensions = dimensions or OPENROUTER_EMBEDDING_DIMENSIONS

    if not api_key:
        raise ValueError("OpenRouter API key not set. Set OPENROUTER_API_KEY environment variable.")

    all_embeddings = []

    # Process in batches
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]

        request_body = {
            "model": model,
            "input": batch_texts,
        }
        if dimensions:
            request_body["dimensions"] = dimensions

        response = requests.post(
            "https://openrouter.ai/api/v1/embeddings",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json=request_body,
            timeout=60,
        )

        if response.status_code != 200:
            raise RuntimeError(f"OpenRouter API error: {response.status_code} - {response.text}")

        data = response.json()

        # Sort by index to preserve order
        batch_embeddings = sorted(data["data"], key=lambda x: x["index"])
        for item in batch_embeddings:
            all_embeddings.append(item["embedding"])

    # Convert to tensor and normalize
    embeddings_tensor = torch.tensor(all_embeddings, dtype=torch.float32)
    return F.normalize(embeddings_tensor, p=2, dim=1)


def get_sen_embedding(model, tokenizer, sentences, use_openrouter: bool = None):
    """Get normalized sentence embeddings.

    Args:
        model: Local model (for local mode) or None (for openrouter mode).
        tokenizer: Local tokenizer (for local mode) or None.
        sentences: List of sentences to embed.
        use_openrouter: Whether to use OpenRouter API; if None, inferred from EMBEDDING_MODE.

    Returns:
        Normalized embeddings tensor.
    """
    if use_openrouter is None:
        use_openrouter = (EMBEDDING_MODE == "openrouter")

    if use_openrouter:
        return get_openrouter_embeddings(sentences)
    else:
        # Local model path
        encoded = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')
        with torch.no_grad():
            output = model(**encoded)
        embeddings = mean_pooling(output, encoded['attention_mask'])
        return F.normalize(embeddings, p=2, dim=1)

