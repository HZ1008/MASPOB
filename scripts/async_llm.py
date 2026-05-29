import asyncio
import os
import httpx
from openai import AsyncOpenAI, RateLimitError, APIError, APIConnectionError, APITimeoutError
from scripts.formatter import BaseFormatter, FormatError

import yaml
from pathlib import Path
from typing import Dict, Optional, Any

# =============================================================================
# API key resolution: env var > yaml (placeholder values treated as unset)
# Env var selected by base_url:
#   openrouter.ai      -> OPENROUTER_API_KEY
#   api.openai.com     -> OPENAI_API_KEY
#   api.anthropic.com  -> ANTHROPIC_API_KEY
#   api.deepseek.com   -> DEEPSEEK_API_KEY
#   fallback           -> LLM_API_KEY
# =============================================================================
_BASE_URL_ENV_MAP = [
    ("openrouter.ai", "OPENROUTER_API_KEY"),
    ("api.openai.com", "OPENAI_API_KEY"),
    ("api.anthropic.com", "ANTHROPIC_API_KEY"),
    ("api.deepseek.com", "DEEPSEEK_API_KEY"),
]


def _is_placeholder_key(api_key: Optional[str]) -> bool:
    if api_key is None or api_key == "":
        return True
    return api_key.startswith("YOUR_") and api_key.endswith("_HERE")


def _resolve_api_key(yaml_key: Optional[str], base_url: str, llm_name: str = "") -> str:
    """Resolve API key: use yaml value if set, else check env var by base_url, fallback to LLM_API_KEY."""
    if not _is_placeholder_key(yaml_key):
        return yaml_key
    for url_marker, env_name in _BASE_URL_ENV_MAP:
        if url_marker in (base_url or ""):
            env_val = os.environ.get(env_name, "")
            if env_val:
                return env_val
    fallback = os.environ.get("LLM_API_KEY", "")
    if fallback:
        return fallback
    raise ValueError(
        f"No API key found for LLM '{llm_name}' (base_url={base_url!r}). "
        f"Set the appropriate env var (e.g. OPENROUTER_API_KEY / OPENAI_API_KEY / LLM_API_KEY) "
        f"or fill api_key in config/config2.yaml."
    )

# =============================================================================
# Retry configuration
# =============================================================================
MAX_RETRIES = 3
INITIAL_RETRY_DELAY = 1.0  # seconds
MAX_RETRY_DELAY = 30.0     # seconds
RETRY_MULTIPLIER = 2.0     # exponential backoff multiplier

# =============================================================================
# Error statistics (for diagnostics)
# =============================================================================
import time
from collections import defaultdict

class ErrorStats:
    """Track API errors for diagnostics."""
    def __init__(self):
        self.counts = defaultdict(int)
        self.last_error_time = {}
        self.start_time = time.time()

    def record(self, error_type: str):
        self.counts[error_type] += 1
        self.last_error_time[error_type] = time.time()

    def summary(self) -> str:
        elapsed = time.time() - self.start_time
        parts = [f"uptime: {elapsed/60:.1f}min"]
        for err_type, count in sorted(self.counts.items()):
            parts.append(f"{err_type}: {count}")
        return " | ".join(parts)

_error_stats = ErrorStats()

class LLMConfig:
    def __init__(self, config: dict):
        self.model = config.get("model", "gpt-4o-mini")
        self.temperature = config.get("temperature", 1)
        self.key = config.get("key", None)
        self.base_url = config.get("base_url", "https://api.openai.com/v1")
        self.top_p = config.get("top_p", 1)

class LLMsConfig:
    """Configuration manager for multiple LLM configurations"""
    
    _instance = None  # For singleton pattern if needed
    _default_config = None
    
    def __init__(self, config_dict: Optional[Dict[str, Any]] = None):
        """Initialize with an optional configuration dictionary"""
        self.configs = config_dict or {}
    
    @classmethod
    def default(cls):
        """Get or create a default configuration from YAML file"""
        if cls._default_config is None:
            # Look for the config file in common locations
            # Prefer paths that are resolved relative to this file so that
            # calling scripts from different working directories still works.
            project_root = Path(__file__).resolve().parent.parent  # MASPOB
            config_paths = [
                project_root / "config" / "config2.yaml",  # MASPOB/config/config2.yaml
                Path("config/config2.yaml"),                # ./config/config2.yaml (if you run from project root)
                Path("config2.yaml"),                       # ./config2.yaml (fallback)
            ]
            
            config_file = None
            for path in config_paths:
                if path.exists():
                    config_file = path
                    break
            
            if config_file is None:
                raise FileNotFoundError("No default configuration file found in the expected locations")
            
            # Load the YAML file
            with open(config_file, 'r', encoding='utf-8') as f:
                config_data = yaml.safe_load(f)
            
            # Your YAML has a 'models' top-level key that contains the model configs
            if 'models' in config_data:
                config_data = config_data['models']
                
            cls._default_config = cls(config_data)
        
        return cls._default_config
    
    def get(self, llm_name: str) -> LLMConfig:
        """Get the configuration for a specific LLM by name"""
        if llm_name not in self.configs:
            raise ValueError(f"Configuration for {llm_name} not found")
        
        config = self.configs[llm_name]
        base_url = config.get("base_url", "https://api.openai.com/v1")

        # Create a config dictionary with the expected keys for LLMConfig
        # Prefer an explicit "model" field from YAML if provided (e.g. OpenRouter model id),
        # otherwise fall back to using the config key name as the model.
        llm_config = {
            "model": config.get("model", llm_name),
            "temperature": config.get("temperature", 1),
            "key": _resolve_api_key(config.get("api_key"), base_url, llm_name),
            "base_url": base_url,
            "top_p": config.get("top_p", 1)  # Add top_p parameter
        }

        # Create and return an LLMConfig instance with the specified configuration
        return LLMConfig(llm_config)
    
    def add_config(self, name: str, config: Dict[str, Any]) -> None:
        """Add or update a configuration"""
        self.configs[name] = config
    
    def get_all_names(self) -> list:
        """Get names of all available LLM configurations"""
        return list(self.configs.keys())
    
class ModelPricing:
    """Pricing information for different models in USD per 1K tokens"""
    PRICES = {
        # GPT-4o models
        "gpt-4o": {"input": 0.0025, "output": 0.01},
        "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
        "gpt-4o-mini-2024-07-18": {"input": 0.00015, "output": 0.0006},
        "claude-3-5-sonnet": {"input": 0.003, "output": 0.015},
        "o3":{"input":0.003, "output":0.015},
        "o3-mini": {"input": 0.0011, "output": 0.0025},
    }
    
    @classmethod
    def get_price(cls, model_name, token_type):
        """Get the price per 1K tokens for a specific model and token type (input/output)"""
        # Try to find exact match first
        if model_name in cls.PRICES:
            return cls.PRICES[model_name][token_type]

        # Try to find a partial match, preferring longer (more specific) keys first
        # This ensures "gpt-4o-mini" matches before "gpt-4o"
        sorted_keys = sorted(cls.PRICES.keys(), key=len, reverse=True)
        for key in sorted_keys:
            if key in model_name:
                return cls.PRICES[key][token_type]

        # Return default pricing if no match found
        return 0

class TokenUsageTracker:
    """Tracks token usage and calculates costs"""
    def __init__(self):
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_cost = 0
        self.usage_history = []
    
    def add_usage(self, model, input_tokens, output_tokens):
        """Add token usage for a specific API call"""
        input_cost = (input_tokens / 1000) * ModelPricing.get_price(model, "input")
        output_cost = (output_tokens / 1000) * ModelPricing.get_price(model, "output")
        total_cost = input_cost + output_cost
        
        usage_record = {
            "model": model,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
            "input_cost": input_cost,
            "output_cost": output_cost,
            "total_cost": total_cost,
            "prices": {
                "input_price": ModelPricing.get_price(model, "input"),
                "output_price": ModelPricing.get_price(model, "output")
            }
        }
        
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        self.total_cost += total_cost
        self.usage_history.append(usage_record)
        
        return usage_record
    
    def get_summary(self):
        """Get a summary of token usage and costs"""
        return {
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_tokens": self.total_input_tokens + self.total_output_tokens,
            "total_cost": self.total_cost,
            "call_count": len(self.usage_history),
            "history": self.usage_history
        }

    def reset(self):
        """Reset the tracker for a new evaluation session"""
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_cost = 0.0
        self.usage_history = []

class AsyncLLM:
    def __init__(self, config, system_msg:str = None):
        """
        Initialize the AsyncLLM with a configuration
        
        Args:
            config: Either an LLMConfig instance or a string representing the LLM name
                   If a string is provided, it will be looked up in the default configuration
            system_msg: Optional system message to include in all prompts
        """
        # Handle the case where config is a string (LLM name)
        if isinstance(config, str):
            llm_name = config
            config = LLMsConfig.default().get(llm_name)

        # At this point, config should be an LLMConfig instance
        self.config = config
        self.sys_msg = system_msg
        self.usage_tracker = TokenUsageTracker()
        self._request_count = 0
        self._rebuild_http_client()

    def _rebuild_http_client(self):
        """Rebuild HTTP client to avoid connection pool issues after long runs."""
        # Close the old client if it exists
        old_client = getattr(self, 'aclient', None)
        if old_client is not None:
            try:
                asyncio.create_task(old_client.close())
            except Exception:
                pass  # Ignore close errors

        # Create new HTTP client with larger connection pool for high concurrency
        limits = httpx.Limits(
            max_connections=500,
            max_keepalive_connections=200,
            keepalive_expiry=60.0
        )

        http_client = httpx.AsyncClient(
            timeout=httpx.Timeout(connect=60.0, read=300.0, write=30.0, pool=60.0),
            limits=limits
        )
        self.aclient = AsyncOpenAI(
            api_key=self.config.key,
            base_url=self.config.base_url,
            http_client=http_client,
            timeout=120.0  # overall OpenAI client timeout
        )
        self._request_count = 0
        print(f"        [HTTP Client] Rebuilt connection pool", flush=True)

    async def __call__(self, prompt):
        # Rebuild connection pool every 1000 requests to avoid pool exhaustion
        self._request_count += 1
        if self._request_count >= 1000:
            self._rebuild_http_client()
            await asyncio.sleep(2.0)  # wait briefly after rebuild

        message = []
        if self.sys_msg is not None:
            message.append({
                "content": self.sys_msg,
                "role": "system"
            })

        message.append({"role": "user", "content": prompt})

        # API call with retry
        last_exception = None
        retry_delay = INITIAL_RETRY_DELAY

        for attempt in range(MAX_RETRIES + 1):
            try:
                response = await self.aclient.chat.completions.create(
                    model=self.config.model,
                    messages=message,
                    temperature=self.config.temperature,
                    top_p=self.config.top_p,
                )

                # Extract token usage from response
                input_tokens = response.usage.prompt_tokens
                output_tokens = response.usage.completion_tokens

                # Track token usage and calculate cost
                usage_record = self.usage_tracker.add_usage(
                    self.config.model,
                    input_tokens,
                    output_tokens
                )

                ret = response.choices[0].message.content
                return ret

            except (RateLimitError, APIConnectionError, APITimeoutError) as e:
                # Retryable errors: rate limit, connection error, timeout
                last_exception = e
                error_type = type(e).__name__
                _error_stats.record(error_type)

                if attempt < MAX_RETRIES:
                    await asyncio.sleep(retry_delay)
                    retry_delay = min(retry_delay * RETRY_MULTIPLIER, MAX_RETRY_DELAY)
                continue

            except APIError as e:
                # Other API errors, potentially transient
                last_exception = e
                _error_stats.record(f"APIError_{e.status_code}")
                print(f"        [LLM APIError] status={e.status_code}: {str(e)[:150]}", flush=True)
                if attempt < MAX_RETRIES and e.status_code in (500, 502, 503, 504):
                    await asyncio.sleep(retry_delay)
                    retry_delay = min(retry_delay * RETRY_MULTIPLIER, MAX_RETRY_DELAY)
                    continue
                raise  # non-retryable error

            except Exception as e:
                # Unknown error: log and re-raise
                _error_stats.record(type(e).__name__)
                print(f"        [LLM Error] {type(e).__name__}: {str(e)[:150]}", flush=True)
                raise

        # All retries exhausted
        print(f"        [LLM Failed] All {MAX_RETRIES} retries exhausted | cumulative: {_error_stats.summary()}", flush=True)
        raise last_exception
    
    async def call_with_format(self, prompt: str, formatter: BaseFormatter):
        """
        Call the LLM with a prompt and format the response using the provided formatter
        
        Args:
            prompt: The prompt to send to the LLM
            formatter: An instance of a BaseFormatter to validate and parse the response
            
        Returns:
            The formatted response data
            
        Raises:
            FormatError: If the response doesn't match the expected format
        """
        # Prepare the prompt with formatting instructions
        formatted_prompt = formatter.prepare_prompt(prompt)
        # Call the LLM
        response = await self.__call__(formatted_prompt)
        
        # Validate and parse the response
        is_valid, parsed_data = formatter.validate_response(response)
        
        if not is_valid:
            error_message = formatter.format_error_message()
            raise FormatError(f"{error_message}. Raw response: {response}")
        
        return parsed_data
    
    def get_usage_summary(self):
        """Get a summary of token usage and costs"""
        return self.usage_tracker.get_summary()

    def reset_usage(self):
        """Reset usage tracker for a new evaluation session"""
        self.usage_tracker.reset()


def create_llm_instance(llm_config):
    """
    Create an AsyncLLM instance using the provided configuration
    
    Args:
        llm_config: Either an LLMConfig instance, a dictionary of configuration values,
                            or a string representing the LLM name to look up in default config
    
    Returns:
        An instance of AsyncLLM configured according to the provided parameters
    """
    # Case 1: llm_config is already an LLMConfig instance
    if isinstance(llm_config, LLMConfig):
        return AsyncLLM(llm_config)
    
    # Case 2: llm_config is a string (LLM name)
    elif isinstance(llm_config, str):
        return AsyncLLM(llm_config)  # AsyncLLM constructor handles lookup
    
    # Case 3: llm_config is a dictionary
    elif isinstance(llm_config, dict):
        # Create an LLMConfig instance from the dictionary
        llm_config = LLMConfig(llm_config)
        return AsyncLLM(llm_config)
    
    else:
        raise TypeError("llm_config must be an LLMConfig instance, a string, or a dictionary")