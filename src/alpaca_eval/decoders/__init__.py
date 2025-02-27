import logging
from typing import Callable, Union


def get_fn_completions(name: Union[str, Callable]) -> Callable:
    """Get a decoder by name."""
    if not isinstance(name, str):
        return name

    print("name:", name)
    if name == "anthropic_completions":
        try:
            from .anthropic import anthropic_completions
        except ImportError as e:
            packages = ["anthropic"]
            logging.exception(f"You need {packages} to use anthropic_completions. Error:")
            raise e

        return anthropic_completions

    elif name == "openai_completions":
        try:
            from .openai import openai_completions
        except ImportError as e:
            packages = ["openai"]
            logging.exception(f"You need {packages} to use openai_completions. Error:")
            raise e

        return openai_completions

    elif name == "huggingface_api_completions":
        try:
            from .huggingface_api import huggingface_api_completions
        except ImportError as e:
            packages = ["huggingface_hub"]
            logging.exception(f"You need {packages} to use huggingface_api_completions. Error:")
            raise e

        return huggingface_api_completions

    elif name == "huggingface_local_completions":
        try:
            from .huggingface_local import huggingface_local_completions
        except ImportError as e:
            packages = ["accelerate", "transformers", "bitsandbytes", "xformers", "peft", "optimum", "scipy"]
            logging.exception(f"You need {packages} to use huggingface_local_completions. Error:")
            raise e

        return huggingface_local_completions

    elif name == "cohere_completions":
        try:
            from .cohere import cohere_completions
        except ImportError as e:
            packages = ["cohere"]
            logging.exception(f"You need {packages} to use cohere_completions. Error:")
            raise e

        return cohere_completions

    elif name == "replicate_completions":
        try:
            from .replicate import replicate_completions
        except ImportError as e:
            packages = ["replicate"]
            logging.exception(f"You need {packages} to use replicate_completions. Error:")
            raise e

        return replicate_completions

    elif name == "jina_chat_completions":
        from .jinachat import jina_chat_completions

        return jina_chat_completions

    elif name == "vllm_local_completions":
        try:
            from .vllm_local import vllm_local_completions

            return vllm_local_completions
        except ImportError as e:
            packages = ["vllm", "ray", "transformers"]
            logging.exception(f"You need {packages} to use vllm_completions. Error:")
            raise e

    elif name == "self_azure_gpt4_completions":
        from .self_azure_gpt4 import self_azure_gpt4_completions
        return self_azure_gpt4_completions

    elif name == "self_closeai_completions":
        from .self_closeai import self_closeai_completions
        return self_closeai_completions

    else:
        raise ValueError(f"Unknown decoder: {name}")
