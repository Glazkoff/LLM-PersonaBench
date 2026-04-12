import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from openai import (
    APIConnectionError,
    APIError,
    APIStatusError,
    APITimeoutError,
    AuthenticationError,
    BadRequestError,
    NotFoundError,
    RateLimitError,
)

from src.models.base import BaseLLM

load_dotenv()


class OpenRouterModel(BaseLLM):
    """
    Wrapper for OpenRouter (OpenAI-compatible API).
    """

    def __init__(
        self,
        model_name: str,
        temperature: float = 0.7,
        timeout: float | None = None,
        max_retries: int | None = None,
    ):
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise RuntimeError(
                "OpenRouter API key is missing. Set OPENROUTER_API_KEY in environment or .env."
            )

        super().__init__(model_name)
        self._api_key = api_key

        llm_kwargs: dict = {
            "model": model_name,
            "api_key": api_key,
            "base_url": "https://openrouter.ai/api/v1",
            "temperature": temperature,
        }

        # Optional OpenRouter attribution headers.
        default_headers: dict[str, str] = {}
        http_referer = os.getenv("OPENROUTER_HTTP_REFERER") or os.getenv("OPENROUTER_SITE_URL")
        app_title = os.getenv("OPENROUTER_APP_TITLE") or os.getenv("OPENROUTER_APP_NAME")
        if http_referer:
            default_headers["HTTP-Referer"] = http_referer
        if app_title:
            default_headers["X-Title"] = app_title
        if default_headers:
            llm_kwargs["default_headers"] = default_headers

        if timeout is not None:
            llm_kwargs["timeout"] = timeout
        if max_retries is not None:
            llm_kwargs["max_retries"] = max_retries

        self.llm = ChatOpenAI(**llm_kwargs)

    def _sanitize_error(self, error: Exception) -> str:
        message = str(error)
        if self._api_key:
            message = message.replace(self._api_key, "***")
            message = message.replace(f"Bearer {self._api_key}", "Bearer ***")
        return message

    def generate(self, prompt: str):
        try:
            response = prompt | self.llm
            return response.invoke({})
        except AuthenticationError as exc:
            raise RuntimeError(
                "OpenRouter authentication failed: API key is invalid or expired (OPENROUTER_API_KEY)."
            ) from exc
        except RateLimitError as exc:
            raise RuntimeError(
                "OpenRouter rate limit reached. Reduce request frequency or retry later."
            ) from exc
        except APITimeoutError as exc:
            raise RuntimeError(
                "OpenRouter request timed out. Increase timeout in config or retry."
            ) from exc
        except NotFoundError as exc:
            raise RuntimeError(
                f"OpenRouter model/provider not available: '{self.model_name}'."
            ) from exc
        except BadRequestError as exc:
            raise RuntimeError(
                f"OpenRouter rejected request for model '{self.model_name}': {self._sanitize_error(exc)}"
            ) from exc
        except APIConnectionError as exc:
            raise RuntimeError(
                f"OpenRouter connection error: {self._sanitize_error(exc)}"
            ) from exc
        except APIStatusError as exc:
            status = getattr(exc, "status_code", None)
            status_text = f" (HTTP {status})" if status is not None else ""
            raise RuntimeError(
                f"OpenRouter API error{status_text}: {self._sanitize_error(exc)}"
            ) from exc
        except APIError as exc:
            raise RuntimeError(
                f"OpenRouter API error: {self._sanitize_error(exc)}"
            ) from exc
        except Exception as exc:
            raise RuntimeError(
                f"OpenRouter unexpected error: {self._sanitize_error(exc)}"
            ) from exc

    def generate_batch(self, prompts: list[str]) -> list[str]:
        return [self.generate(p) for p in prompts]
