from src.models.providers import cloud_api, openrouter_api


def get_model(config: dict):
    provider = str(config["provider"]).lower()
    model_name = config["model_name"]

    # timeout/max_retries — опциональные настройки, чтобы вызовы не могли висеть бесконечно
    timeout = config.get("timeout") or config.get("request_timeout")
    max_retries = config.get("max_retries")

    if provider == "cloud":
        return cloud_api.CloudAPIModel(
            model_name=model_name,
            temperature=config.get("temperature", 0.7),
            timeout=timeout,
            max_retries=max_retries,
        )

    if provider == "openrouter":
        return openrouter_api.OpenRouterModel(
            model_name=model_name,
            temperature=config.get("temperature", 0.7),
            timeout=timeout,
            max_retries=max_retries,
        )

    raise ValueError(f"Unknown provider: {provider}. Supported providers: cloud, openrouter")
