"""
The brain package: one interface, several providers.

Use `create_brain(config)` to get the brain selected in config.toml.
"""

from jarvis.brain.base import Brain, BrainResponse


def create_brain(config: dict) -> Brain:
    """Build the brain chosen in config['brain']['provider']."""
    settings = config["brain"]
    provider = settings["provider"]

    if provider == "ollama":
        from jarvis.brain.ollama_brain import OllamaBrain
        return OllamaBrain(settings)

    if provider == "openai_compatible":
        raise NotImplementedError(
            "The cloud brain is not built yet. We add it when we evaluate the free-tier APIs."
        )

    raise ValueError(f"Unknown brain provider in config.toml: '{provider}'")
