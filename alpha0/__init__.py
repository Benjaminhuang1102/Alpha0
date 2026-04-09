"""Alpha0: AlphaZero-inspired reinforcement learning system for equity portfolio allocation."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
CONFIG_PATH = PROJECT_ROOT / "config" / "default.yaml"


def load_config(path: Path | str | None = None) -> dict:
    """Load YAML config. Defaults to config/default.yaml relative to project root."""
    import yaml

    config_path = Path(path) if path is not None else CONFIG_PATH
    with open(config_path) as f:
        return yaml.safe_load(f)
