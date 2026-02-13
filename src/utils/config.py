"""
Configuration loader and manager.
Handles loading YAML configs and providing access to settings.
"""

import yaml
import os
from pathlib import Path
from typing import Dict, Any


class Config:
    """Configuration manager for transliteration system."""

    def __init__(self, config_dir: str = "configs"):
        """
        Initialize configuration manager.

        Args:
            config_dir: Directory containing configuration files
        """
        self.config_dir = Path(config_dir)
        self._rules_config = None
        self._model_config = None

    @property
    def rules(self) -> Dict[str, Any]:
        """Load and return transliteration rules configuration."""
        if self._rules_config is None:
            config_path = self.config_dir / "transliteration_rules.yaml"
            with open(config_path, 'r', encoding='utf-8') as f:
                self._rules_config = yaml.safe_load(f)
        return self._rules_config

    @property
    def model(self) -> Dict[str, Any]:
        """Load and return model configuration."""
        if self._model_config is None:
            config_path = self.config_dir / "model_config.yaml"
            with open(config_path, 'r', encoding='utf-8') as f:
                self._model_config = yaml.safe_load(f)
        return self._model_config

    def get(self, key_path: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation.

        Args:
            key_path: Path to config value (e.g., 'model.encoder.hidden_size')
            default: Default value if key not found

        Returns:
            Configuration value

        Example:
            >>> config = Config()
            >>> hidden_size = config.get('model.encoder.hidden_size')
        """
        keys = key_path.split('.')

        # Determine which config to use
        if keys[0] == 'rules':
            config_dict = self.rules
            keys = keys[1:]
        elif keys[0] == 'model':
            config_dict = self.model
            keys = keys[1:]
        else:
            # Try both
            try:
                return self._get_nested(self.model, keys, default)
            except (KeyError, TypeError):
                return self._get_nested(self.rules, keys, default)

        return self._get_nested(config_dict, keys, default)

    @staticmethod
    def _get_nested(config: Dict, keys: list, default: Any = None) -> Any:
        """Helper to get nested dictionary values."""
        current = config
        for key in keys:
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                return default
        return current


# Global config instance
_global_config = None


def get_config() -> Config:
    """Get global configuration instance."""
    global _global_config
    if _global_config is None:
        _global_config = Config()
    return _global_config
