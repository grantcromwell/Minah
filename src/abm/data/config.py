""
Configuration for the ABM data module.
"""
from typing import Dict, Any, Optional
import os
import yaml
from pathlib import Path

class DataConfig:
    """
    Configuration manager for the data module.
    """
    
    # Default configuration
    DEFAULTS = {
        # Data source configuration
        'data_sources': {
            'default': 'binance',
            'available': {
                'binance': {
                    'class': 'ccxt.binance',
                    'rate_limit': True,
                    'enable_rate_limit': True,
                    'timeout': 30000,
                },
                'coinbase': {
                    'class': 'ccxt.coinbasepro',
                    'rate_limit': True,
                    'enable_rate_limit': True,
                    'timeout': 30000,
                },
                'file': {
                    'class': 'file',
                    'data_dir': 'data/market',
                    'formats': ['csv', 'parquet', 'json']
                }
            }
        },
        
        # Data processing configuration
        'processing': {
            'resample_freq': '1m',
            'min_liquidity': 1e-6,
            'max_price_change': 0.1,  # 10%
            'max_volume_change': 5.0,  # 500%
            'zscore_threshold': 3.0,
            'min_data_points': 100,
            'max_gap_minutes': 60,
            'confidence_level': 0.95
        },
        
        # Validation configuration
        'validation': {
            'price_threshold': 0.05,  # 5% price change threshold
            'volume_threshold': 5.0,  # 5x volume change threshold
            'zscore_threshold': 3.0,  # Z-score for outlier detection
            'min_liquidity': 1e-6,    # Minimum liquidity threshold
            'orderbook': {
                'price_tolerance': 0.1,  # 10% spread tolerance
                'min_depth_levels': 5,   # Minimum number of price levels
                'min_volume': 0.0        # Minimum volume per price level
            }
        },
        
        # Caching configuration
        'caching': {
            'enabled': True,
            'cache_dir': '.cache/abm_data',
            'max_size_mb': 1024,  # 1GB max cache size
            'ttl_days': 7        # 7 days time-to-live
        },
        
        # Logging configuration
        'logging': {
            'level': 'INFO',
            'file': 'abm_data.log',
            'max_size_mb': 10,
            'backup_count': 5
        },
        
        # Monitoring configuration
        'monitoring': {
            'enabled': True,
            'metrics_port': 9090,
            'push_gateway': None,
            'push_interval': 60  # seconds
        }
    }
    
    def __init__(self, config_file: Optional[str] = None):
        """
        Initialize the configuration.
        
        Args:
            config_file: Path to a YAML configuration file
        """
        self.config = self.DEFAULTS.copy()
        self.config_file = config_file
        
        if config_file and os.path.exists(config_file):
            self.load_config(config_file)
    
    def load_config(self, config_file: str) -> None:
        """
        Load configuration from a YAML file.
        
        Args:
            config_file: Path to the YAML configuration file
        """
        with open(config_file, 'r') as f:
            user_config = yaml.safe_load(f) or {}
            self._deep_update(self.config, user_config)
        
        self.config_file = config_file
        
        # Ensure cache directory exists
        cache_dir = self.get('caching.cache_dir')
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
    
    def save_config(self, config_file: Optional[str] = None) -> None:
        """
        Save the current configuration to a YAML file.
        
        Args:
            config_file: Path to save the configuration (uses the loaded file if None)
        """
        if not config_file and not self.config_file:
            raise ValueError("No config file specified")
            
        save_path = config_file or self.config_file
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        
        with open(save_path, 'w') as f:
            yaml.safe_dump(self.config, f, default_flow_style=False, sort_keys=False)
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a configuration value using dot notation.
        
        Args:
            key: Configuration key in dot notation (e.g., 'data_sources.default')
            default: Default value if key is not found
            
        Returns:
            The configuration value or default if not found
        """
        keys = key.split('.')
        value = self.config
        
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default
    
    def set(self, key: str, value: Any) -> None:
        """
        Set a configuration value using dot notation.
        
        Args:
            key: Configuration key in dot notation
            value: Value to set
        """
        keys = key.split('.')
        current = self.config
        
        for k in keys[:-1]:
            if k not in current:
                current[k] = {}
            current = current[k]
            
        current[keys[-1]] = value
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Get the configuration as a dictionary.
        
        Returns:
            Dictionary containing the configuration
        """
        return self.config.copy()
    
    def _deep_update(self, original: Dict, update: Dict) -> Dict:
        """
        Recursively update a dictionary.
        
        Args:
            original: Original dictionary to update
            update: Dictionary with updates
            
        Returns:
            Updated dictionary
        """
        for key, value in update.items():
            if isinstance(value, dict) and key in original and isinstance(original[key], dict):
                original[key] = self._deep_update(original[key], value)
            else:
                original[key] = value
        return original

# Global configuration instance
config = DataConfig()

def get_config() -> DataConfig:
    """
    Get the global configuration instance.
    
    Returns:
        The global DataConfig instance
    """
    return config

def load_config(config_file: str) -> DataConfig:
    """
    Load configuration from a file into the global instance.
    
    Args:
        config_file: Path to the configuration file
        
    Returns:
        The updated global DataConfig instance
    """
    config.load_config(config_file)
    return config

def save_config(config_file: Optional[str] = None) -> None:
    """
    Save the current configuration to a file.
    
    Args:
        config_file: Path to save the configuration (uses the loaded file if None)
    """
    config.save_config(config_file)

# Initialize with environment variables if present
if os.environ.get('ABM_DATA_CONFIG'):
    config.load_config(os.environ['ABM_DATA_CONFIG'])
elif os.path.exists('config/abm_data.yaml'):
    config.load_config('config/abm_data.yaml')
