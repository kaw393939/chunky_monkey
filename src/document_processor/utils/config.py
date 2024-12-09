"""Configuration settings for the document processor."""

from pathlib import Path
from typing import Dict, Any

# Model configurations
MODEL_CONFIGS = {
    'gpt-3.5': {'tokens': 4096, 'encoding': 'cl100k_base'},
    'gpt-4': {'tokens': 8192, 'encoding': 'cl100k_base'},
    'gpt-4-32k': {'tokens': 32768, 'encoding': 'cl100k_base'},
    'claude': {'tokens': 8192, 'encoding': 'cl100k_base'},
    'claude-2': {'tokens': 100000, 'encoding': 'cl100k_base'},
    'gpt-4o': {'tokens': 16384, 'encoding': 'cl100k_base'}
}

# Default settings
DEFAULT_MODEL_NAME = "gpt-4"
DEFAULT_SPACY_MODEL = "en_core_web_sm"
MAX_CONCURRENT_FILES = 10

# Verification thresholds
MAX_ALLOWED_TOKEN_DIFFERENCE = 5
STRICT_SIMILARITY_THRESHOLD = 0.995
LENIENT_SIMILARITY_THRESHOLD = 0.95
TOKEN_SIMILARITY_THRESHOLD = 0.98

# File paths
DEFAULT_OUTPUT_DIR = Path("output")
DEFAULT_CHUNK_DIR = Path("chunks")
DEFAULT_MANIFEST_FILENAME = "manifest.json"
DEFAULT_DOC_INFO_FILENAME = "document_info.json"

# Processing settings
DEFAULT_CHUNK_REDUCTION_FACTOR = 1.0
MAX_RETRIES = 3
RETRY_DELAY = 1.0  # seconds

class Config:
    """Configuration class for document processor settings."""
    
    def __init__(self, **kwargs: Dict[str, Any]):
        self.model_name = kwargs.get('model_name', DEFAULT_MODEL_NAME)
        self.spacy_model = kwargs.get('spacy_model', DEFAULT_SPACY_MODEL)
        self.max_concurrent_files = kwargs.get('max_concurrent_files', MAX_CONCURRENT_FILES)
        self.chunk_reduction_factor = kwargs.get('chunk_reduction_factor', DEFAULT_CHUNK_REDUCTION_FACTOR)
        self.output_dir = Path(kwargs.get('output_dir', DEFAULT_OUTPUT_DIR))
        self.chunk_dir = Path(kwargs.get('chunk_dir', DEFAULT_CHUNK_DIR))
        
        # Ensure model name is valid
        if self.model_name not in MODEL_CONFIGS:
            raise ValueError(f"Invalid model name. Choose from: {list(MODEL_CONFIGS.keys())}")
        
        # Set token limit based on model and reduction factor
        base_limit = MODEL_CONFIGS[self.model_name]['tokens']
        self.token_limit = int(base_limit * self.chunk_reduction_factor)
        
        # Create necessary directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.chunk_dir.mkdir(parents=True, exist_ok=True)

    @property
    def model_config(self) -> Dict[str, Any]:
        """Get the configuration for the selected model."""
        return MODEL_CONFIGS[self.model_name]

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            'model_name': self.model_name,
            'spacy_model': self.spacy_model,
            'max_concurrent_files': self.max_concurrent_files,
            'chunk_reduction_factor': self.chunk_reduction_factor,
            'output_dir': str(self.output_dir),
            'chunk_dir': str(self.chunk_dir),
            'token_limit': self.token_limit
        }