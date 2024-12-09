# File: src/document_processor/utils/config.py

from pydantic import BaseModel, Field
from typing import Dict, Union

MODEL_CONFIGS = {
    'gpt-3.5': {'tokens': 4096, 'encoding': 'cl100k_base'},
    'gpt-4': {'tokens': 8192, 'encoding': 'cl100k_base'},
    'gpt-4-32k': {'tokens': 32768, 'encoding': 'cl100k_base'},
    'claude': {'tokens': 8192, 'encoding': 'cl100k_base'},
    'claude-2': {'tokens': 100000, 'encoding': 'cl100k_base'},
    'gpt-4o': {'tokens': 16384, 'encoding': 'cl100k_base'}
}

class AppConfig(BaseModel):  # Renamed from Config to AppConfig
    model_name: str = Field(default="gpt-4")
    spacy_model: str = Field(default="en_core_web_sm")
    max_concurrent_files: int = Field(default=10)
    chunk_reduction_factor: float = Field(default=1.0)
    output_dir: str = Field(default="output")
    chunk_dir: str = Field(default="chunks")

    model_configs: Dict[str, Dict[str, Union[int, str]]] = Field(default_factory=lambda: MODEL_CONFIGS)

    model_config = {  # Defined as a class variable (dictionary)
        "json_schema_extra": {
            "example": {
                "model_name": "gpt-4",
                "spacy_model": "en_core_web_sm",
                "max_concurrent_files": 10,
                "chunk_reduction_factor": 1.0,
                "output_dir": "output",
                "chunk_dir": "chunks"
            }
        }
    }
