# File: src/document_processor/core/models.py

import json
from pydantic import BaseModel, Field
from typing import Any, List, Dict, Optional, Union
from datetime import datetime

# -----------------------------------------------------------------------------
# DATA MODELS WITH PYDANTIC
# -----------------------------------------------------------------------------
class ChunkInfo(BaseModel):
    id: str
    number: int
    tokens: int
    doc_id: str
    content_hash: str
    character_count: int
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat())

class ContentVersion(BaseModel):
    version_id: str
    created_at: str
    content_hash: str
    parent_version_id: Optional[str] = None
    changes_description: Optional[str] = None

class ContentManifest(BaseModel):
    manifest_id: str
    created_at: str
    updated_at: str
    version_history: List[ContentVersion]
    document_ids: List[str]
    total_chunks: int
    total_tokens: int
    model_name: str
    content_hashes: List[str]
    metadata: Dict[str, Any] = Field(default_factory=dict)

class ProcessingMetadata(BaseModel):
    processing_id: str
    started_at: str
    completed_at: Optional[str]
    manifest_id: str
    version_id: str
    document_ids: List[str]
    chunk_ids: List[str]
    status: str
    error: Optional[str] = None
    processing_stats: Dict[str, Any] = Field(default_factory=dict)

class DocumentInfo(BaseModel):
    id: str
    filename: str
    original_path: str
    total_chunks: int
    total_tokens: int
    total_chars: int
    total_lines: int
    model_name: str
    token_limit: int
    md5_hash: str
    file_size: int
    chunks: List[ChunkInfo]
    processed_at: str = Field(default_factory=lambda: datetime.now().isoformat())
    version_id: Optional[str] = None
    manifest_id: Optional[str] = None

class ProcessingProgress(BaseModel):
    total_files: int
    processed_files: int
    current_file: str
    start_time: datetime
    processed_chunks: int
    total_tokens: int
    current_chunk: int
    total_chunks: Optional[int] = None
    bytes_processed: int = 0

class ProcessingState(BaseModel):
    doc_id: str
    current_chunk: int
    processed_chunks: List[str]
    is_complete: bool
    error_message: Optional[str] = None
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())

class VersionHistoryItem(BaseModel):
    version_id: str
    parent_version_id: Optional[str]
    timestamp: str
    action: str
    details: Optional[Dict[str, Any]] = None

class ChunkMetadata(BaseModel):
    id: str
    number: int
    tokens: int
    doc_id: str
    content_hash: str
    character_count: int
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat())
    llm_analysis: Dict[str, Any] = Field(default_factory=dict)
    llm_entity_extraction: Dict[str, Any] = Field(default_factory=dict)
    version_id: str = Field(default_factory=lambda: datetime.now().isoformat())
    parent_version_id: Optional[str] = None
    version_history: List[VersionHistoryItem] = Field(default_factory=list)

class DocumentMetadata(BaseModel):
    id: str
    filename: str
    processed_at: str
    chunks: List[Dict[str, Any]] = Field(default_factory=list)
    version_id: str = Field(default_factory=lambda: datetime.now().isoformat())
    version_history: List[VersionHistoryItem] = Field(default_factory=list)

class DateTimeEncoder(json.JSONEncoder):
    """Custom JSON Encoder to handle datetime objects."""
    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        return super().default(obj)

# -----------------------------------------------------------------------------
# PROCESSING STATE MODEL
# -----------------------------------------------------------------------------

class ProcessingState(BaseModel):
    doc_id: str
    current_chunk: int
    processed_chunks: List[str]
    is_complete: bool
    error_message: Optional[str] = None

# -----------------------------------------------------------------------------
# PROCESSING PROGRESS MODEL
# -----------------------------------------------------------------------------

class ProcessingProgress(BaseModel):
    total_files: int
    processed_files: int
    current_file: str
    start_time: datetime
    processed_chunks: int
    total_tokens: int
    current_chunk: int