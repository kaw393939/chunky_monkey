"""Core data models for the document processor."""

from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Any, Union
from pydantic import BaseModel, Field, ConfigDict, validator

class VersionHistoryItem(BaseModel):
    """Represents a single version history entry."""
    version_id: str
    parent_version_id: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.now)
    action: str = "updated"
    changes_description: Optional[str] = None
    details: Optional[Dict[str, Any]] = Field(default_factory=dict)

class ChunkMetadata(BaseModel):
    """Metadata for a single document chunk."""
    id: str
    number: int
    tokens: int
    doc_id: str
    content_hash: str
    character_count: int
    created_at: datetime = Field(default_factory=datetime.now)
    llm_analysis: Dict[str, Any] = Field(default_factory=dict)
    llm_entity_extraction: Dict[str, Any] = Field(default_factory=dict)
    version_id: str = Field(default_factory=lambda: datetime.now().isoformat())
    parent_version_id: Optional[str] = None
    version_history: List[VersionHistoryItem] = Field(default_factory=list)

    model_config = ConfigDict(from_attributes=True)

    @validator('number')
    def validate_number(cls, v):
        if v < 0:
            raise ValueError("Chunk number cannot be negative")
        return v

class DocumentMetadata(BaseModel):
    """Metadata for a document."""
    id: str
    filename: str
    processed_at: datetime = Field(default_factory=datetime.now)
    chunks: List[ChunkMetadata] = Field(default_factory=list)
    version_id: str = Field(default_factory=lambda: datetime.now().isoformat())
    version_history: List[VersionHistoryItem] = Field(default_factory=list)
    total_chunks: int = 0
    total_tokens: int = 0
    model_name: str
    token_limit: int
    file_size: int
    content_hash: str

    model_config = ConfigDict(from_attributes=True)

    @validator('chunks')
    def validate_chunks(cls, v):
        """Ensure chunks are properly ordered."""
        return sorted(v, key=lambda x: x.number)

class ContentManifest(BaseModel):
    """Manifest tracking all documents and their relationships."""
    manifest_id: str
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    version_history: List[VersionHistoryItem] = Field(default_factory=list)
    document_ids: List[str] = Field(default_factory=list)
    total_chunks: int = 0
    total_tokens: int = 0
    model_name: str
    content_hashes: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)

class ProcessingMetadata(BaseModel):
    """Metadata about document processing operations."""
    processing_id: str
    started_at: datetime = Field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None
    manifest_id: str
    version_id: str
    document_ids: List[str]
    chunk_ids: List[str]
    status: str
    error: Optional[str] = None
    processing_stats: Dict[str, Any] = Field(default_factory=dict)

class ProcessingProgress(BaseModel):
    """Tracks progress during document processing."""
    total_files: int
    processed_files: int
    current_file: str
    start_time: datetime = Field(default_factory=datetime.now)
    processed_chunks: int = 0
    total_tokens: int = 0
    current_chunk: int = 0
    total_chunks: Optional[int] = None
    bytes_processed: int = 0

    def calculate_progress(self) -> float:
        """Calculate progress percentage."""
        if not self.total_chunks:
            return (self.processed_files / self.total_files) * 100
        return (self.current_chunk / self.total_chunks) * 100

class ProcessingState(BaseModel):
    """Represents the current state of document processing."""
    doc_id: str
    current_chunk: int
    processed_chunks: List[str] = Field(default_factory=list)
    is_complete: bool = False
    error_message: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.now)

class DocumentInfo(BaseModel):
    """Complete information about a processed document."""
    id: str
    filename: str
    original_path: Path
    total_chunks: int
    total_tokens: int
    total_chars: int
    total_lines: int
    model_name: str
    token_limit: int
    md5_hash: str
    file_size: int
    chunks: List[ChunkMetadata]
    processed_at: datetime = Field(default_factory=datetime.now)
    version_id: Optional[str] = None
    manifest_id: Optional[str] = None

    model_config = ConfigDict(
        from_attributes=True,
        arbitrary_types_allowed=True  # Needed for Path objects
    )

    def get_chunk_by_number(self, number: int) -> Optional[ChunkMetadata]:
        """Retrieve a chunk by its number."""
        for chunk in self.chunks:
            if chunk.number == number:
                return chunk
        return None

    def get_chunk_by_id(self, chunk_id: str) -> Optional[ChunkMetadata]:
        """Retrieve a chunk by its ID."""
        for chunk in self.chunks:
            if chunk.id == chunk_id:
                return chunk
        return None