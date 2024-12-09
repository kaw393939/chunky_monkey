"""Service for managing document chunks."""

import asyncio
from pathlib import Path
from typing import Optional, Dict, Any, List
import json
import hashlib

import aiofiles
from filelock import FileLock

from ..core.models import ChunkMetadata, VersionHistoryItem
from ..utils import get_logger
from ..utils.config import Config

logger = get_logger(__name__)

class ChunkService:
    """Service for managing document chunks and their metadata."""

    def __init__(self, config: Config):
        """
        Initialize the chunk service.
        
        Args:
            config: Application configuration
        """
        self.config = config
        self.chunk_dir = config.chunk_dir
        self.chunk_dir.mkdir(parents=True, exist_ok=True)
        self._locks: Dict[str, FileLock] = {}

    def _get_chunk_path(self, chunk_id: str) -> Path:
        """Get the path for a chunk's content file."""
        return self.chunk_dir / f"{chunk_id}.txt"

    def _get_metadata_path(self, chunk_id: str) -> Path:
        """Get the path for a chunk's metadata file."""
        return self.chunk_dir / f"{chunk_id}.json"

    def _get_lock(self, chunk_id: str) -> FileLock:
        """Get or create a lock for a chunk."""
        if chunk_id not in self._locks:
            lock_path = self.chunk_dir / f"{chunk_id}.lock"
            self._locks[chunk_id] = FileLock(str(lock_path))
        return self._locks[chunk_id]

    @staticmethod
    def calculate_hash(content: str) -> str:
        """Calculate MD5 hash of content."""
        return hashlib.md5(content.encode()).hexdigest()

    async def create_chunk(
        self,
        chunk_id: str,
        content: str,
        doc_id: str,
        number: int,
        token_count: int
    ) -> ChunkMetadata:
        """
        Create a new chunk with content and metadata.
        
        Args:
            chunk_id: Unique identifier for the chunk
            content: Chunk content
            doc_id: Parent document ID
            number: Chunk sequence number
            token_count: Number of tokens in chunk
            
        Returns:
            ChunkMetadata for the created chunk
        
        Raises:
            FileExistsError: If chunk already exists
            ValueError: If invalid parameters
        """
        if not content:
            raise ValueError("Chunk content cannot be empty")

        chunk_path = self._get_chunk_path(chunk_id)
        metadata_path = self._get_metadata_path(chunk_id)
        lock = self._get_lock(chunk_id)

        async with lock:
            if chunk_path.exists() or metadata_path.exists():
                raise FileExistsError(f"Chunk {chunk_id} already exists")

            # Create metadata
            content_hash = self.calculate_hash(content)
            metadata = ChunkMetadata(
                id=chunk_id,
                number=number,
                tokens=token_count,
                doc_id=doc_id,
                content_hash=content_hash,
                character_count=len(content)
            )

            # Write content and metadata atomically
            try:
                async with aiofiles.open(chunk_path, 'w') as f:
                    await f.write(content)

                async with aiofiles.open(metadata_path, 'w') as f:
                    await f.write(metadata.model_dump_json(indent=2))

                logger.info(f"Created chunk {chunk_id} for document {doc_id}")
                return metadata

            except Exception as e:
                # Cleanup on failure
                if chunk_path.exists():
                    chunk_path.unlink()
                if metadata_path.exists():
                    metadata_path.unlink()
                logger.error(f"Failed to create chunk {chunk_id}: {e}")
                raise

    async def read_chunk(self, chunk_id: str) -> tuple[str, ChunkMetadata]:
        """
        Read a chunk's content and metadata.
        
        Args:
            chunk_id: Chunk identifier
            
        Returns:
            Tuple of (content, metadata)
            
        Raises:
            FileNotFoundError: If chunk doesn't exist
        """
        chunk_path = self._get_chunk_path(chunk_id)
        metadata_path = self._get_metadata_path(chunk_id)
        lock = self._get_lock(chunk_id)

        async with lock:
            if not chunk_path.exists() or not metadata_path.exists():
                raise FileNotFoundError(f"Chunk {chunk_id} not found")

            try:
                async with aiofiles.open(chunk_path, 'r') as f:
                    content = await f.read()

                async with aiofiles.open(metadata_path, 'r') as f:
                    metadata_json = await f.read()
                    metadata = ChunkMetadata.model_validate_json(metadata_json)

                # Verify content integrity
                current_hash = self.calculate_hash(content)
                if current_hash != metadata.content_hash:
                    logger.error(f"Content hash mismatch for chunk {chunk_id}")
                    raise ValueError("Content integrity check failed")

                return content, metadata

            except Exception as e:
                logger.error(f"Failed to read chunk {chunk_id}: {e}")
                raise

    async def update_chunk(
        self,
        chunk_id: str,
        content: Optional[str] = None,
        metadata_updates: Optional[Dict[str, Any]] = None
    ) -> ChunkMetadata:
        """
        Update a chunk's content and/or metadata.
        
        Args:
            chunk_id: Chunk identifier
            content: New content (optional)
            metadata_updates: Metadata fields to update (optional)
            
        Returns:
            Updated ChunkMetadata
            
        Raises:
            FileNotFoundError: If chunk doesn't exist
        """
        if not content and not metadata_updates:
            raise ValueError("No updates provided")

        lock = self._get_lock(chunk_id)
        async with lock:
            # Read current state
            current_content, metadata = await self.read_chunk(chunk_id)

            # Update content if provided
            if content is not None:
                content_hash = self.calculate_hash(content)
                metadata.content_hash = content_hash
                metadata.character_count = len(content)
                await self._write_chunk_content(chunk_id, content)

            # Update metadata if provided
            if metadata_updates:
                for key, value in metadata_updates.items():
                    if hasattr(metadata, key):
                        setattr(metadata, key, value)

            # Track version history
            metadata.version_history.append(
                VersionHistoryItem(
                    version_id=metadata.version_id,
                    parent_version_id=metadata.parent_version_id,
                    action="updated",
                    details={"content_updated": content is not None}
                )
            )

            # Write updated metadata
            await self._write_chunk_metadata(chunk_id, metadata)
            logger.info(f"Updated chunk {chunk_id}")
            return metadata

    async def delete_chunk(self, chunk_id: str) -> bool:
        """
        Delete a chunk and its metadata.
        
        Args:
            chunk_id: Chunk identifier
            
        Returns:
            True if deleted, False if not found
        """
        chunk_path = self._get_chunk_path(chunk_id)
        metadata_path = self._get_metadata_path(chunk_id)
        lock = self._get_lock(chunk_id)

        async with lock:
            try:
                if chunk_path.exists():
                    chunk_path.unlink()
                if metadata_path.exists():
                    metadata_path.unlink()
                logger.info(f"Deleted chunk {chunk_id}")
                return True
            except Exception as e:
                logger.error(f"Failed to delete chunk {chunk_id}: {e}")
                return False
            finally:
                # Clean up lock
                if chunk_id in self._locks:
                    del self._locks[chunk_id]

    async def verify_chunk(self, chunk_id: str) -> bool:
        """
        Verify a chunk's content integrity.
        
        Args:
            chunk_id: Chunk identifier
            
        Returns:
            True if verified, False if failed
        """
        try:
            content, metadata = await self.read_chunk(chunk_id)
            current_hash = self.calculate_hash(content)
            return current_hash == metadata.content_hash
        except Exception as e:
            logger.error(f"Chunk verification failed for {chunk_id}: {e}")
            return False

    async def _write_chunk_content(self, chunk_id: str, content: str):
        """Write chunk content atomically."""
        chunk_path = self._get_chunk_path(chunk_id)
        temp_path = chunk_path.with_suffix('.tmp')

        try:
            async with aiofiles.open(temp_path, 'w') as f:
                await f.write(content)
            temp_path.rename(chunk_path)
        except Exception:
            if temp_path.exists():
                temp_path.unlink()
            raise

    async def _write_chunk_metadata(self, chunk_id: str, metadata: ChunkMetadata):
        """Write chunk metadata atomically."""
        metadata_path = self._get_metadata_path(chunk_id)
        temp_path = metadata_path.with_suffix('.tmp')

        try:
            async with aiofiles.open(temp_path, 'w') as f:
                await f.write(metadata.model_dump_json(indent=2))
            temp_path.rename(metadata_path)
        except Exception:
            if temp_path.exists():
                temp_path.unlink()
            raise

    async def list_chunks(self, doc_id: Optional[str] = None) -> List[ChunkMetadata]:
        """
        List all chunks, optionally filtered by document ID.
        
        Args:
            doc_id: Optional document ID to filter by
            
        Returns:
            List of ChunkMetadata
        """
        chunks = []
        for metadata_path in self.chunk_dir.glob("*.json"):
            try:
                async with aiofiles.open(metadata_path, 'r') as f:
                    metadata_json = await f.read()
                    metadata = ChunkMetadata.model_validate_json(metadata_json)
                    if not doc_id or metadata.doc_id == doc_id:
                        chunks.append(metadata)
            except Exception as e:
                logger.error(f"Failed to read metadata from {metadata_path}: {e}")

        return sorted(chunks, key=lambda x: x.number)