"""Service for managing complete documents and their metadata."""

import asyncio
import shutil
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
import uuid

import aiofiles
from filelock import FileLock

from ..core.models import (
    DocumentMetadata,
    ChunkMetadata,
    VersionHistoryItem,
    DocumentInfo
)
from .chunk_service import ChunkService
from ..utils import get_logger
from ..utils.config import Config

logger = get_logger(__name__)

class DocumentService:
    """Service for managing documents and coordinating with chunks."""

    def __init__(self, config: Config, chunk_service: ChunkService):
        """
        Initialize the document service.
        
        Args:
            config: Application configuration
            chunk_service: ChunkService instance for managing chunks
        """
        self.config = config
        self.chunk_service = chunk_service
        self.docs_dir = config.output_dir / "documents"
        self.docs_dir.mkdir(parents=True, exist_ok=True)
        self._locks: Dict[str, FileLock] = {}

    def _get_doc_dir(self, doc_id: str) -> Path:
        """Get the directory for a document's files."""
        return self.docs_dir / doc_id

    def _get_metadata_path(self, doc_id: str) -> Path:
        """Get the path for a document's metadata file."""
        return self._get_doc_dir(doc_id) / "document_info.json"

    def _get_lock(self, doc_id: str) -> FileLock:
        """Get or create a lock for a document."""
        if doc_id not in self._locks:
            lock_path = self._get_doc_dir(doc_id) / "doc.lock"
            self._locks[doc_id] = FileLock(str(lock_path))
        return self._locks[doc_id]

    async def create_document(
        self,
        content: str,
        filename: str,
        model_name: str,
        token_limit: int,
        chunk_size: Optional[int] = None
    ) -> DocumentInfo:
        """
        Create a new document from content.
        
        Args:
            content: Document content
            filename: Original filename
            model_name: Name of the model being used
            token_limit: Token limit for chunks
            chunk_size: Optional custom chunk size
            
        Returns:
            DocumentInfo for the created document
            
        Raises:
            ValueError: If invalid parameters
        """
        if not content:
            raise ValueError("Document content cannot be empty")

        doc_id = str(uuid.uuid4())
        doc_dir = self._get_doc_dir(doc_id)
        doc_dir.mkdir(parents=True)
        lock = self._get_lock(doc_id)

        async with lock:
            try:
                # Create source directory and save original content
                source_dir = doc_dir / "source"
                source_dir.mkdir()
                source_path = source_dir / filename
                
                async with aiofiles.open(source_path, 'w') as f:
                    await f.write(content)

                # Calculate basic document metrics
                file_size = len(content.encode('utf-8'))
                total_lines = content.count('\n') + 1
                content_hash = self.chunk_service.calculate_hash(content)

                # Create initial document metadata
                doc_info = DocumentInfo(
                    id=doc_id,
                    filename=filename,
                    original_path=source_path,
                    total_chunks=0,  # Will be updated after chunking
                    total_tokens=0,  # Will be updated after chunking
                    total_chars=len(content),
                    total_lines=total_lines,
                    model_name=model_name,
                    token_limit=token_limit,
                    md5_hash=content_hash,
                    file_size=file_size,
                    chunks=[]
                )

                # Save initial metadata
                await self._write_doc_metadata(doc_id, doc_info)
                
                logger.info(f"Created document {doc_id} from {filename}")
                return doc_info

            except Exception as e:
                # Cleanup on failure
                if doc_dir.exists():
                    shutil.rmtree(doc_dir)
                logger.error(f"Failed to create document from {filename}: {e}")
                raise

    async def read_document(
        self,
        doc_id: str,
        include_content: bool = False
    ) -> Tuple[Optional[str], DocumentInfo]:
        """
        Read a document's metadata and optionally its content.
        
        Args:
            doc_id: Document identifier
            include_content: Whether to include document content
            
        Returns:
            Tuple of (content or None, metadata)
            
        Raises:
            FileNotFoundError: If document doesn't exist
        """
        doc_dir = self._get_doc_dir(doc_id)
        lock = self._get_lock(doc_id)

        async with lock:
            if not doc_dir.exists():
                raise FileNotFoundError(f"Document {doc_id} not found")

            try:
                # Read metadata
                metadata_path = self._get_metadata_path(doc_id)
                doc_info = await self._read_doc_metadata(doc_id)

                # Read content if requested
                content = None
                if include_content:
                    source_path = Path(doc_info.original_path)
                    if source_path.exists():
                        async with aiofiles.open(source_path, 'r') as f:
                            content = await f.read()
                    else:
                        logger.warning(f"Source file missing for document {doc_id}")

                return content, doc_info

            except Exception as e:
                logger.error(f"Failed to read document {doc_id}: {e}")
                raise

    async def update_document(
        self,
        doc_id: str,
        content: Optional[str] = None,
        metadata_updates: Optional[Dict[str, Any]] = None
    ) -> DocumentInfo:
        """
        Update a document's content and/or metadata.
        
        Args:
            doc_id: Document identifier
            content: New content (optional)
            metadata_updates: Metadata fields to update (optional)
            
        Returns:
            Updated DocumentInfo
            
        Raises:
            FileNotFoundError: If document doesn't exist
        """
        if not content and not metadata_updates:
            raise ValueError("No updates provided")

        lock = self._get_lock(doc_id)
        async with lock:
            # Read current state
            doc_info = await self._read_doc_metadata(doc_id)

            # Update content if provided
            if content is not None:
                source_path = Path(doc_info.original_path)
                if source_path.exists():
                    # Create backup
                    backup_path = source_path.with_suffix('.bak')
                    shutil.copy2(source_path, backup_path)

                    try:
                        # Write new content
                        async with aiofiles.open(source_path, 'w') as f:
                            await f.write(content)

                        # Update basic metrics
                        doc_info.file_size = len(content.encode('utf-8'))
                        doc_info.total_chars = len(content)
                        doc_info.total_lines = content.count('\n') + 1
                        doc_info.md5_hash = self.chunk_service.calculate_hash(content)

                    except Exception:
                        # Restore from backup on failure
                        shutil.copy2(backup_path, source_path)
                        raise
                    finally:
                        if backup_path.exists():
                            backup_path.unlink()

            # Update metadata if provided
            if metadata_updates:
                for key, value in metadata_updates.items():
                    if hasattr(doc_info, key):
                        setattr(doc_info, key, value)

            # Save updated metadata
            await self._write_doc_metadata(doc_id, doc_info)
            logger.info(f"Updated document {doc_id}")
            return doc_info

    async def delete_document(self, doc_id: str) -> bool:
        """
        Delete a document and all its associated data.
        
        Args:
            doc_id: Document identifier
            
        Returns:
            True if deleted, False if not found
        """
        doc_dir = self._get_doc_dir(doc_id)
        lock = self._get_lock(doc_id)

        async with lock:
            try:
                if not doc_dir.exists():
                    return False

                # Get document info for chunk cleanup
                doc_info = await self._read_doc_metadata(doc_id)

                # Delete all associated chunks
                for chunk in doc_info.chunks:
                    await self.chunk_service.delete_chunk(chunk.id)

                # Delete document directory
                shutil.rmtree(doc_dir)

                # Clean up lock
                if doc_id in self._locks:
                    del self._locks[doc_id]

                logger.info(f"Deleted document {doc_id}")
                return True

            except Exception as e:
                logger.error(f"Failed to delete document {doc_id}: {e}")
                return False

    async def _read_doc_metadata(self, doc_id: str) -> DocumentInfo:
        """Read document metadata."""
        metadata_path = self._get_metadata_path(doc_id)
        async with aiofiles.open(metadata_path, 'r') as f:
            metadata_json = await f.read()
            return DocumentInfo.model_validate_json(metadata_json)

    async def _write_doc_metadata(self, doc_id: str, doc_info: DocumentInfo):
        """Write document metadata atomically."""
        metadata_path = self._get_metadata_path(doc_id)
        temp_path = metadata_path.with_suffix('.tmp')

        try:
            async with aiofiles.open(temp_path, 'w') as f:
                await f.write(doc_info.model_dump_json(indent=2))
            temp_path.rename(metadata_path)
        except Exception:
            if temp_path.exists():
                temp_path.unlink()
            raise

    async def list_documents(self) -> List[DocumentInfo]:
        """
        List all documents in the system.
        
        Returns:
            List of DocumentInfo
        """
        docs = []
        for doc_dir in self.docs_dir.iterdir():
            if doc_dir.is_dir():
                try:
                    metadata_path = doc_dir / "document_info.json"
                    if metadata_path.exists():
                        doc_info = await self._read_doc_metadata(doc_dir.name)
                        docs.append(doc_info)
                except Exception as e:
                    logger.error(f"Failed to read metadata from {doc_dir}: {e}")

        return sorted(docs, key=lambda x: x.filename)