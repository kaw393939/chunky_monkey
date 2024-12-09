"""Service for managing system-wide metadata, manifests, and processing state."""

import asyncio
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List
import uuid

import aiofiles
from filelock import FileLock

from ..core.models import (
    ContentManifest,
    ProcessingMetadata,
    ProcessingState,
    DocumentInfo,
    VersionHistoryItem
)
from ..utils import get_logger
from ..utils.config import Config

logger = get_logger(__name__)

class MetadataService:
    """Service for managing system-wide metadata and processing state."""

    def __init__(self, config: Config):
        """
        Initialize the metadata service.
        
        Args:
            config: Application configuration
        """
        self.config = config
        self.metadata_dir = config.output_dir / "metadata"
        self.metadata_dir.mkdir(parents=True, exist_ok=True)
        self._locks: Dict[str, FileLock] = {}

    def _get_manifest_path(self) -> Path:
        """Get the path for the content manifest file."""
        return self.metadata_dir / "manifest.json"

    def _get_processing_path(self, processing_id: str) -> Path:
        """Get the path for a processing metadata file."""
        return self.metadata_dir / f"processing_{processing_id}.json"

    def _get_state_path(self, doc_id: str) -> Path:
        """Get the path for a document's processing state file."""
        return self.metadata_dir / f"state_{doc_id}.json"

    def _get_lock(self, name: str) -> FileLock:
        """Get or create a lock for a metadata file."""
        if name not in self._locks:
            lock_path = self.metadata_dir / f"{name}.lock"
            self._locks[name] = FileLock(str(lock_path))
        return self._locks[name]

    async def create_or_update_manifest(
        self,
        doc_info: DocumentInfo,
        previous_version_id: Optional[str] = None,
        changes_description: Optional[str] = None
    ) -> ContentManifest:
        """
        Create or update the content manifest.
        
        Args:
            doc_info: Document information to add/update
            previous_version_id: Optional ID of previous version
            changes_description: Optional description of changes
            
        Returns:
            Updated ContentManifest
        """
        manifest_path = self._get_manifest_path()
        lock = self._get_lock("manifest")

        async with lock:
            try:
                # Read existing manifest or create new one
                if manifest_path.exists():
                    async with aiofiles.open(manifest_path, 'r') as f:
                        manifest_data = await f.read()
                        manifest = ContentManifest.model_validate_json(manifest_data)
                else:
                    manifest = ContentManifest(
                        manifest_id=str(uuid.uuid4()),
                        created_at=datetime.now(),
                        updated_at=datetime.now(),
                        version_history=[],
                        document_ids=[],
                        total_chunks=0,
                        total_tokens=0,
                        model_name=self.config.model_name,
                        content_hashes=[]
                    )

                # Create version history item
                version = VersionHistoryItem(
                    version_id=str(uuid.uuid4()),
                    parent_version_id=previous_version_id,
                    timestamp=datetime.now(),
                    action="updated" if previous_version_id else "created",
                    changes_description=changes_description
                )
                manifest.version_history.append(version)

                # Update manifest data
                if doc_info.id not in manifest.document_ids:
                    manifest.document_ids.append(doc_info.id)
                    manifest.total_chunks += doc_info.total_chunks
                    manifest.total_tokens += doc_info.total_tokens
                manifest.content_hashes.append(doc_info.md5_hash)
                manifest.updated_at = datetime.now()

                # Write updated manifest
                await self._write_json(manifest_path, manifest.model_dump())
                logger.info(f"Updated manifest with document {doc_info.id}")
                return manifest

            except Exception as e:
                logger.error(f"Failed to update manifest: {e}")
                raise

    async def track_processing(
        self,
        doc_info: DocumentInfo,
        manifest_id: str,
        version_id: str
    ) -> ProcessingMetadata:
        """
        Create and track processing metadata.
        
        Args:
            doc_info: Document being processed
            manifest_id: ID of the content manifest
            version_id: Version ID for this processing
            
        Returns:
            ProcessingMetadata for tracking
        """
        processing_id = str(uuid.uuid4())
        metadata = ProcessingMetadata(
            processing_id=processing_id,
            started_at=datetime.now(),
            manifest_id=manifest_id,
            version_id=version_id,
            document_ids=[doc_info.id],
            chunk_ids=[chunk.id for chunk in doc_info.chunks],
            status="processing",
            processing_stats={
                "model_name": doc_info.model_name,
                "token_limit": doc_info.token_limit,
                "total_chunks": doc_info.total_chunks,
                "total_tokens": doc_info.total_tokens
            }
        )

        metadata_path = self._get_processing_path(processing_id)
        await self._write_json(metadata_path, metadata.model_dump())
        logger.info(f"Created processing metadata {processing_id}")
        return metadata

    async def update_processing_status(
        self,
        processing_id: str,
        status: str,
        error: Optional[str] = None
    ) -> ProcessingMetadata:
        """
        Update the status of a processing operation.
        
        Args:
            processing_id: Processing identifier
            status: New status
            error: Optional error message
            
        Returns:
            Updated ProcessingMetadata
        """
        metadata_path = self._get_processing_path(processing_id)
        lock = self._get_lock(f"processing_{processing_id}")

        async with lock:
            try:
                # Read current metadata
                async with aiofiles.open(metadata_path, 'r') as f:
                    metadata_json = await f.read()
                    metadata = ProcessingMetadata.model_validate_json(metadata_json)

                # Update status
                metadata.status = status
                metadata.error = error
                if status in ["completed", "failed"]:
                    metadata.completed_at = datetime.now()

                # Update processing stats
                if metadata.processing_stats:
                    metadata.processing_stats["end_time"] = datetime.now().isoformat()
                    if metadata.started_at:
                        duration = (datetime.now() - metadata.started_at).total_seconds()
                        metadata.processing_stats["duration"] = duration

                # Write updated metadata
                await self._write_json(metadata_path, metadata.model_dump())
                logger.info(f"Updated processing status to {status} for {processing_id}")
                return metadata

            except Exception as e:
                logger.error(f"Failed to update processing status for {processing_id}: {e}")
                raise

    async def save_processing_state(
        self,
        doc_id: str,
        current_chunk: int,
        processed_chunks: List[str],
        is_complete: bool,
        error_message: Optional[str] = None
    ) -> ProcessingState:
        """
        Save the current processing state for a document.
        
        Args:
            doc_id: Document identifier
            current_chunk: Current chunk number
            processed_chunks: List of processed chunk IDs
            is_complete: Whether processing is complete
            error_message: Optional error message
            
        Returns:
            Saved ProcessingState
        """
        state = ProcessingState(
            doc_id=doc_id,
            current_chunk=current_chunk,
            processed_chunks=processed_chunks,
            is_complete=is_complete,
            error_message=error_message
        )

        state_path = self._get_state_path(doc_id)
        await self._write_json(state_path, state.model_dump())
        logger.info(f"Saved processing state for document {doc_id}")
        return state

    async def read_processing_state(self, doc_id: str) -> Optional[ProcessingState]:
        """
        Read the processing state for a document.
        
        Args:
            doc_id: Document identifier
            
        Returns:
            ProcessingState if found, None otherwise
        """
        state_path = self._get_state_path(doc_id)
        try:
            if state_path.exists():
                async with aiofiles.open(state_path, 'r') as f:
                    state_json = await f.read()
                    return ProcessingState.model_validate_json(state_json)
            return None
        except Exception as e:
            logger.error(f"Failed to read processing state for {doc_id}: {e}")
            return None

    async def get_manifest(self) -> Optional[ContentManifest]:
        """
        Get the current content manifest.
        
        Returns:
            ContentManifest if exists, None otherwise
        """
        manifest_path = self._get_manifest_path()
        try:
            if manifest_path.exists():
                async with aiofiles.open(manifest_path, 'r') as f:
                    manifest_json = await f.read()
                    return ContentManifest.model_validate_json(manifest_json)
            return None
        except Exception as e:
            logger.error(f"Failed to read manifest: {e}")
            return None

    async def get_processing_metadata(self, processing_id: str) -> Optional[ProcessingMetadata]:
        """
        Get processing metadata by ID.
        
        Args:
            processing_id: Processing identifier
            
        Returns:
            ProcessingMetadata if found, None otherwise
        """
        metadata_path = self._get_processing_path(processing_id)
        try:
            if metadata_path.exists():
                async with aiofiles.open(metadata_path, 'r') as f:
                    metadata_json = await f.read()
                    return ProcessingMetadata.model_validate_json(metadata_json)
            return None
        except Exception as e:
            logger.error(f"Failed to read processing metadata for {processing_id}: {e}")
            return None

    async def _write_json(self, path: Path, data: Dict[str, Any]):
        """Write JSON data atomically with a temporary file."""
        temp_path = path.with_suffix('.tmp')
        try:
            async with aiofiles.open(temp_path, 'w') as f:
                await f.write(json.dumps(data, indent=2))
            temp_path.rename(path)
        except Exception:
            if temp_path.exists():
                temp_path.unlink()
            raise

    async def cleanup_old_metadata(self, age_days: int = 7):
        """
        Clean up old processing metadata and states.
        
        Args:
            age_days: Age in days after which to clean up
        """
        cutoff = datetime.now() - timedelta(days=age_days)
        
        try:
            # Clean up processing metadata
            for path in self.metadata_dir.glob("processing_*.json"):
                try:
                    metadata = await self.get_processing_metadata(path.stem[11:])  # Remove 'processing_' prefix
                    if metadata and metadata.started_at and metadata.started_at < cutoff:
                        if metadata.status in ["completed", "failed"]:
                            path.unlink()
                            logger.info(f"Cleaned up old processing metadata: {path.name}")
                except Exception as e:
                    logger.error(f"Failed to clean up {path}: {e}")

            # Clean up processing states
            for path in self.metadata_dir.glob("state_*.json"):
                try:
                    state = await self.read_processing_state(path.stem[6:])  # Remove 'state_' prefix
                    if state and state.timestamp < cutoff and state.is_complete:
                        path.unlink()
                        logger.info(f"Cleaned up old processing state: {path.name}")
                except Exception as e:
                    logger.error(f"Failed to clean up {path}: {e}")

        except Exception as e:
            logger.error(f"Failed to clean up old metadata: {e}")