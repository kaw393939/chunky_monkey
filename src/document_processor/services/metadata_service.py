# File: src/document_processor/services/metadata_service.py

import json
import logging
from pathlib import Path
from typing import Optional, Dict
import uuid
from datetime import datetime

from ..core.models import DocumentInfo, ContentManifest, ContentVersion, ProcessingMetadata, DateTimeEncoder
from ..utils.config import AppConfig
import aiofiles

logger = logging.getLogger(__name__)

class MetadataManager:
    def __init__(self, model_name: str, config: AppConfig):
        self.model_name = model_name
        self.config = config

    async def create_or_update_manifest(
        self,
        doc_info: DocumentInfo,
        output_dir: Path,
        previous_version_id: Optional[str] = None,
        changes_description: Optional[str] = None
    ) -> ContentManifest:
        """Creates or updates the content manifest."""
        manifest_path = output_dir / "manifest.json"
        manifest_exists = manifest_path.exists()

        if manifest_exists:
            async with aiofiles.open(manifest_path, 'r') as f:
                manifest_data = json.loads(await f.read())
                manifest = ContentManifest(**manifest_data)
        else:
            manifest = ContentManifest(
                manifest_id=str(uuid.uuid4()),
                created_at=datetime.now().isoformat(),
                updated_at=datetime.now().isoformat(),
                version_history=[],
                document_ids=[],
                total_chunks=0,
                total_tokens=0,
                model_name=self.model_name,
                content_hashes=[],
                metadata={}
            )

        version = ContentVersion(
            version_id=str(uuid.uuid4()),
            created_at=datetime.now().isoformat(),
            content_hash=doc_info.md5_hash,
            parent_version_id=previous_version_id,
            changes_description=changes_description
        )
        
        manifest.version_history.append(version)
        manifest.document_ids.append(doc_info.id)
        manifest.total_chunks += doc_info.total_chunks
        manifest.total_tokens += doc_info.total_tokens
        manifest.content_hashes.append(doc_info.md5_hash)
        manifest.updated_at = datetime.now().isoformat()

        await self._write_json(manifest_path, manifest.model_dump())
        return manifest

    async def track_processing(
        self,
        manifest: ContentManifest,
        doc_info: DocumentInfo,
        output_dir: Path
    ) -> ProcessingMetadata:
        """Tracks the processing metadata."""
        processing_stats = {
            "start_time": datetime.now().isoformat(),
            "total_tokens": doc_info.total_tokens,
            "total_chunks": doc_info.total_chunks,
            "model_name": doc_info.model_name,
            "token_limit": doc_info.token_limit
        }

        metadata = ProcessingMetadata(
            processing_id=str(uuid.uuid4()),
            started_at=datetime.now().isoformat(),
            completed_at=None,
            manifest_id=manifest.manifest_id,
            version_id=manifest.version_history[-1].version_id,
            document_ids=[doc_info.id],
            chunk_ids=[chunk.id for chunk in doc_info.chunks],
            status="processing",
            error=None,
            processing_stats=processing_stats
        )
        
        metadata_path = output_dir / f"processing_{metadata.processing_id}.json"
        await self._write_json(metadata_path, metadata.model_dump())
        
        return metadata

    async def update_processing_status(
        self,
        metadata: ProcessingMetadata,
        output_dir: Path,
        status: str,
        error: Optional[str] = None
    ):
        """Updates the processing status."""
        metadata.status = status
        metadata.error = error
        if status in ["completed", "failed"]:
            metadata.completed_at = datetime.now().isoformat()
            if metadata.processing_stats:
                start_time = datetime.fromisoformat(metadata.processing_stats["start_time"])
                end_time = datetime.fromisoformat(metadata.completed_at)
                metadata.processing_stats["end_time"] = metadata.completed_at
                metadata.processing_stats["duration"] = (end_time - start_time).total_seconds()
        
        metadata_path = output_dir / f"processing_{metadata.processing_id}.json"
        await self._write_json(metadata_path, metadata.model_dump())

    @staticmethod
    async def _write_json(path: Path, data: dict):
        """Writes a dictionary to a JSON file asynchronously."""
        async with aiofiles.open(path, 'w', encoding='utf-8') as f:
            json_str = json.dumps(data, indent=2, cls=DateTimeEncoder)
            await f.write(json_str)
