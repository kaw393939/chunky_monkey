# File: src/document_processor/services/document_service.py

import datetime
import logging
from pathlib import Path
from typing import Dict, Any, Optional

from ..core.models import DocumentMetadata, VersionHistoryItem
from ..utils.logging import get_logger
from ..core.models import DateTimeEncoder
from filelock import FileLock
import aiofiles
import json

logger = get_logger(__name__)

# -----------------------------------------------------------------------------
# DOCUMENT SERVICE
# -----------------------------------------------------------------------------
class DocumentService:
    def __init__(self, doc_info_path: Path):
        self.doc_info_path = doc_info_path

    async def read_document(self) -> Optional[DocumentMetadata]:
        """Reads and returns document metadata asynchronously."""
        if not self.doc_info_path.exists():
            logger.error(f"Document info not found: {self.doc_info_path}")
            return None
        try:
            async with aiofiles.open(self.doc_info_path, 'r', encoding='utf-8') as f:
                doc_data = json.loads(await f.read())
                return DocumentMetadata(**doc_data)
        except Exception as e:
            logger.error(f"Error reading document metadata: {e}")
            return None

    async def update_document(self, chunk_update: Dict[str, Any]):
        """Updates document metadata with chunk references asynchronously."""
        if not self.doc_info_path.exists():
            logger.error(f"Document info not found: {self.doc_info_path}")
            raise FileNotFoundError(f"{self.doc_info_path} not found.")

        lock_path = f"{self.doc_info_path}.lock"
        lock = FileLock(lock_path)
        try:
            with lock:
                async with aiofiles.open(self.doc_info_path, 'r', encoding='utf-8') as f:
                    doc_data = json.loads(await f.read())
                    doc_metadata = DocumentMetadata(**doc_data)

                # Create a new version
                new_version_id = datetime.now().isoformat()
                doc_metadata.version_history.append(VersionHistoryItem(
                    version_id=new_version_id,
                    parent_version_id=doc_metadata.version_id,
                    timestamp=datetime.now().isoformat(),
                    action="chunk_updated",
                    details=chunk_update
                ).model_dump())

                doc_metadata.version_id = new_version_id

                # Apply the updates
                for chunk in doc_metadata.chunks:
                    if chunk["id"] == chunk_update["id"]:
                        chunk.update(chunk_update)
                        break
                else:
                    doc_metadata.chunks.append(chunk_update)

                await self._write_json(self.doc_info_path, doc_metadata.model_dump())
                logger.info(f"Document metadata updated for chunk {chunk_update['id']}.")
        except Exception as e:
            logger.error(f"Error updating document metadata: {e}")
            raise

    async def _write_json(self, path: Path, data: dict):
        """Writes a dictionary to a JSON file asynchronously."""
        async with aiofiles.open(path, 'w', encoding='utf-8') as f:
            json_str = json.dumps(data, indent=2, cls=DateTimeEncoder)
            await f.write(json_str)
