# File: src/document_processor/services/chunk_service.py

import datetime
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Union

from ..core.models import ChunkMetadata, VersionHistoryItem
from ..utils.logging import get_logger
from ..core.models import DateTimeEncoder
from filelock import FileLock
import aiofiles
import json

logger = get_logger(__name__)

# -----------------------------------------------------------------------------
# CHUNK SERVICE
# -----------------------------------------------------------------------------
class ChunkService:
    def __init__(self, chunk_dir: Path):
        self.chunk_dir = chunk_dir
        if not self.chunk_dir.exists():
            self.chunk_dir.mkdir(parents=True, exist_ok=True)

    async def create_chunk(self, chunk_data: ChunkMetadata):
        """Creates a new chunk asynchronously."""
        chunk_path = self.chunk_dir / f"{chunk_data.id}.json"
        if chunk_path.exists():
            logger.error(f"Chunk {chunk_data.id} already exists.")
            raise FileExistsError(f"Chunk {chunk_data.id} already exists.")
        chunk_data.version_history.append(VersionHistoryItem(
            version_id=chunk_data.version_id,
            parent_version_id=None,
            timestamp=datetime.now().isoformat(),
            action="created",
            details=None
        ))
        await self._write_json(chunk_path, chunk_data.model_dump())
        logger.info(f"Chunk {chunk_data.id} created.")

    async def read_chunk(self, chunk_id: str) -> Optional[Dict[str, Any]]:
        """Reads and returns chunk metadata asynchronously."""
        chunk_path = self.chunk_dir / f"{chunk_id}.json"
        if not chunk_path.exists():
            logger.error(f"Chunk {chunk_id} does not exist.")
            return None
        try:
            async with aiofiles.open(chunk_path, 'r', encoding='utf-8') as f:
                chunk_data = json.loads(await f.read())
                return chunk_data
        except Exception as e:
            logger.error(f"Error reading chunk {chunk_id}: {e}")
            return None

    async def update_chunk(self, chunk_id: str, update_data: Dict[str, Union[str, int, dict]]):
        """Updates an existing chunk asynchronously."""
        chunk_path = self.chunk_dir / f"{chunk_id}.json"
        if not chunk_path.exists():
            logger.error(f"Chunk {chunk_id} does not exist.")
            raise FileNotFoundError(f"Chunk {chunk_id} does not exist.")

        lock_path = f"{chunk_path}.lock"
        lock = FileLock(lock_path)
        try:
            with lock:
                async with aiofiles.open(chunk_path, 'r', encoding='utf-8') as f:
                    chunk_data = json.loads(await f.read())
                
                # Update fields
                for key, value in update_data.items():
                    if hasattr(chunk_data, key):
                        setattr(chunk_data, key, value)
                    else:
                        chunk_data[key] = value
                
                # Add version history
                new_version_id = datetime.now().isoformat()
                chunk_data['version_history'].append(VersionHistoryItem(
                    version_id=new_version_id,
                    parent_version_id=chunk_data.get('version_id'),
                    timestamp=datetime.now().isoformat(),
                    action="updated",
                    details=update_data
                ).model_dump())

                chunk_data['version_id'] = new_version_id

                await self._write_json(chunk_path, chunk_data)
                logger.info(f"Chunk {chunk_id} updated.")
        except Exception as e:
            logger.error(f"Error updating chunk {chunk_id}: {e}")
            raise

    async def delete_chunk(self, chunk_id: str):
        """Deletes a chunk asynchronously with file locking."""
        chunk_path = self.chunk_dir / f"{chunk_id}.json"
        lock_path = f"{chunk_path}.lock"
        lock = FileLock(lock_path)
        try:
            with lock:
                if chunk_path.exists():
                    await aiofiles.os.remove(chunk_path)
                    logger.info(f"Chunk {chunk_id} deleted.")
                else:
                    logger.warning(f"Chunk {chunk_id} does not exist.")
        except Exception as e:
            logger.error(f"Error deleting chunk {chunk_id}: {e}")
            raise

    async def _write_json(self, path: Path, data: dict):
        """Writes a dictionary to a JSON file asynchronously."""
        async with aiofiles.open(path, 'w', encoding='utf-8') as f:
            json_str = json.dumps(data, indent=2, cls=DateTimeEncoder)
            await f.write(json_str)
