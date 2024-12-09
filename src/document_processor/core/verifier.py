# File: src/document_processor/core/verifier.py

import logging
from pathlib import Path
from typing import Dict, Optional

import aiofiles
import json

from .models import DocumentInfo, ChunkInfo

logger = logging.getLogger(__name__)

class Verifier:
    def __init__(self, token_counter):
        self.token_counter = token_counter
    
    async def verify_all_documents(self, output_dir: Path, mode: str) -> Dict[str, bool]:
        """Verifies all documents in the output directory."""
        verification_results = {}
        docs_dir = output_dir

        for doc_dir in docs_dir.iterdir():
            if doc_dir.is_dir():
                doc_info_path = doc_dir / "document_info.json"
                if doc_info_path.exists():
                    async with aiofiles.open(doc_info_path, 'r', encoding='utf-8') as f:
                        doc_info_data = json.loads(await f.read())
                        doc_info = DocumentInfo(**doc_info_data)
                        is_valid = await self.verify_document(doc_info, doc_dir, mode)
                        verification_results[doc_info.id] = is_valid
        return verification_results
    
    async def verify_document(self, doc_info: DocumentInfo, doc_dir: Path, mode: str) -> bool:
        """Verifies a single document."""
        try:
            # Implement verification logic based on mode
            if mode == 'lenient':
                # Lenient verification: Check if all chunks exist
                chunks_dir = doc_dir / "chunks"
                for chunk in doc_info.chunks:
                    chunk_path = chunks_dir / f"{chunk.id}.txt"
                    if not chunk_path.exists():
                        logger.warning(f"Missing chunk file: {chunk_path}")
                        return False
                return True
            elif mode == 'strict':
                # Strict verification: Check token counts and hashes
                chunks_dir = doc_dir / "chunks"
                total_tokens = 0
                for chunk in doc_info.chunks:
                    chunk_path = chunks_dir / f"{chunk.id}.txt"
                    if not chunk_path.exists():
                        logger.warning(f"Missing chunk file: {chunk_path}")
                        return False
                    async with aiofiles.open(chunk_path, 'r', encoding='utf-8') as f:
                        content = await f.read()
                        tokens = self.token_counter.count_tokens(content)
                        if tokens != chunk.tokens:
                            logger.warning(f"Token count mismatch for chunk {chunk.id}: expected {chunk.tokens}, got {tokens}")
                            return False
                        calculated_hash = self.calculate_md5(content)
                        if calculated_hash != chunk.content_hash:
                            logger.warning(f"Hash mismatch for chunk {chunk.id}: expected {chunk.content_hash}, got {calculated_hash}")
                            return False
                        total_tokens += tokens
                if total_tokens != doc_info.total_tokens:
                    logger.warning(f"Total tokens mismatch for document {doc_info.id}: expected {doc_info.total_tokens}, got {total_tokens}")
                    return False
                return True
            elif mode == 'token':
                # Token-only verification: Check total tokens
                chunks_dir = doc_dir / "chunks"
                total_tokens = 0
                for chunk in doc_info.chunks:
                    chunk_path = chunks_dir / f"{chunk.id}.txt"
                    if not chunk_path.exists():
                        logger.warning(f"Missing chunk file: {chunk_path}")
                        return False
                    async with aiofiles.open(chunk_path, 'r', encoding='utf-8') as f:
                        content = await f.read()
                        tokens = self.token_counter.count_tokens(content)
                        total_tokens += tokens
                if total_tokens != doc_info.total_tokens:
                    logger.warning(f"Total tokens mismatch for document {doc_info.id}: expected {doc_info.total_tokens}, got {total_tokens}")
                    return False
                return True
            else:
                logger.warning(f"Unknown verification mode: {mode}")
                return False
        except Exception as e:
            logger.error(f"Error verifying document {doc_info.id}: {e}")
            return False

    @staticmethod
    def calculate_md5(content: str) -> str:
        """Calculates MD5 hash of the given content."""
        import hashlib
        return hashlib.md5(content.encode()).hexdigest()
