"""Main document processor coordinating all services and operations."""

import asyncio
import uuid
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple
from datetime import datetime

import aiofiles  # Add this import

from ..core.models import (
    DocumentInfo,
    ProcessingProgress,
    ProcessingState,
    ChunkMetadata
)
from ..core.tokenizer import Chunker
from ..core.verifier import Verifier
from ..services import ChunkService, DocumentService, MetadataService
from ..utils import get_logger
from ..utils.config import Config

logger = get_logger(__name__)

class ProcessingError(Exception):
    """Custom exception for processing errors."""
    pass

class DocumentProcessor:
    """Main processor coordinating document operations."""

    def __init__(
        self,
        config: Config,
        chunk_service: Optional[ChunkService] = None,
        doc_service: Optional[DocumentService] = None,
        metadata_service: Optional[MetadataService] = None,
        chunker: Optional[Chunker] = None,
        verifier: Optional[Verifier] = None
    ):
        """Initialize the document processor."""
        self.config = config
        
        # Initialize services if not provided
        self.chunk_service = chunk_service or ChunkService(config)
        self.doc_service = doc_service or DocumentService(config, self.chunk_service)
        self.metadata_service = metadata_service or MetadataService(config)
        self.chunker = chunker or Chunker(config)
        self.verifier = verifier or Verifier(
            config,
            self.chunk_service,
            self.doc_service,
            self.metadata_service
        )

        # Processing state
        self._processing_semaphore = asyncio.Semaphore(config.max_concurrent_files)
        self._active_tasks: Dict[str, asyncio.Task] = {}

    async def process_document(
        self,
        input_path: Path,
        verify: bool = True,
        verification_mode: str = 'strict'
    ) -> DocumentInfo:
        """Process a single document."""
        async with self._processing_semaphore:
            doc_id = str(uuid.uuid4())
            state = None
            
            try:
                # Initialize processing state
                state = ProcessingState(
                    doc_id=doc_id,
                    current_chunk=0,
                    processed_chunks=[],
                    is_complete=False
                )
                await self.metadata_service.save_processing_state(
                    doc_id=doc_id,
                    current_chunk=0,
                    processed_chunks=[],
                    is_complete=False
                )

                # Read input file
                async with aiofiles.open(input_path, 'r', encoding='utf-8') as f:
                    content = await f.read()

                # Create initial document
                doc_info = await self.doc_service.create_document(
                    content=content,
                    filename=input_path.name,
                    model_name=self.config.model_name,
                    token_limit=self.config.token_limit
                )

                # Chunk document
                chunks = await self.chunker.chunk_document(content, doc_id)
                
                # Process chunks
                processed_chunks = []
                for chunk_number, (chunk_text, chunk_metadata) in enumerate(chunks):
                    try:
                        # Update state
                        state.current_chunk = chunk_number
                        await self.metadata_service.save_processing_state(
                            doc_id=doc_id,
                            current_chunk=chunk_number,
                            processed_chunks=state.processed_chunks,
                            is_complete=False
                        )

                        # Process chunk
                        await self.chunk_service.create_chunk(
                            chunk_id=chunk_metadata.id,
                            content=chunk_text,
                            doc_id=doc_id,
                            number=chunk_number,
                            token_count=chunk_metadata.tokens
                        )

                        processed_chunks.append(chunk_metadata)
                        state.processed_chunks.append(chunk_metadata.id)

                    except Exception as e:
                        logger.error(f"Error processing chunk {chunk_number}: {e}")
                        raise ProcessingError(f"Chunk processing failed: {e}")

                # Update document with processed chunks
                doc_info.chunks = processed_chunks
                doc_info.total_chunks = len(processed_chunks)
                doc_info.total_tokens = sum(c.tokens for c in processed_chunks)

                # Create manifest entry
                manifest = await self.metadata_service.create_or_update_manifest(doc_info)
                doc_info.manifest_id = manifest.manifest_id

                # Verify if requested
                if verify:
                    verification_result = await self.verifier.verify_document(
                        doc_id,
                        mode=verification_mode
                    )
                    if not verification_result.is_valid:
                        raise ProcessingError(
                            f"Verification failed: {verification_result.error_message}"
                        )

                # Mark processing as complete
                state.is_complete = True
                await self.metadata_service.save_processing_state(
                    doc_id=doc_id,
                    current_chunk=len(processed_chunks),
                    processed_chunks=state.processed_chunks,
                    is_complete=True
                )

                return doc_info

            except Exception as e:
                logger.error(f"Processing failed for {input_path}: {e}")
                if state:
                    await self.metadata_service.save_processing_state(
                        doc_id=doc_id,
                        current_chunk=state.current_chunk,
                        processed_chunks=state.processed_chunks,
                        is_complete=False,
                        error_message=str(e)
                    )
                raise ProcessingError(f"Document processing failed: {e}")

    # Add other methods here (process_directory, get_document, etc.)
    # as defined in the previous processor implementation