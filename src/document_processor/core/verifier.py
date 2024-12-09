"""Verification service for document and chunk integrity."""

import difflib
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import asyncio

import aiofiles

from ..services import ChunkService, DocumentService, MetadataService
from ..core.models import DocumentInfo, ChunkMetadata, ProcessingState
from ..utils import get_logger
from ..utils.config import Config

logger = get_logger(__name__)

class VerificationResult:
    """Results of a verification operation."""
    
    def __init__(self):
        self.is_valid = False
        self.similarity_ratio: float = 0.0
        self.token_difference: int = 0
        self.error_message: Optional[str] = None
        self.differences: List[Tuple[int, str, str]] = []  # [(line_num, original, reconstructed)]
        self.missing_chunks: List[str] = []
        self.corrupt_chunks: List[str] = []

class Verifier:
    """Service for verifying document and chunk integrity."""

    def __init__(
        self,
        config: Config,
        chunk_service: ChunkService,
        doc_service: DocumentService,
        metadata_service: MetadataService
    ):
        """Initialize the verifier with required services."""
        self.config = config
        self.chunk_service = chunk_service
        self.doc_service = doc_service
        self.metadata_service = metadata_service
        
        # Verification thresholds
        self.strict_threshold = config.STRICT_SIMILARITY_THRESHOLD
        self.lenient_threshold = config.LENIENT_SIMILARITY_THRESHOLD
        self.token_threshold = config.TOKEN_SIMILARITY_THRESHOLD
        self.max_token_difference = config.MAX_ALLOWED_TOKEN_DIFFERENCE

    async def verify_document(
        self,
        doc_id: str,
        mode: str = 'strict',
        repair: bool = False
    ) -> VerificationResult:
        """
        Verify a document's integrity.
        
        Args:
            doc_id: Document identifier
            mode: Verification mode ('strict', 'lenient', or 'token')
            repair: Whether to attempt repairs on verification failures
            
        Returns:
            VerificationResult with details
        """
        result = VerificationResult()
        
        try:
            # Get document content and metadata
            content, doc_info = await self.doc_service.read_document(doc_id, include_content=True)
            if not content or not doc_info:
                result.error_message = f"Document {doc_id} not found or empty"
                return result

            # Reconstruct from chunks
            reconstructed, chunk_results = await self._reconstruct_from_chunks(doc_info)
            if not reconstructed:
                result.error_message = "Failed to reconstruct document from chunks"
                return result

            # Track missing or corrupt chunks
            result.missing_chunks = chunk_results['missing']
            result.corrupt_chunks = chunk_results['corrupt']

            # Calculate similarity metrics
            result.similarity_ratio = self._calculate_similarity(content, reconstructed)
            result.token_difference = await self._calculate_token_difference(content, reconstructed)
            result.differences = self._find_differences(content, reconstructed)

            # Verify based on mode
            result.is_valid = await self._verify_by_mode(
                mode,
                result.similarity_ratio,
                result.token_difference
            )

            # Attempt repair if needed and requested
            if repair and not result.is_valid:
                await self._attempt_repair(doc_info, result)

            return result

        except Exception as e:
            logger.error(f"Verification failed for document {doc_id}: {e}")
            result.error_message = str(e)
            return result

    async def verify_chunk(self, chunk_id: str) -> bool:
        """
        Verify a single chunk's integrity.
        
        Args:
            chunk_id: Chunk identifier
            
        Returns:
            True if valid, False otherwise
        """
        try:
            content, metadata = await self.chunk_service.read_chunk(chunk_id)
            current_hash = self.chunk_service.calculate_hash(content)
            return current_hash == metadata.content_hash
        except Exception as e:
            logger.error(f"Chunk verification failed for {chunk_id}: {e}")
            return False

    async def verify_all_documents(
        self,
        mode: str = 'strict',
        parallel: bool = True
    ) -> Dict[str, VerificationResult]:
        """
        Verify all documents in the system.
        
        Args:
            mode: Verification mode
            parallel: Whether to verify in parallel
            
        Returns:
            Dictionary mapping document IDs to their verification results
        """
        manifest = await self.metadata_service.get_manifest()
        if not manifest:
            logger.warning("No manifest found")
            return {}

        async def verify_single(doc_id: str) -> Tuple[str, VerificationResult]:
            result = await self.verify_document(doc_id, mode)
            return doc_id, result

        if parallel:
            tasks = [verify_single(doc_id) for doc_id in manifest.document_ids]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            return {doc_id: result for doc_id, result in results if not isinstance(result, Exception)}
        else:
            results = {}
            for doc_id in manifest.document_ids:
                results[doc_id] = await self.verify_document(doc_id, mode)
            return results

    async def _reconstruct_from_chunks(
        self,
        doc_info: DocumentInfo
    ) -> Tuple[Optional[str], Dict[str, List[str]]]:
        """Reconstruct document content from chunks."""
        results = {'missing': [], 'corrupt': []}
        chunks_content = []

        for chunk in sorted(doc_info.chunks, key=lambda x: x.number):
            try:
                is_valid = await self.verify_chunk(chunk.id)
                if not is_valid:
                    results['corrupt'].append(chunk.id)
                    continue

                content, _ = await self.chunk_service.read_chunk(chunk.id)
                chunks_content.append(content)
            except Exception as e:
                logger.error(f"Failed to read chunk {chunk.id}: {e}")
                results['missing'].append(chunk.id)

        if not chunks_content:
            return None, results
        
        return ''.join(chunks_content), results

    def _calculate_similarity(self, original: str, reconstructed: str) -> float:
        """Calculate similarity ratio between original and reconstructed text."""
        return difflib.SequenceMatcher(None, original, reconstructed).ratio()

    async def _calculate_token_difference(self, original: str, reconstructed: str) -> int:
        """Calculate token count difference between original and reconstructed text."""
        return abs(
            self.config.token_counter.count_tokens(original) -
            self.config.token_counter.count_tokens(reconstructed)
        )

    def _find_differences(
        self,
        original: str,
        reconstructed: str,
        context_lines: int = 3
    ) -> List[Tuple[int, str, str]]:
        """Find differences between original and reconstructed text."""
        differences = []
        orig_lines = original.splitlines()
        recon_lines = reconstructed.splitlines()

        for i, (orig, recon) in enumerate(zip(orig_lines, recon_lines)):
            if orig != recon:
                context_start = max(0, i - context_lines)
                context_end = min(len(orig_lines), i + context_lines + 1)
                differences.append((i, orig, recon))
                
        return differences

    async def _verify_by_mode(
        self,
        mode: str,
        similarity: float,
        token_difference: int
    ) -> bool:
        """Verify content based on specified mode."""
        if mode == 'strict':
            return similarity >= self.strict_threshold
        elif mode == 'lenient':
            return similarity >= self.lenient_threshold
        elif mode == 'token':
            return (
                token_difference <= self.max_token_difference and
                similarity >= self.token_threshold
            )
        else:
            raise ValueError(f"Unknown verification mode: {mode}")

    async def _attempt_repair(
        self,
        doc_info: DocumentInfo,
        result: VerificationResult
    ) -> None:
        """Attempt to repair document integrity issues."""
        if result.missing_chunks or result.corrupt_chunks:
            # Log repair attempt
            logger.info(f"Attempting repair for document {doc_info.id}")
            
            # Save current state
            await self.metadata_service.save_processing_state(
                doc_id=doc_info.id,
                current_chunk=0,
                processed_chunks=[],
                is_complete=False,
                error_message="Repair initiated"
            )

            try:
                # Reprocess corrupt chunks
                for chunk_id in result.corrupt_chunks:
                    chunk = doc_info.get_chunk_by_id(chunk_id)
                    if chunk:
                        # Re-read original content for this chunk
                        content, _ = await self.doc_service.read_document(
                            doc_info.id,
                            include_content=True
                        )
                        if content:
                            # Calculate chunk bounds and reprocess
                            start_pos = chunk.character_count * chunk.number
                            end_pos = start_pos + chunk.character_count
                            chunk_content = content[start_pos:end_pos]
                            
                            # Update chunk
                            await self.chunk_service.update_chunk(
                                chunk_id=chunk_id,
                                content=chunk_content
                            )

                # Update processing state
                await self.metadata_service.save_processing_state(
                    doc_id=doc_info.id,
                    current_chunk=len(doc_info.chunks),
                    processed_chunks=[c.id for c in doc_info.chunks],
                    is_complete=True
                )

            except Exception as e:
                logger.error(f"Repair failed for document {doc_info.id}: {e}")
                await self.metadata_service.save_processing_state(
                    doc_id=doc_info.id,
                    current_chunk=0,
                    processed_chunks=[],
                    is_complete=False,
                    error_message=f"Repair failed: {str(e)}"
                )