# File: src/document_processor/core/processor.py

import asyncio
import hashlib
import json
import logging
import shutil
from pathlib import Path
from typing import List, Tuple, Optional
import difflib
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import cpu_count
from asyncio import Semaphore
import uuid

from .models import (
    DocumentInfo,
    ChunkInfo,
    ContentManifest,
    ContentVersion,
    ProcessingMetadata,
    DateTimeEncoder,
    ProcessingState
)
from document_processor.core.errors import ProcessingError

from ..utils.config import AppConfig, MODEL_CONFIGS
from ..services.metadata_service import MetadataManager
from ..core.verifier import Verifier

import aiofiles
import tiktoken
import spacy

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# TOKEN COUNTER PROTOCOL
# -----------------------------------------------------------------------------

class TokenCounter:
    def count_tokens(self, text: str) -> int:
        raise NotImplementedError

class TiktokenCounter(TokenCounter):
    def __init__(self, model_encoding: str):
        self.encoding = tiktoken.get_encoding(model_encoding)
    
    def count_tokens(self, text: str) -> int:
        return len(self.encoding.encode(text))

class SpacyTokenCounter(TokenCounter):
    def __init__(self, nlp):
        self.nlp = nlp
    
    def count_tokens(self, text: str) -> int:
        return len(self.nlp.tokenizer(text))

# -----------------------------------------------------------------------------
# PROCESSING PROGRESS TRACKER
# -----------------------------------------------------------------------------

class ProcessingProgress:
    def __init__(
        self, 
        total_files: int, 
        processed_files: int = 0, 
        current_file: str = "", 
        start_time: Optional[datetime] = None, 
        processed_chunks: int = 0, 
        total_tokens: int = 0, 
        current_chunk: int = 0
    ):
        self.total_files = total_files
        self.processed_files = processed_files
        self.current_file = current_file
        self.start_time = start_time or datetime.now()
        self.processed_chunks = processed_chunks
        self.total_tokens = total_tokens
        self.current_chunk = current_chunk
        self.lock = asyncio.Lock()
    
    async def update_file_progress(self, file_name: str):
        async with self.lock:
            self.current_file = file_name
    
    async def increment_processed_files(self):
        async with self.lock:
            self.processed_files += 1
            self.current_file = ""
    
    async def increment_chunks_tokens(self, chunks: int, tokens: int):
        async with self.lock:
            self.processed_chunks += chunks
            self.total_tokens += tokens
    
    async def monitor(self):
        """Periodically logs the current processing progress."""
        try:
            while True:
                async with self.lock:
                    elapsed = datetime.now() - self.start_time
                    logger.info(
                        f"Processed {self.processed_files}/{self.total_files} files. "
                        f"Current file: {self.current_file}. "
                        f"Processed chunks: {self.processed_chunks}. "
                        f"Total tokens: {self.total_tokens}. "
                        f"Elapsed time: {elapsed}"
                    )
                await asyncio.sleep(5)  # Adjust the interval as needed
        except asyncio.CancelledError:
            logger.debug("Progress monitoring task cancelled.")

# -----------------------------------------------------------------------------
# CHUNKING LOGIC (CPU-BOUND)
# -----------------------------------------------------------------------------

def chunk_text_synchronously(text: str, doc_id: str, token_limit: int, token_counter: TokenCounter) -> List[Tuple[str, ChunkInfo]]:
    def calculate_md5(content: str) -> str:
        return hashlib.md5(content.encode()).hexdigest()

    if isinstance(token_counter, SpacyTokenCounter):
        doc = token_counter.nlp(text)
        sentences = [sent.text_with_ws for sent in doc.sents]
    else:
        sentences = text.split('.')
        sentences = [s.strip() + '.' for s in sentences if s.strip()]

    def split_large_sentence(sentence: str) -> List[str]:
        to_process = [sentence]
        results = []
        while to_process:
            seg = to_process.pop()
            seg_tokens = token_counter.count_tokens(seg)
            if seg_tokens <= token_limit:
                results.append(seg)
            else:
                mid = len(seg) // 2
                while mid > 0 and not seg[mid].isspace():
                    mid -= 1
                if mid == 0:
                    # Cannot split, force split
                    mid = len(seg) // 2
                left = seg[:mid].strip()
                right = seg[mid:].strip()
                if left:
                    to_process.append(left)
                if right:
                    to_process.append(right)
        return results

    chunks: List[Tuple[str, ChunkInfo]] = []
    chunk_number = 1  # Start from 1 for readability
    buffer = []
    buffer_tokens = 0

    for sentence in sentences:
        sentence_tokens = token_counter.count_tokens(sentence)
        if sentence_tokens > token_limit:
            subsentences = split_large_sentence(sentence)
            for subsent in subsentences:
                subsent_tokens = token_counter.count_tokens(subsent)
                if buffer_tokens + subsent_tokens > token_limit:
                    if buffer:
                        chunk_text = ''.join(buffer)
                        chunk_hash = calculate_md5(chunk_text)
                        chunk_info = ChunkInfo(
                            id=f"{doc_id}-chunk-{chunk_number}",
                            number=chunk_number,
                            tokens=token_counter.count_tokens(chunk_text),
                            doc_id=doc_id,
                            content_hash=chunk_hash,
                            character_count=len(chunk_text)
                        )
                        chunks.append((chunk_text, chunk_info))
                        chunk_number += 1
                        buffer = []
                        buffer_tokens = 0
                buffer.append(subsent)
                buffer_tokens += subsent_tokens
        else:
            if buffer_tokens + sentence_tokens <= token_limit:
                buffer.append(sentence)
                buffer_tokens += sentence_tokens
            else:
                if buffer:
                    chunk_text = ''.join(buffer)
                    chunk_hash = calculate_md5(chunk_text)
                    chunk_info = ChunkInfo(
                        id=f"{doc_id}-chunk-{chunk_number}",
                        number=chunk_number,
                        tokens=token_counter.count_tokens(chunk_text),
                        doc_id=doc_id,
                        content_hash=chunk_hash,
                        character_count=len(chunk_text)
                    )
                    chunks.append((chunk_text, chunk_info))
                    chunk_number += 1
                buffer = [sentence]
                buffer_tokens = sentence_tokens

    if buffer:
        chunk_text = ''.join(buffer)
        chunk_hash = calculate_md5(chunk_text)
        chunk_info = ChunkInfo(
            id=f"{doc_id}-chunk-{chunk_number}",
            number=chunk_number,
            tokens=token_counter.count_tokens(chunk_text),
            doc_id=doc_id,
            content_hash=chunk_hash,
            character_count=len(chunk_text)
        )
        chunks.append((chunk_text, chunk_info))

    return chunks

# -----------------------------------------------------------------------------
# TEXT PROCESSOR
# -----------------------------------------------------------------------------

class TextProcessor:
    def __init__(
        self, 
        config: AppConfig,
        chunk_reduction_factor: float = 1.0  # New parameter with default
    ):
        model_name = config.model_name
        if model_name not in MODEL_CONFIGS:
            raise ValueError(f"Unsupported model. Choose from: {', '.join(MODEL_CONFIGS.keys())}")
        
        self.model_name = model_name
        self.model_config = config.model_configs[model_name]
        base_limit = self.model_config['tokens']
        self.chunk_reduction_factor = chunk_reduction_factor
        self.token_limit = int(base_limit * self.chunk_reduction_factor)

        try:
            self.token_counter = TiktokenCounter(self.model_config['encoding'])
            logger.info("Using tiktoken for token counting")
        except Exception as e:
            logger.warning(f"Failed to initialize tiktoken: {e}. Falling back to spaCy")
            nlp = spacy.load(config.spacy_model, disable=['ner', 'parser', 'attribute_ruler', 'lemmatizer'])
            if 'sentencizer' not in nlp.pipe_names:
                nlp.add_pipe('sentencizer')
            self.token_counter = SpacyTokenCounter(nlp)
        
        self.num_workers = min(32, cpu_count() * 2)
        self.process_pool = ProcessPoolExecutor(max_workers=self.num_workers)
        self.progress_tracker = ProcessingProgress(
            total_files=0,
            processed_files=0,
            current_file="",
            start_time=datetime.now(),
            processed_chunks=0,
            total_tokens=0,
            current_chunk=0
        )
        self.max_concurrent_files = config.max_concurrent_files
        self.metadata_manager = MetadataManager(model_name=self.model_name, config=config)
        self.verifier = Verifier(token_counter=self.token_counter)
    
    async def _write_json(self, path: Path, data: dict):
        """Writes a dictionary to a JSON file asynchronously."""
        try:
            async with aiofiles.open(path, 'w', encoding='utf-8') as f:
                json_str = json.dumps(data, indent=2, cls=DateTimeEncoder)
                await f.write(json_str)
        except Exception as e:
            logger.error(f"Error writing JSON to {path}: {e}")
            raise ProcessingError(f"Error writing JSON to {path}: {e}") from e
    
    async def _write_chunk(self, chunk_text: str, chunk_info: ChunkInfo, doc_dir: Path):
        """Writes chunk text and metadata to files."""
        try:
            chunks_dir = doc_dir / "chunks"
            chunks_dir.mkdir(exist_ok=True)
            
            chunk_path = chunks_dir / f"{chunk_info.id}.txt"
            meta_path = chunks_dir / f"{chunk_info.id}.json"
            
            async with aiofiles.open(chunk_path, 'w', encoding='utf-8') as f:
                await f.write(chunk_text)
            
            await self._write_json(meta_path, chunk_info.model_dump())
        except Exception as e:
            logger.error(f"Error writing chunk {chunk_info.id}: {e}")
            raise ProcessingError(f"Error writing chunk {chunk_info.id}: {e}") from e
    
    async def _save_processing_state(self, state: 'ProcessingState', doc_dir: Path):
        """Saves the current processing state."""
        try:
            state_path = doc_dir / "processing_state.json"
            await self._write_json(state_path, state.model_dump())
        except Exception as e:
            logger.error(f"Error saving processing state for {state.doc_id}: {e}")
            raise ProcessingError(f"Error saving processing state for {state.doc_id}: {e}") from e
    
    @staticmethod
    def calculate_md5(content: str) -> str:
        """Calculates MD5 hash of the given content."""
        return hashlib.md5(content.encode()).hexdigest()
    
    async def chunk_document_async(self, text: str, doc_id: str) -> List[Tuple[str, ChunkInfo]]:
        """Asynchronously chunks a document."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            self.process_pool, 
            chunk_text_synchronously, 
            text, doc_id, self.token_limit, self.token_counter
        )
    
    async def process_file(
        self,
        input_path: Path,
        output_dir: Path,
        progress: 'ProcessingProgress',
        previous_version_id: Optional[str] = None,
        changes_description: Optional[str] = None
    ) -> DocumentInfo:
        """Processes a single file."""
        doc_id = str(uuid.uuid4())
        doc_dir = output_dir / doc_id
        doc_dir.mkdir(parents=True, exist_ok=True)

        source_dir = doc_dir / "source"
        source_dir.mkdir(exist_ok=True)
        try:
            shutil.copy2(input_path, source_dir / input_path.name)
        except Exception as e:
            logger.error(f"Error copying file {input_path} to {source_dir}: {e}")
            raise ProcessingError(f"Error copying file {input_path} to {source_dir}: {e}") from e

        state = ProcessingState(
            doc_id=doc_id,
            current_chunk=0,
            processed_chunks=[],
            is_complete=False,
            error_message=None
        )

        metadata = None

        try:
            async with aiofiles.open(input_path, 'r', encoding='utf-8') as f:
                content = await f.read()
            
            if not content.strip():
                logger.warning(f"File {input_path} is empty. Skipping.")
                raise ProcessingError(f"File {input_path} is empty.")

            chunks = await self.chunk_document_async(content, doc_id)

            if not chunks:
                logger.warning(f"No chunks created for file {input_path}. Skipping.")
                raise ProcessingError(f"No chunks created for file {input_path}.")

            total_chunks = len(chunks)
            total_tokens = sum(chunk_info.tokens for _, chunk_info in chunks)

            # Write all chunks
            for chunk_text, chunk_info in chunks:
                await self._write_chunk(chunk_text, chunk_info, doc_dir)
                state.current_chunk = chunk_info.number
                state.processed_chunks.append(chunk_info.id)

            # Update progress
            await self.progress_tracker.increment_processed_files()
            await self.progress_tracker.increment_chunks_tokens(total_chunks, total_tokens)

            state.is_complete = True
            await self._save_processing_state(state, doc_dir)

            # Create DocumentInfo
            doc_info = DocumentInfo(
                id=doc_id,
                filename=input_path.name,
                original_path=str(source_dir / input_path.name),
                total_chunks=total_chunks,
                total_tokens=total_tokens,
                total_chars=len(content),
                total_lines=content.count('\n') + 1,
                model_name=self.model_name,
                token_limit=self.token_limit,
                md5_hash=self.calculate_md5(content),
                file_size=len(content.encode('utf-8')),
                chunks=[info for _, info in chunks]
            )

            # Update Manifest
            manifest = await self.metadata_manager.create_or_update_manifest(
                doc_info,
                output_dir,
                previous_version_id,
                changes_description
            )
            
            doc_info.version_id = manifest.version_history[-1].version_id
            doc_info.manifest_id = manifest.manifest_id
            
            # Track Processing
            metadata = await self.metadata_manager.track_processing(manifest, doc_info, output_dir)
            
            # Write Document Info
            await self._write_json(doc_dir / "document_info.json", doc_info.model_dump())

            # Update Processing Status
            await self.metadata_manager.update_processing_status(metadata, output_dir, "completed")

            return doc_info

        except ProcessingError as pe:
            logger.error(f"Processing failed for {input_path}: {pe}")
            state.error_message = str(pe)
            await self._cleanup_incomplete_processing(doc_dir, state)
            if metadata:
                await self.metadata_manager.update_processing_status(metadata, output_dir, "failed", str(pe))
            raise pe
        except Exception as e:
            logger.error(f"Unexpected error processing file {input_path}: {e}")
            state.error_message = str(e)
            await self._cleanup_incomplete_processing(doc_dir, state)
            if metadata:
                await self.metadata_manager.update_processing_status(metadata, output_dir, "failed", str(e))
            raise ProcessingError(f"Unexpected error processing file {input_path}: {e}") from e

    async def _cleanup_incomplete_processing(self, doc_dir: Path, state: 'ProcessingState'):
        """Cleans up in case of incomplete processing."""
        state.is_complete = False
        await self._save_processing_state(state, doc_dir)
        logger.error(f"Processing incomplete for doc {state.doc_id}. State saved.")

    async def cleanup(self):
        """Cleans up resources."""
        self.process_pool.shutdown(wait=True)
