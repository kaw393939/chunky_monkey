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
from ..utils.config import AppConfig
from ..services.metadata_service import MetadataManager
from ..core.verifier import Verifier

import aiofiles
import tiktoken
import spacy

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# CUSTOM EXCEPTIONS
# -----------------------------------------------------------------------------

class ProcessingError(Exception):
    """Custom exception for processing errors."""
    pass

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
        processed_files: int, 
        current_file: str, 
        start_time: datetime, 
        processed_chunks: int, 
        total_tokens: int, 
        current_chunk: int
    ):
        self.total_files = total_files
        self.processed_files = processed_files
        self.current_file = current_file
        self.start_time = start_time
        self.processed_chunks = processed_chunks
        self.total_tokens = total_tokens
        self.current_chunk = current_chunk
        self.lock = asyncio.Lock()
    
    async def update(self, progress: 'ProcessingProgress'):
        """Updates the current progress."""
        async with self.lock:
            self.processed_files = progress.processed_files
            self.current_file = progress.current_file
            self.processed_chunks = progress.processed_chunks
            self.total_tokens = progress.total_tokens
            self.current_chunk = progress.current_chunk
    
    async def monitor(self):
        """Periodically logs the current processing progress."""
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
    chunk_number = 0
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
    ):
        model_name = config.model_name
        if model_name not in {
            'gpt-3.5', 'gpt-4', 'gpt-4-32k', 'claude', 'claude-2', 'gpt-4o'
        }:
            raise ValueError(f"Unsupported model. Choose from: gpt-3.5, gpt-4, gpt-4-32k, claude, claude-2, gpt-4o")
        
        self.model_name = model_name
        self.model_config = config.model_configs[model_name]
        base_limit = self.model_config['tokens']
        self.token_limit = int(base_limit * config.chunk_reduction_factor)

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
        async with aiofiles.open(path, 'w', encoding='utf-8') as f:
            json_str = json.dumps(data, indent=2, cls=DateTimeEncoder)
            await f.write(json_str)
    
    async def _write_chunk(self, chunk_text: str, chunk_info: ChunkInfo, doc_dir: Path):
        """Writes chunk text and metadata to files."""
        chunks_dir = doc_dir / "chunks"
        chunks_dir.mkdir(exist_ok=True)
        
        chunk_path = chunks_dir / f"{chunk_info.id}.txt"
        meta_path = chunks_dir / f"{chunk_info.id}.json"
        
        async with aiofiles.open(chunk_path, 'w', encoding='utf-8') as f:
            await f.write(chunk_text)
        
        await self._write_json(meta_path, chunk_info.model_dump())
    
    async def _save_processing_state(self, state: 'ProcessingState', doc_dir: Path):
        """Saves the current processing state."""
        state_path = doc_dir / "processing_state.json"
        await self._write_json(state_path, state.model_dump())
    
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
        shutil.copy2(input_path, source_dir / input_path.name)

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

            chunks = await self.chunk_document_async(content, doc_id)

            for chunk_text, chunk_info in chunks:
                progress.current_chunk = chunk_info.number
                progress.processed_chunks += 1
                progress.total_tokens += chunk_info.tokens
                # If you have a bytes_processed attribute, uncomment below
                # progress.bytes_processed += len(chunk_text.encode('utf-8'))
                await self.progress_tracker.update(progress)
                
                state.current_chunk = chunk_info.number
                state.processed_chunks.append(chunk_info.id)
                
                await self._write_chunk(chunk_text, chunk_info, doc_dir)
                await self._save_processing_state(state, doc_dir)

            state.is_complete = True
            await self._save_processing_state(state, doc_dir)

            doc_info = DocumentInfo(
                id=doc_id,
                filename=input_path.name,
                original_path=str(source_dir / input_path.name),
                total_chunks=len(chunks),
                total_tokens=sum(info.tokens for _, info in chunks),
                total_chars=len(content),
                total_lines=content.count('\n') + 1,
                model_name=self.model_name,
                token_limit=self.token_limit,
                md5_hash=self.calculate_md5(content),
                file_size=len(content.encode('utf-8')),
                chunks=[info for _, info in chunks]
            )

            manifest = await self.metadata_manager.create_or_update_manifest(
                doc_info,
                output_dir,
                previous_version_id,
                changes_description
            )
            
            doc_info.version_id = manifest.version_history[-1].version_id
            doc_info.manifest_id = manifest.manifest_id
            
            metadata = await self.metadata_manager.track_processing(manifest, doc_info, output_dir)
            
            await self._write_json(doc_dir / "document_info.json", doc_info.model_dump())
            
            await self.metadata_manager.update_processing_status(metadata, output_dir, "completed")
            
            return doc_info

        except Exception as e:
            logger.error(f"Error processing file {input_path}: {e}")
            state.error_message = str(e)
            await self._cleanup_incomplete_processing(doc_dir, state)
            if metadata:
                await self.metadata_manager.update_processing_status(metadata, output_dir, "failed", str(e))
            raise ProcessingError(f"Failed to process file {input_path}: {e}") from e

    async def _cleanup_incomplete_processing(self, doc_dir: Path, state: 'ProcessingState'):
        """Cleans up in case of incomplete processing."""
        state.is_complete = False
        await self._save_processing_state(state, doc_dir)
        logger.error(f"Processing incomplete for doc {state.doc_id}. State saved.")

    async def process_directory(self, input_dir: Path, output_dir: Path) -> List[DocumentInfo]:
        """Processes all files in a directory asynchronously."""
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        files = list(input_path.glob("**/*.txt"))
        total_files = len(files)
        
        self.progress_tracker.total_files = total_files
        
        # Start monitoring progress
        monitor_task = asyncio.create_task(self.progress_tracker.monitor())
        results: List[DocumentInfo] = []

        sem = Semaphore(self.max_concurrent_files)

        async def worker(file_path: Path, idx: int):
            async with sem:
                progress = ProcessingProgress(
                    total_files=total_files,
                    processed_files=idx,
                    current_file=file_path.name,
                    start_time=datetime.now(),
                    processed_chunks=0,
                    total_tokens=0,
                    current_chunk=0
                )
                try:
                    doc_info = await self.process_file(file_path, output_path, progress)
                    results.append(doc_info)
                except ProcessingError as err:
                    logger.error(f"Failed to process {file_path}: {err}")

        await asyncio.gather(*(worker(file_path, i) for i, file_path in enumerate(files, 1)))

        monitor_task.cancel()
        try:
            await monitor_task
        except asyncio.CancelledError:
            pass
        
        return results

    async def update_document(
        self,
        doc_id: str,
        content: str,
        verify: bool = False,
        verification_mode: str = 'strict'
    ) -> Optional[DocumentInfo]:
        """Updates an existing document with new content."""
        # Locate the document directory
        output_dir = Path(self.metadata_manager.config.output_dir)
        doc_dir = output_dir / doc_id
        if not doc_dir.exists():
            logger.error(f"Document directory does not exist: {doc_dir}")
            return None

        # Read existing document info
        doc_info_path = doc_dir / "document_info.json"
        if not doc_info_path.exists():
            logger.error(f"Document info not found: {doc_info_path}")
            return None

        async with aiofiles.open(doc_info_path, 'r', encoding='utf-8') as f:
            doc_info_data = json.loads(await f.read())
            doc_info = DocumentInfo(**doc_info_data)

        # Update content
        chunks = await self.chunk_document_async(content, doc_id)

        # Clear existing chunks
        chunks_dir = doc_dir / "chunks"
        if chunks_dir.exists():
            shutil.rmtree(chunks_dir)
        chunks_dir.mkdir(exist_ok=True)

        # Write new chunks
        for chunk_text, chunk_info in chunks:
            await self._write_chunk(chunk_text, chunk_info, doc_dir)

        # Update document info
        doc_info.total_chunks = len(chunks)
        doc_info.total_tokens = sum(info.tokens for _, info in chunks)
        doc_info.total_chars = len(content)
        doc_info.total_lines = content.count('\n') + 1
        doc_info.md5_hash = self.calculate_md5(content)
        doc_info.file_size = len(content.encode('utf-8'))
        doc_info.chunks = [info for _, info in chunks]
        doc_info.processed_at = datetime.now().isoformat()

        # Update manifest
        manifest = await self.metadata_manager.create_or_update_manifest(
            doc_info,
            output_dir,
            previous_version_id=doc_info.version_id,
            changes_description="Document content updated."
        )
        doc_info.version_id = manifest.version_history[-1].version_id
        doc_info.manifest_id = manifest.manifest_id

        # Track processing
        metadata = await self.metadata_manager.track_processing(manifest, doc_info, output_dir)

        # Write updated document info
        await self._write_json(doc_info_path, doc_info.model_dump())

        # Update processing status
        await self.metadata_manager.update_processing_status(metadata, output_dir, "completed")

        # Optionally perform verification
        if verify:
            logger.info("Starting verification after update...")
            is_valid = await self.verifier.verify_document(
                doc_info=doc_info,
                doc_dir=doc_dir,
                mode=verification_mode
            )
            if is_valid:
                logger.info(f"Verification passed for document {doc_id}.")
            else:
                logger.warning(f"Verification failed for document {doc_id}.")

        logger.info(f"Successfully updated document: {doc_info.filename}")
        return doc_info

    async def delete_document(self, doc_id: str) -> bool:
        """Deletes a document and its associated data."""
        output_dir = Path(self.metadata_manager.config.output_dir)
        doc_dir = output_dir / doc_id
        if not doc_dir.exists():
            logger.error(f"Document directory does not exist: {doc_dir}")
            return False

        try:
            shutil.rmtree(doc_dir)
            logger.info(f"Successfully deleted document directory: {doc_dir}")
            return True
        except Exception as e:
            logger.error(f"Error deleting document {doc_id}: {e}")
            return False

    async def cleanup(self):
        """Cleans up resources."""
        self.process_pool.shutdown(wait=True)
