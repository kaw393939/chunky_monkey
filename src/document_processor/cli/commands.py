# File: src/document_processor/cli/commands.py

"""Command-line interface for the document processor."""

import asyncio
import sys
from pathlib import Path
from typing import Optional, List
import argparse
import logging
from datetime import datetime

from document_processor.core.processor import TextProcessor, ProcessingError
from document_processor.core.models import DocumentInfo, ProcessingProgress
from document_processor.core.verifier import Verifier
from document_processor.services.chunk_service import ChunkService
from document_processor.services.document_service import DocumentService
from document_processor.services.metadata_service import MetadataManager
from document_processor.utils.config import AppConfig
from document_processor.utils.logging import get_logger
import aiofiles  # Ensure aiofiles is imported for asynchronous file operations

logger = get_logger(__name__)

class CommandHandler:
    """Handles CLI command execution."""
    
    def __init__(self, config: AppConfig):
        self.processor: Optional[TextProcessor] = None
        self.verifier: Optional[Verifier] = None
        self.chunk_service: Optional[ChunkService] = None
        self.document_service: Optional[DocumentService] = None
        self.metadata_manager: Optional[MetadataManager] = None
        self.config = config  # Ensure self.config is assigned here
        
    async def initialize(self, args: argparse.Namespace) -> None:
        """Initialize all services with configuration."""
        self.processor = TextProcessor(self.config)
        self.verifier = Verifier(token_counter=self.processor.token_counter)
        self.chunk_service = ChunkService(Path(self.config.chunk_dir))
        self.metadata_manager = MetadataManager(model_name=self.config.model_name, config=self.config)
        self.document_service = DocumentService(Path(self.config.output_dir) / "document_info.json")
    
    async def import_documents(self, args: argparse.Namespace) -> int:
        """Import documents from input directory or file."""
        try:
            input_path = Path(args.input)
            output_dir = Path(args.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

            files = list(input_path.glob("**/*.txt"))
            total_files = len(files)
            self.processor.progress_tracker.total_files = total_files

            if total_files == 0:
                logger.warning(f"No .txt files found in the input path: {input_path}")
                return 1

            # Start monitoring progress
            monitor_task = asyncio.create_task(self.processor.progress_tracker.monitor())
            results: List[DocumentInfo] = []

            sem = asyncio.Semaphore(self.config.max_concurrent_files)

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
                        doc_info = await self.processor.process_file(file_path, output_dir, progress)
                        results.append(doc_info)
                    except ProcessingError as err:
                        logger.error(f"Failed to process {file_path}: {err}")

            await asyncio.gather(*(worker(file_path, i) for i, file_path in enumerate(files, 1)))

            monitor_task.cancel()
            try:
                await monitor_task
            except asyncio.CancelledError:
                pass

            logger.info(f"Successfully processed {len(results)} documents.")

            # Optionally perform verification
            if args.verify:
                logger.info("Starting verification...")
                verification_results = await self.verifier.verify_all_documents(
                    output_dir=output_dir,
                    mode=args.verify_mode
                )
                valid = sum(verification_results.values())
                invalid = len(verification_results) - valid
                logger.info(f"Verification completed: {valid} valid, {invalid} invalid.")

            return 0

        except ProcessingError as e:
            logger.error(f"Processing failed: {e}")
            return 1
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            return 1

    async def retrieve_document(self, args: argparse.Namespace) -> int:
        """Retrieve a document by ID."""
        try:
            doc_info = await self.document_service.read_document()
            if not doc_info or doc_info.id != args.doc_id:
                logger.error(f"Document not found: {args.doc_id}")
                return 1

            # Save document content if requested
            if args.output_file:
                output_path = Path(args.output_file)
                chunks_dir = Path(self.config.output_dir) / args.doc_id / "chunks"
                chunk_files = sorted(
                    chunks_dir.glob("*.txt"),
                    key=lambda p: int(p.stem.split('-')[-1])
                )
                reconstructed_text = ''
                for chunk_file in chunk_files:
                    async with aiofiles.open(chunk_file, 'r', encoding='utf-8') as f:
                        chunk_content = await f.read()
                        reconstructed_text += chunk_content
                async with aiofiles.open(output_path, 'w', encoding='utf-8') as f:
                    await f.write(reconstructed_text)
                logger.info(f"Document content saved to: {output_path}")

            # Print document metadata
            print(f"Document ID: {doc_info.id}")
            print(f"Filename: {doc_info.filename}")
            print(f"Total chunks: {doc_info.total_chunks}")
            print(f"Total tokens: {doc_info.total_tokens}")
            return 0

        except Exception as e:
            logger.error(f"Error retrieving document: {e}")
            return 1

    async def retrieve_chunk(self, args: argparse.Namespace) -> int:
        """Retrieve a specific chunk from a document."""
        try:
            chunk_result = await self.chunk_service.read_chunk(args.chunk_id)
            if not chunk_result:
                logger.error(f"Chunk not found: {args.chunk_id}")
                return 1

            # Save chunk content if requested
            if args.output_file:
                output_path = Path(args.output_file)
                chunk_text_path = Path(self.config.chunk_dir) / f"{args.chunk_id}.txt"
                if not chunk_text_path.exists():
                    logger.error(f"Chunk text file does not exist: {chunk_text_path}")
                    return 1
                async with aiofiles.open(chunk_text_path, 'r', encoding='utf-8') as f:
                    chunk_content = await f.read()
                async with aiofiles.open(output_path, 'w', encoding='utf-8') as f:
                    await f.write(chunk_content)
                logger.info(f"Chunk content saved to: {output_path}")

            # Print chunk metadata
            print(f"Chunk ID: {chunk_result['id']}")
            print(f"Document ID: {chunk_result['doc_id']}")
            print(f"Chunk number: {chunk_result['number']}")
            print(f"Tokens: {chunk_result['tokens']}")
            return 0

        except Exception as e:
            logger.error(f"Error retrieving chunk: {e}")
            return 1

    async def update_document(self, args: argparse.Namespace) -> int:
        """Update an existing document."""
        try:
            input_path = Path(args.input_file)
            if not input_path.exists():
                logger.error(f"Input file does not exist: {input_path}")
                return 1

            async with aiofiles.open(input_path, 'r', encoding='utf-8') as f:
                content = await f.read()
            doc_info = await self.processor.update_document(
                doc_id=args.doc_id,
                content=content,
                verify=args.verify,
                verification_mode=args.verify_mode
            )

            if not doc_info:
                logger.error(f"Failed to update document: {args.doc_id}")
                return 1

            logger.info(f"Successfully updated document: {doc_info.filename}")
            return 0

        except Exception as e:
            logger.error(f"Error updating document: {e}")
            return 1

    async def delete_document(self, args: argparse.Namespace) -> int:
        """Delete a document and its chunks."""
        try:
            success = await self.processor.delete_document(args.doc_id)
            if not success:
                logger.error(f"Failed to delete document: {args.doc_id}")
                return 1

            logger.info(f"Successfully deleted document: {args.doc_id}")
            return 0

        except Exception as e:
            logger.error(f"Error deleting document: {e}")
            return 1

def create_parser() -> argparse.ArgumentParser:
    """Create the command-line argument parser."""
    parser = argparse.ArgumentParser(
        description="Document Processing CLI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Global Arguments
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Set the logging level"
    )

    parser.add_argument(
        "--model-name",
        choices=[
            'gpt-3.5', 'gpt-4', 'gpt-4-32k', 'claude', 'claude-2', 'gpt-4o'
        ],
        default="gpt-4",
        help="Model name"
    )
    parser.add_argument(
        "--spacy-model",
        default="en_core_web_sm",
        help="SpaCy model name for NLP tasks"
    )
    parser.add_argument(
        "--max-concurrent-files",
        type=int,
        default=10,
        help="Maximum concurrent files to process"
    )
    parser.add_argument(
        "--chunk-reduction-factor",
        type=float,
        default=1.0,
        help="Factor to reduce chunk size"
    )
    parser.add_argument(
        "--output-dir",
        "--output",
        type=str,
        default="output",
        help="Directory to store processed documents"
    )
    parser.add_argument(
        "--chunk-dir",
        type=str,
        default="chunks",
        help="Directory to store document chunks"
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Import command
    import_parser = subparsers.add_parser("import", help="Import documents")
    import_parser.add_argument("--input", required=True, help="Input file or directory")
    import_parser.add_argument("--verify", action="store_true", help="Verify after import")
    import_parser.add_argument(
        "--verify-mode",
        choices=['strict', 'lenient', 'token'],
        default='strict',
        help="Verification mode"
    )
    # Adding aliases for --output-dir and --chunk-reduction-factor
    import_parser.add_argument(
        "--output-dir",
        "--output",
        type=str,
        default="output",
        help="Directory to store processed documents"
    )
    import_parser.add_argument(
        "--chunk-reduction-factor",
        type=float,
        default=1.0,
        help="Factor to reduce chunk size"
    )
    import_parser.add_argument(
        "--chunk-dir",
        type=str,
        default="chunks",
        help="Directory to store document chunks"
    )

    # Retrieve document command
    retrieve_parser = subparsers.add_parser("get", help="Retrieve document")
    retrieve_parser.add_argument("--doc-id", required=True, help="Document ID")
    retrieve_parser.add_argument("--output-file", help="Save content to file")

    # Retrieve chunk command
    chunk_parser = subparsers.add_parser("get-chunk", help="Retrieve chunk")
    chunk_parser.add_argument("--doc-id", required=True, help="Document ID")
    chunk_parser.add_argument("--chunk-id", required=True, help="Chunk ID")
    chunk_parser.add_argument("--output-file", help="Save content to file")

    # Update command
    update_parser = subparsers.add_parser("update", help="Update document")
    update_parser.add_argument("--doc-id", required=True, help="Document ID")
    update_parser.add_argument("--input-file", required=True, help="Input file")
    update_parser.add_argument("--verify", action="store_true", help="Verify after update")
    update_parser.add_argument(
        "--verify-mode",
        choices=['strict', 'lenient', 'token'],
        default='strict',
        help="Verification mode"
    )
    update_parser.add_argument(
        "--chunk-reduction-factor",
        type=float,
        default=1.0,
        help="Factor to reduce chunk size"
    )
    update_parser.add_argument(
        "--output-dir",
        "--output",
        type=str,
        default="output",
        help="Directory to store processed documents"
    )
    update_parser.add_argument(
        "--chunk-dir",
        type=str,
        default="chunks",
        help="Directory to store document chunks"
    )

    # Delete command
    delete_parser = subparsers.add_parser("delete", help="Delete document")
    delete_parser.add_argument("--doc-id", required=True, help="Document ID")

    return parser

async def main() -> int:
    """Main entry point for the CLI."""
    parser = create_parser()
    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    if not args.command:
        parser.print_help()
        return 1

    try:
        # Initialize Config with CLI arguments
        config = AppConfig(
            model_name=args.model_name,
            spacy_model=args.spacy_model,
            max_concurrent_files=args.max_concurrent_files,
            chunk_reduction_factor=args.chunk_reduction_factor,
            output_dir=args.output_dir,
            chunk_dir=args.chunk_dir
        )
        
        handler = CommandHandler(config)
        await handler.initialize(args)

        command_map = {
            "import": handler.import_documents,
            "get": handler.retrieve_document,
            "get-chunk": handler.retrieve_chunk,
            "update": handler.update_document,
            "delete": handler.delete_document,
        }

        return await command_map[args.command](args)

    except KeyboardInterrupt:
        logger.info("Operation cancelled by user")
        return 1
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return 1
    finally:
        if handler.processor:
            await handler.processor.cleanup()

def cli_main():
    """Entry point for console_scripts."""
    try:
        return asyncio.run(main())
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(cli_main())
