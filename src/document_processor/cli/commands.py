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
from document_processor.core.models import DocumentInfo
from document_processor.utils.config import AppConfig, MODEL_CONFIGS
from document_processor.utils.logging import get_logger
import aiofiles  # Ensure aiofiles is imported for asynchronous file operations

logger = get_logger(__name__)

class CommandHandler:
    """Handles CLI command execution."""

    def __init__(self, config: AppConfig, chunk_reduction_factor: float):
        self.processor: Optional[TextProcessor] = None
        self.config = config
        self.chunk_reduction_factor = chunk_reduction_factor  # New attribute

    async def initialize(self, args: argparse.Namespace) -> None:
        """Initialize all services with configuration."""
        self.processor = TextProcessor(self.config, chunk_reduction_factor=self.chunk_reduction_factor)

    async def import_documents(self, args: argparse.Namespace) -> int:
        """Import documents from input directory or file."""
        try:
            input_path = Path(args.input_dir)
            output_dir = Path(args.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

            # Gather all .txt files in the input path
            if input_path.is_file() and input_path.suffix == '.txt':
                files = [input_path]
            elif input_path.is_dir():
                files = list(input_path.glob("**/*.txt"))
            else:
                logger.warning(f"Input path is neither a directory nor a .txt file: {input_path}")
                return 1

            total_files = len(files)
            if total_files == 0:
                logger.warning(f"No .txt files found in the input path: {input_path}")
                return 1

            logger.info(f"Starting import of {total_files} file(s) from {input_path} to {output_dir}.")

            # Initialize Progress Tracker
            progress_tracker = self.processor.progress_tracker
            progress_tracker.total_files = total_files
            progress_tracker.current_file = "Initializing..."
            progress_tracker.start_time = datetime.now()

            # Start monitoring progress
            monitor_task = asyncio.create_task(progress_tracker.monitor())

            results: List[DocumentInfo] = []

            sem = asyncio.Semaphore(self.config.max_concurrent_files)

            async def worker(file_path: Path, idx: int):
                async with sem:
                    progress_tracker.processed_files = idx
                    progress_tracker.current_file = file_path.name
                    try:
                        doc_info = await self.processor.process_file(file_path, output_dir, progress_tracker)
                        results.append(doc_info)
                        logger.info(f"Successfully processed file: {file_path.name}")
                    except ProcessingError as err:
                        logger.error(f"Failed to process {file_path}: {err}")

            await asyncio.gather(*(worker(file_path, i) for i, file_path in enumerate(files, 1)))

            monitor_task.cancel()
            try:
                await monitor_task
            except asyncio.CancelledError:
                pass

            logger.info(f"Successfully processed {len(results)} out of {total_files} document(s).")

            # Optionally perform verification
            if args.verify:
                logger.info("Starting verification...")
                verification_results = await self.processor.verifier.verify_all_documents(
                    output_dir=output_dir,
                    mode=args.verify_mode
                )
                valid = sum(verification_results.values())
                invalid = len(verification_results) - valid
                logger.info(f"Verification completed: {valid} valid, {invalid} invalid.")

            return 0

        except Exception as e:
            logger.error(f"Unexpected error during import: {e}")
            return 1

def create_parser() -> argparse.ArgumentParser:
    """Create the command-line argument parser."""
    # Parent parser for global arguments
    parent_parser = argparse.ArgumentParser(add_help=False)
    parent_parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Set the logging level"
    )
    parent_parser.add_argument(
        "--model-name",
        choices=MODEL_CONFIGS.keys(),
        default="gpt-4",
        help="Model name"
    )
    parent_parser.add_argument(
        "--spacy-model",
        default="en_core_web_sm",
        help="SpaCy model name for NLP tasks"
    )
    parent_parser.add_argument(
        "--max-concurrent-files",
        type=int,
        default=10,
        help="Maximum concurrent files to process"
    )
    parent_parser.add_argument(
        "--output-dir",
        type=str,
        default="output",
        help="Directory to store processed documents"
    )
    parent_parser.add_argument(
        "--chunk-reduction-factor",
        type=float,
        default=1.0,  # Changed default to 1.0
        help="Factor to reduce chunk size (default: 1.0, chunk to max tokens)"
    )
    # Removed --chunk-dir argument

    # Main parser
    parser = argparse.ArgumentParser(
        description="Document Processing CLI",
        parents=[parent_parser],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands", required=True)
    
    # Import command
    import_parser = subparsers.add_parser("import", help="Import documents")
    import_parser.add_argument("--input-dir", required=True, help="Input directory containing documents")
    import_parser.add_argument("--verify", action="store_true", help="Verify after import")
    import_parser.add_argument(
        "--verify-mode",
        choices=['strict', 'lenient', 'token'],
        default='strict',
        help="Verification mode"
    )
    # No --chunk-reduction-factor here; it's a global argument
    
    return parser

async def main() -> int:
    """Main entry point for the CLI."""
    parser = create_parser()
    args = parser.parse_args()

    # Validate chunk_reduction_factor
    if args.chunk_reduction_factor <= 0 or args.chunk_reduction_factor > 1.0:
        logger.error("--chunk-reduction-factor must be greater than 0 and at most 1.0")
        return 1

    # Configure logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    try:
        # Initialize Config with CLI arguments
        config = AppConfig(
            model_name=args.model_name,
            spacy_model=args.spacy_model,
            max_concurrent_files=args.max_concurrent_files,
            output_dir=args.output_dir
            # Removed chunk_reduction_factor from config
        )
        
        handler = CommandHandler(config, chunk_reduction_factor=args.chunk_reduction_factor)
        await handler.initialize(args)

        if args.command == "import":
            return await handler.import_documents(args)

        # No other commands to handle

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
