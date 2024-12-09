"""Command-line interface for the document processor."""

import asyncio
import sys
from pathlib import Path
from typing import Optional, List
import argparse
import logging

from ..core.processor import DocumentProcessor, ProcessingError
from ..utils.config import Config
from ..utils import get_logger

logger = get_logger(__name__)

class CommandHandler:
    """Handles CLI command execution."""
    
    def __init__(self):
        self.processor: Optional[DocumentProcessor] = None
        
    async def initialize(self, args: argparse.Namespace) -> None:
        """Initialize processor with configuration."""
        config = Config(
            model_name=args.model,
            output_dir=Path(args.output),
            max_concurrent_files=args.max_concurrent_files,
            chunk_reduction_factor=getattr(args, 'chunk_reduction_factor', 1.0)
        )
        self.processor = DocumentProcessor(config)

    async def import_documents(self, args: argparse.Namespace) -> int:
        """Import documents from input directory."""
        try:
            input_path = Path(args.input)
            if not input_path.exists():
                logger.error(f"Input path does not exist: {input_path}")
                return 1

            if input_path.is_file():
                doc_info = await self.processor.process_document(
                    input_path=input_path,
                    verify=args.verify,
                    verification_mode=args.verify_mode
                )
                logger.info(f"Successfully processed document: {doc_info.filename}")
                return 0
            
            elif input_path.is_dir():
                results = await self.processor.process_directory(
                    input_dir=input_path,
                    verify=args.verify,
                    verification_mode=args.verify_mode
                )
                logger.info(f"Successfully processed {len(results)} documents")
                return 0
            
            else:
                logger.error(f"Invalid input path: {input_path}")
                return 1

        except ProcessingError as e:
            logger.error(f"Processing failed: {e}")
            return 1
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            return 1

    async def retrieve_document(self, args: argparse.Namespace) -> int:
        """Retrieve a document by ID."""
        try:
            doc_info = await self.processor.get_document(args.doc_id)
            if not doc_info:
                logger.error(f"Document not found: {args.doc_id}")
                return 1

            # Save document content if requested
            if args.output_file:
                output_path = Path(args.output_file)
                content, _ = await self.processor.doc_service.read_document(
                    args.doc_id,
                    include_content=True
                )
                if content:
                    output_path.write_text(content)
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
            chunk_result = await self.processor.get_chunk(args.doc_id, args.chunk_id)
            if not chunk_result:
                logger.error(f"Chunk not found: {args.chunk_id}")
                return 1

            content, metadata = chunk_result
            
            # Save chunk content if requested
            if args.output_file:
                output_path = Path(args.output_file)
                output_path.write_text(content)
                logger.info(f"Chunk content saved to: {output_path}")

            # Print chunk metadata
            print(f"Chunk ID: {metadata.id}")
            print(f"Document ID: {metadata.doc_id}")
            print(f"Chunk number: {metadata.number}")
            print(f"Tokens: {metadata.tokens}")
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

            content = input_path.read_text()
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
    
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Set the logging level"
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Import command
    import_parser = subparsers.add_parser("import", help="Import documents")
    import_parser.add_argument("--input", required=True, help="Input file or directory")
    import_parser.add_argument("--output", required=True, help="Output directory")
    import_parser.add_argument("--model", default="gpt-4", help="Model name")
    import_parser.add_argument("--verify", action="store_true", help="Verify after import")
    import_parser.add_argument(
        "--verify-mode",
        choices=['strict', 'lenient', 'token'],
        default='strict',
        help="Verification mode"
    )
    import_parser.add_argument(
        "--max-concurrent-files",
        type=int,
        default=10,
        help="Maximum concurrent files to process"
    )
    import_parser.add_argument(
        "--chunk-reduction-factor",
        type=float,
        default=1.0,
        help="Factor to reduce chunk size"
    )

    # Retrieve document command
    retrieve_parser = subparsers.add_parser("get", help="Retrieve document")
    retrieve_parser.add_argument("--doc-id", required=True, help="Document ID")
    retrieve_parser.add_argument("--output", required=True, help="Output directory")
    retrieve_parser.add_argument("--output-file", help="Save content to file")

    # Retrieve chunk command
    chunk_parser = subparsers.add_parser("get-chunk", help="Retrieve chunk")
    chunk_parser.add_argument("--doc-id", required=True, help="Document ID")
    chunk_parser.add_argument("--chunk-id", required=True, help="Chunk ID")
    chunk_parser.add_argument("--output", required=True, help="Output directory")
    chunk_parser.add_argument("--output-file", help="Save content to file")

    # Update command
    update_parser = subparsers.add_parser("update", help="Update document")
    update_parser.add_argument("--doc-id", required=True, help="Document ID")
    update_parser.add_argument("--input-file", required=True, help="Input file")
    update_parser.add_argument("--output", required=True, help="Output directory")
    update_parser.add_argument("--verify", action="store_true", help="Verify after update")
    update_parser.add_argument(
        "--verify-mode",
        choices=['strict', 'lenient', 'token'],
        default='strict',
        help="Verification mode"
    )

    # Delete command
    delete_parser = subparsers.add_parser("delete", help="Delete document")
    delete_parser.add_argument("--doc-id", required=True, help="Document ID")
    delete_parser.add_argument("--output", required=True, help="Output directory")

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
        handler = CommandHandler()
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