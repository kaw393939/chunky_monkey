"""Token counting and text chunking components."""

import re
from abc import ABC, abstractmethod
from typing import List, Protocol, Tuple, Optional
import tiktoken
import spacy
from spacy.language import Language

from ..utils import get_logger
from ..utils.config import Config
from ..core.models import ChunkMetadata

logger = get_logger(__name__)

class TokenCounter(Protocol):
    """Protocol defining the interface for token counters."""
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in the given text."""
        ...

    def encode(self, text: str) -> List[int]:
        """Encode text to token IDs."""
        ...

    def decode(self, tokens: List[int]) -> str:
        """Decode token IDs back to text."""
        ...

class TiktokenCounter:
    """Token counter using tiktoken encodings."""
    
    def __init__(self, model_encoding: str):
        """
        Initialize with a specific model encoding.
        
        Args:
            model_encoding: Name of the tiktoken encoding to use
        """
        try:
            self.encoding = tiktoken.get_encoding(model_encoding)
            logger.info(f"Initialized tiktoken with encoding {model_encoding}")
        except Exception as e:
            logger.error(f"Failed to initialize tiktoken encoding {model_encoding}: {e}")
            raise

    def count_tokens(self, text: str) -> int:
        """Count tokens in text using tiktoken."""
        return len(self.encode(text))

    def encode(self, text: str) -> List[int]:
        """Encode text to token IDs."""
        return self.encoding.encode(text)

    def decode(self, tokens: List[int]) -> str:
        """Decode token IDs back to text."""
        return self.encoding.decode(tokens)

class SpacyTokenCounter:
    """Token counter using spaCy tokenization."""
    
    def __init__(self, model_name: str):
        """
        Initialize with a spaCy model.
        
        Args:
            model_name: Name of the spaCy model to use
        """
        try:
            self.nlp = spacy.load(model_name, disable=['ner', 'parser', 'attribute_ruler', 'lemmatizer'])
            if 'sentencizer' not in self.nlp.pipe_names:
                self.nlp.add_pipe('sentencizer')
            logger.info(f"Initialized spaCy with model {model_name}")
        except Exception as e:
            logger.error(f"Failed to initialize spaCy model {model_name}: {e}")
            raise

    def count_tokens(self, text: str) -> int:
        """Count tokens in text using spaCy."""
        return len(self.encode(text))

    def encode(self, text: str) -> List[int]:
        """Convert text to token IDs (using hash values as proxies)."""
        doc = self.nlp(text)
        return [hash(token.text) for token in doc]

    def decode(self, tokens: List[int]) -> str:
        """
        Decode is not properly supported in spaCy counter.
        This is a best-effort implementation.
        """
        logger.warning("SpacyTokenCounter decode() is not fully supported")
        return " ".join(str(token) for token in tokens)

class ChunkingStrategy(ABC):
    """Abstract base class for different chunking strategies."""
    
    @abstractmethod
    async def chunk_text(
        self,
        text: str,
        doc_id: str,
        token_counter: TokenCounter,
        max_tokens: int
    ) -> List[Tuple[str, ChunkMetadata]]:
        """
        Chunk text according to the strategy.
        
        Args:
            text: Text to chunk
            doc_id: Document ID
            token_counter: TokenCounter instance
            max_tokens: Maximum tokens per chunk
            
        Returns:
            List of (chunk_text, chunk_metadata) tuples
        """
        pass

class SentenceChunkingStrategy(ChunkingStrategy):
    """Chunk text by sentences while respecting token limits."""
    
    async def chunk_text(
        self,
        text: str,
        doc_id: str,
        token_counter: TokenCounter,
        max_tokens: int
    ) -> List[Tuple[str, ChunkMetadata]]:
        # Split into sentences
        sentences = self._split_into_sentences(text)
        chunks: List[Tuple[str, ChunkMetadata]] = []
        current_chunk: List[str] = []
        current_tokens = 0
        chunk_number = 0

        for sentence in sentences:
            sentence_tokens = token_counter.count_tokens(sentence)

            # Handle sentences that exceed max_tokens
            if sentence_tokens > max_tokens:
                # First, save current chunk if it exists
                if current_chunk:
                    chunk_text = ' '.join(current_chunk)
                    chunks.append(self._create_chunk(
                        chunk_text, doc_id, chunk_number, token_counter
                    ))
                    chunk_number += 1
                    current_chunk = []
                    current_tokens = 0

                # Split long sentence
                sentence_chunks = self._split_long_sentence(
                    sentence, max_tokens, token_counter
                )
                for sent_chunk in sentence_chunks:
                    chunks.append(self._create_chunk(
                        sent_chunk, doc_id, chunk_number, token_counter
                    ))
                    chunk_number += 1

            # Normal case: add sentence to current chunk or start new chunk
            elif current_tokens + sentence_tokens <= max_tokens:
                current_chunk.append(sentence)
                current_tokens += sentence_tokens
            else:
                # Save current chunk
                chunk_text = ' '.join(current_chunk)
                chunks.append(self._create_chunk(
                    chunk_text, doc_id, chunk_number, token_counter
                ))
                chunk_number += 1
                
                # Start new chunk with current sentence
                current_chunk = [sentence]
                current_tokens = sentence_tokens

        # Don't forget the last chunk
        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            chunks.append(self._create_chunk(
                chunk_text, doc_id, chunk_number, token_counter
            ))

        return chunks

    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences using regex."""
        # This is a simple sentence splitter; consider using spaCy for better accuracy
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]

    def _split_long_sentence(
        self,
        sentence: str,
        max_tokens: int,
        token_counter: TokenCounter
    ) -> List[str]:
        """Split a long sentence into smaller chunks."""
        words = sentence.split()
        chunks: List[str] = []
        current_chunk: List[str] = []
        current_tokens = 0

        for word in words:
            word_tokens = token_counter.count_tokens(word + ' ')
            if current_tokens + word_tokens > max_tokens:
                if current_chunk:
                    chunks.append(' '.join(current_chunk))
                    current_chunk = [word]
                    current_tokens = word_tokens
                else:
                    # Word itself exceeds max_tokens, split by character
                    chunks.extend(self._split_by_char(word, max_tokens, token_counter))
            else:
                current_chunk.append(word)
                current_tokens += word_tokens

        if current_chunk:
            chunks.append(' '.join(current_chunk))

        return chunks

    def _split_by_char(
        self,
        text: str,
        max_tokens: int,
        token_counter: TokenCounter
    ) -> List[str]:
        """Split text by characters when words are too long."""
        result: List[str] = []
        current_chunk = ""
        
        for char in text:
            if token_counter.count_tokens(current_chunk + char) > max_tokens:
                if current_chunk:
                    result.append(current_chunk)
                    current_chunk = char
                else:
                    # Single character exceeds max_tokens, log warning and skip
                    logger.warning(f"Character '{char}' exceeds token limit, skipping")
            else:
                current_chunk += char

        if current_chunk:
            result.append(current_chunk)

        return result

    def _create_chunk(
        self,
        text: str,
        doc_id: str,
        number: int,
        token_counter: TokenCounter
    ) -> Tuple[str, ChunkMetadata]:
        """Create a chunk with its metadata."""
        return text, ChunkMetadata(
            id=f"{doc_id}-chunk-{number}",
            number=number,
            tokens=token_counter.count_tokens(text),
            doc_id=doc_id,
            content_hash=str(hash(text)),  # Use proper hash in production
            character_count=len(text)
        )

class Chunker:
    """Main chunking coordinator."""
    
    def __init__(
        self,
        config: Config,
        strategy: Optional[ChunkingStrategy] = None
    ):
        """
        Initialize chunker with configuration.
        
        Args:
            config: Application configuration
            strategy: Optional chunking strategy (defaults to SentenceChunkingStrategy)
        """
        self.config = config
        self.strategy = strategy or SentenceChunkingStrategy()
        
        # Initialize token counter based on model
        model_config = config.model_config
        try:
            self.token_counter = TiktokenCounter(model_config['encoding'])
        except Exception as e:
            logger.warning(f"Falling back to spaCy: {e}")
            self.token_counter = SpacyTokenCounter(config.spacy_model)

    async def chunk_document(
        self,
        text: str,
        doc_id: str
    ) -> List[Tuple[str, ChunkMetadata]]:
        """
        Chunk a document's text.
        
        Args:
            text: Document text to chunk
            doc_id: Document identifier
            
        Returns:
            List of (chunk_text, chunk_metadata) tuples
        """
        return await self.strategy.chunk_text(
            text=text,
            doc_id=doc_id,
            token_counter=self.token_counter,
            max_tokens=self.config.token_limit
        )

    def count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        return self.token_counter.count_tokens(text)