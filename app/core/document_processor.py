"""Document processing module — identical to BasicRAG project.

Supports PDF, TXT, and CSV with RecursiveCharacterTextSplitter.
Chunk settings default to 900 / 150 (from the SRAG/CRAG notebooks).
"""

import tempfile
from pathlib import Path
from typing import BinaryIO

from langchain_community.document_loaders import CSVLoader, PyPDFLoader, TextLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import get_settings
from app.utils.logger import get_logger

logger = get_logger(__name__)


class DocumentProcessor:
    """Load and chunk documents for the RAG pipeline."""

    SUPPORTED_EXTENSIONS: set[str] = {".pdf", ".txt", ".csv"}

    def __init__(
        self,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
    ) -> None:
        """Initialise the document processor.

        Args:
            chunk_size: Token chunk size (defaults to settings.chunk_size).
            chunk_overlap: Chunk overlap (defaults to settings.chunk_overlap).
        """
        settings = get_settings()
        self.chunk_size = chunk_size or settings.chunk_size
        self.chunk_overlap = chunk_overlap or settings.chunk_overlap

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
            length_function=len,
        )
        logger.info(
            f"DocumentProcessor ready — "
            f"chunk_size={self.chunk_size}, chunk_overlap={self.chunk_overlap}"
        )

    # ------------------------------------------------------------------
    # Private loaders
    # ------------------------------------------------------------------

    def _load_pdf(self, file_path: Path) -> list[Document]:
        logger.info(f"Loading PDF: {file_path.name}")
        docs = PyPDFLoader(str(file_path)).load()
        logger.info(f"Loaded {len(docs)} pages from {file_path.name}")
        return docs

    def _load_text(self, file_path: Path) -> list[Document]:
        logger.info(f"Loading text: {file_path.name}")
        docs = TextLoader(str(file_path), encoding="utf-8").load()
        logger.info(f"Loaded {file_path.name}")
        return docs

    def _load_csv(self, file_path: Path) -> list[Document]:
        logger.info(f"Loading CSV: {file_path.name}")
        docs = CSVLoader(str(file_path), encoding="utf-8").load()
        logger.info(f"Loaded {len(docs)} rows from {file_path.name}")
        return docs

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load_file(self, file_path: str | Path) -> list[Document]:
        """Load a file by extension.

        Args:
            file_path: Path to the file.

        Returns:
            List of :class:`Document` objects.

        Raises:
            ValueError: If the extension is not supported.
        """
        file_path = Path(file_path)
        ext = file_path.suffix.lower()
        if ext not in self.SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Unsupported extension '{ext}'. "
                f"Supported: {self.SUPPORTED_EXTENSIONS}"
            )
        loaders = {
            ".pdf": self._load_pdf,
            ".txt": self._load_text,
            ".csv": self._load_csv,
        }
        return loaders[ext](file_path)

    def load_from_upload(self, file: BinaryIO, filename: str) -> list[Document]:
        """Load a document from an uploaded file-like object.

        Writes to a temp file, loads, then cleans up.

        Args:
            file: File-like binary object.
            filename: Original filename (used for extension detection).

        Returns:
            List of :class:`Document` objects.
        """
        ext = Path(filename).suffix.lower()
        if ext not in self.SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Unsupported extension '{ext}'. "
                f"Supported: {self.SUPPORTED_EXTENSIONS}"
            )
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
            tmp.write(file.read())
            tmp_path = tmp.name

        try:
            docs = self.load_file(tmp_path)
            for doc in docs:
                doc.metadata["source"] = filename
            return docs
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def split_documents(self, documents: list[Document]) -> list[Document]:
        """Split documents into chunks.

        Args:
            documents: Raw :class:`Document` objects.

        Returns:
            Chunked :class:`Document` objects.
        """
        logger.info(f"Splitting {len(documents)} documents into chunks")
        chunks = self.text_splitter.split_documents(documents)
        logger.info(f"Created {len(chunks)} chunks")
        return chunks

    def process_upload(self, file: BinaryIO, filename: str) -> list[Document]:
        """Load and split an uploaded file in one step.

        Args:
            file: File-like binary object.
            filename: Original filename.

        Returns:
            List of chunked :class:`Document` objects.
        """
        docs = self.load_from_upload(file, filename)
        return self.split_documents(docs)
