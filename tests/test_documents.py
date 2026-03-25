"""Tests for document processing and upload endpoints."""

import io
from unittest.mock import MagicMock, patch

import pytest


class TestDocumentProcessor:
    """Test DocumentProcessor class functionality."""

    def test_supported_extensions(self):
        """Test that supported extensions are defined."""
        from app.core.document_processor import DocumentProcessor

        processor = DocumentProcessor()
        assert ".pdf" in processor.SUPPORTED_EXTENSIONS
        assert ".txt" in processor.SUPPORTED_EXTENSIONS
        assert ".csv" in processor.SUPPORTED_EXTENSIONS

    def test_processor_initialization(self, mock_settings):
        """Test processor initialization with settings."""
        from app.core.document_processor import DocumentProcessor

        processor = DocumentProcessor(chunk_size=500, chunk_overlap=100)
        assert processor.chunk_size == 500
        assert processor.chunk_overlap == 100

    def test_processor_initialises_with_defaults(self):
        """Test processor uses settings defaults when no args given."""
        from app.core.document_processor import DocumentProcessor

        processor = DocumentProcessor()
        assert processor.chunk_size == 900
        assert processor.chunk_overlap == 150

    def test_load_from_upload_txt_returns_documents(self):
        """Test loading a TXT file returns a list of Documents."""
        from app.core.document_processor import DocumentProcessor

        processor = DocumentProcessor()
        content = b"This is a test document with enough content to process."
        docs = processor.load_from_upload(io.BytesIO(content), "test.txt")
        assert isinstance(docs, list)
        assert len(docs) >= 1

    def test_load_from_upload_sets_source_metadata(self):
        """Test that uploaded files have the correct source metadata."""
        from app.core.document_processor import DocumentProcessor

        processor = DocumentProcessor()
        content = b"Sample document content for metadata testing."
        docs = processor.load_from_upload(io.BytesIO(content), "my_file.txt")
        for doc in docs:
            assert doc.metadata.get("source") == "my_file.txt"

    def test_load_from_upload_csv_returns_documents(self):
        """Test loading a CSV file returns Documents."""
        from app.core.document_processor import DocumentProcessor

        processor = DocumentProcessor()
        content = b"name,age\nAlice,30\nBob,25\n"
        docs = processor.load_from_upload(io.BytesIO(content), "data.csv")
        assert isinstance(docs, list)
        assert len(docs) >= 1

    def test_load_from_upload_unsupported_extension_raises_value_error(self):
        """Test that unsupported file types raise ValueError."""
        from app.core.document_processor import DocumentProcessor

        processor = DocumentProcessor()
        with pytest.raises(ValueError, match="Unsupported extension"):
            processor.load_from_upload(io.BytesIO(b"content"), "file.docx")

    def test_load_file_unsupported_raises_value_error(self):
        """Test that load_file raises ValueError for unsupported extension."""
        from app.core.document_processor import DocumentProcessor

        processor = DocumentProcessor()
        with pytest.raises(ValueError, match="Unsupported extension"):
            processor.load_file("file.xyz")

    def test_process_upload_returns_chunks(self):
        """Test that process_upload returns a list of document chunks."""
        from app.core.document_processor import DocumentProcessor

        processor = DocumentProcessor()
        content = b"This is a sample document. " * 20
        chunks = processor.process_upload(io.BytesIO(content), "test.txt")
        assert isinstance(chunks, list)
        assert len(chunks) >= 1

    def test_split_documents(self, sample_chunks):
        """Test document splitting returns a list of chunks."""
        from app.core.document_processor import DocumentProcessor

        processor = DocumentProcessor()
        result = processor.split_documents(sample_chunks)
        assert isinstance(result, list)

    def test_split_documents_respects_chunk_size(self):
        """Test that split documents honours the configured chunk size."""
        from app.core.document_processor import DocumentProcessor
        from langchain_core.documents import Document

        processor = DocumentProcessor(chunk_size=500, chunk_overlap=50)
        long_text = "This is a sentence. " * 100
        docs = [Document(page_content=long_text, metadata={"source": "test"})]
        chunks = processor.split_documents(docs)
        assert len(chunks) >= 1
        for chunk in chunks:
            assert len(chunk.page_content) <= processor.chunk_size + processor.chunk_overlap

    def test_temp_file_is_cleaned_up(self):
        """Test that the temporary file is deleted after processing."""
        import tempfile
        from pathlib import Path
        from app.core.document_processor import DocumentProcessor

        processor = DocumentProcessor()
        content = b"Temporary file cleanup test."
        tmp_paths = []
        original_named_temp = tempfile.NamedTemporaryFile

        def capture_tmp(**kwargs):
            tmp = original_named_temp(**kwargs)
            tmp_paths.append(tmp.name)
            return tmp

        with patch("tempfile.NamedTemporaryFile", side_effect=capture_tmp):
            processor.load_from_upload(io.BytesIO(content), "test.txt")

        for path in tmp_paths:
            assert not Path(path).exists()


class TestDocumentEndpoints:
    """Test document API endpoints."""

    def test_get_collection_info(self, client, mock_vector_store):
        """Test getting collection information."""
        response = client.get("/documents/info")

        assert response.status_code == 200
        data = response.json()
        assert "collection_name" in data
        assert "total_documents" in data
        assert "status" in data

    def test_get_collection_info_name(self, client):
        """Test collection info returns the correct collection name."""
        response = client.get("/documents/info")

        data = response.json()
        assert data["collection_name"] == "csrag_documents"

    def test_upload_valid_text_file(self, client, mock_vector_store, sample_text_bytes):
        """Test uploading a valid text file."""
        files = {"file": ("test.txt", io.BytesIO(sample_text_bytes), "text/plain")}

        response = client.post("/documents/upload", files=files)

        assert response.status_code == 200
        data = response.json()
        assert data["filename"] == "test.txt"
        assert "chunks_created" in data
        assert "document_ids" in data

    def test_upload_valid_csv_file(self, client, sample_csv_bytes):
        """Test uploading a valid CSV file succeeds."""
        files = {"file": ("data.csv", io.BytesIO(sample_csv_bytes), "text/csv")}

        response = client.post("/documents/upload", files=files)

        assert response.status_code == 200

    def test_upload_invalid_file_type(self, client):
        """Test uploading unsupported file type returns 400."""
        file_content = b"test content"
        files = {"file": ("test.xyz", io.BytesIO(file_content), "application/octet-stream")}

        with patch("app.api.routes.documents.DocumentProcessor") as mock_processor:
            mock_instance = MagicMock()
            mock_instance.process_upload.side_effect = ValueError("Unsupported file extension")
            mock_processor.return_value = mock_instance

            response = client.post("/documents/upload", files=files)

            assert response.status_code == 400

    def test_upload_missing_filename_returns_422(self, client):
        """Test that an empty filename returns 422.

        FastAPI's UploadFile validation catches empty filenames at the
        framework level before the endpoint code runs, returning 422
        (Unprocessable Entity) instead of 400.
        """
        files = {"file": ("", io.BytesIO(b"content"), "text/plain")}
        response = client.post("/documents/upload", files=files)

        assert response.status_code == 422

    def test_upload_success_message(self, client, sample_text_bytes):
        """Test successful upload returns a success message."""
        files = {"file": ("test.txt", io.BytesIO(sample_text_bytes), "text/plain")}
        response = client.post("/documents/upload", files=files)

        data = response.json()
        assert "message" in data
        assert "successfully" in data["message"].lower()

    def test_upload_calls_add_documents(self, client, mock_vector_store, sample_text_bytes):
        """Test that upload triggers add_documents on the vector store."""
        files = {"file": ("test.txt", io.BytesIO(sample_text_bytes), "text/plain")}
        client.post("/documents/upload", files=files)

        mock_vector_store.add_documents.assert_called_once()

    def test_delete_collection(self, client, mock_vector_store):
        """Test deleting the collection."""
        response = client.delete("/documents/collection")

        assert response.status_code == 200
        data = response.json()
        assert "message" in data

    def test_delete_collection_calls_vector_store(self, client, mock_vector_store):
        """Test that delete triggers delete_collection on the vector store."""
        client.delete("/documents/collection")

        mock_vector_store.delete_collection.assert_called_once()
