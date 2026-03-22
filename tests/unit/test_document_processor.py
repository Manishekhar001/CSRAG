import io
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from app.core.document_processor import DocumentProcessor


@pytest.fixture
def processor():
    return DocumentProcessor(chunk_size=500, chunk_overlap=50)


def test_processor_initialises_with_defaults():
    proc = DocumentProcessor()
    assert proc.chunk_size == 900
    assert proc.chunk_overlap == 150


def test_processor_accepts_custom_chunk_size():
    proc = DocumentProcessor(chunk_size=500, chunk_overlap=50)
    assert proc.chunk_size == 500
    assert proc.chunk_overlap == 50


def test_supported_extensions_contains_pdf(processor):
    assert ".pdf" in processor.SUPPORTED_EXTENSIONS


def test_supported_extensions_contains_txt(processor):
    assert ".txt" in processor.SUPPORTED_EXTENSIONS


def test_supported_extensions_contains_csv(processor):
    assert ".csv" in processor.SUPPORTED_EXTENSIONS


def test_load_from_upload_txt_returns_documents(processor):
    content = b"This is a test document with enough content to process."
    docs = processor.load_from_upload(io.BytesIO(content), "test.txt")
    assert isinstance(docs, list)
    assert len(docs) >= 1


def test_load_from_upload_sets_source_metadata(processor):
    content = b"Sample document content for metadata testing."
    docs = processor.load_from_upload(io.BytesIO(content), "my_file.txt")
    for doc in docs:
        assert doc.metadata.get("source") == "my_file.txt"


def test_load_from_upload_csv_returns_documents(processor):
    content = b"name,age\nAlice,30\nBob,25\n"
    docs = processor.load_from_upload(io.BytesIO(content), "data.csv")
    assert isinstance(docs, list)
    assert len(docs) >= 1


def test_load_from_upload_unsupported_extension_raises_value_error(processor):
    with pytest.raises(ValueError, match="Unsupported extension"):
        processor.load_from_upload(io.BytesIO(b"content"), "file.docx")


def test_process_upload_returns_chunks(processor):
    content = b"This is a sample document. " * 20
    chunks = processor.process_upload(io.BytesIO(content), "test.txt")
    assert isinstance(chunks, list)
    assert len(chunks) >= 1


def test_split_documents_returns_smaller_chunks(processor):
    from langchain_core.documents import Document
    long_text = "This is a sentence. " * 100
    docs = [Document(page_content=long_text, metadata={"source": "test"})]
    chunks = processor.split_documents(docs)
    assert len(chunks) >= 1
    for chunk in chunks:
        assert len(chunk.page_content) <= processor.chunk_size + processor.chunk_overlap


def test_load_file_unsupported_raises_value_error(processor):
    with pytest.raises(ValueError, match="Unsupported extension"):
        processor.load_file("file.xyz")


def test_temp_file_is_cleaned_up(processor):
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
