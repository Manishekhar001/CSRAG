import io
from unittest.mock import MagicMock, patch

import pytest


def test_upload_txt_returns_200(client, sample_text_bytes):
    response = client.post(
        "/documents/upload",
        files={"file": ("test.txt", io.BytesIO(sample_text_bytes), "text/plain")},
    )
    assert response.status_code == 200


def test_upload_txt_returns_correct_filename(client, sample_text_bytes):
    response = client.post(
        "/documents/upload",
        files={"file": ("test.txt", io.BytesIO(sample_text_bytes), "text/plain")},
    )
    data = response.json()
    assert data["filename"] == "test.txt"


def test_upload_txt_returns_document_ids(client, sample_text_bytes):
    response = client.post(
        "/documents/upload",
        files={"file": ("test.txt", io.BytesIO(sample_text_bytes), "text/plain")},
    )
    data = response.json()
    assert "document_ids" in data
    assert isinstance(data["document_ids"], list)


def test_upload_txt_returns_chunks_created(client, sample_text_bytes):
    response = client.post(
        "/documents/upload",
        files={"file": ("test.txt", io.BytesIO(sample_text_bytes), "text/plain")},
    )
    data = response.json()
    assert "chunks_created" in data
    assert isinstance(data["chunks_created"], int)


def test_upload_csv_returns_200(client, sample_csv_bytes):
    response = client.post(
        "/documents/upload",
        files={"file": ("data.csv", io.BytesIO(sample_csv_bytes), "text/csv")},
    )
    assert response.status_code == 200


def test_upload_unsupported_format_returns_400(client):
    response = client.post(
        "/documents/upload",
        files={"file": ("file.docx", io.BytesIO(b"fake content"), "application/octet-stream")},
    )
    assert response.status_code == 400


def test_upload_missing_filename_returns_400(client):
    response = client.post(
        "/documents/upload",
        files={"file": ("", io.BytesIO(b"content"), "text/plain")},
    )
    assert response.status_code == 400


def test_upload_success_message(client, sample_text_bytes):
    response = client.post(
        "/documents/upload",
        files={"file": ("test.txt", io.BytesIO(sample_text_bytes), "text/plain")},
    )
    data = response.json()
    assert "message" in data
    assert "successfully" in data["message"].lower()


def test_collection_info_returns_200(client):
    response = client.get("/documents/info")
    assert response.status_code == 200


def test_collection_info_returns_collection_name(client):
    response = client.get("/documents/info")
    data = response.json()
    assert data["collection_name"] == "csrag_documents"


def test_collection_info_returns_total_documents(client):
    response = client.get("/documents/info")
    data = response.json()
    assert "total_documents" in data
    assert isinstance(data["total_documents"], int)


def test_collection_info_returns_status(client):
    response = client.get("/documents/info")
    data = response.json()
    assert "status" in data


def test_delete_collection_returns_200(client):
    response = client.delete("/documents/collection")
    assert response.status_code == 200


def test_delete_collection_returns_success_message(client):
    response = client.delete("/documents/collection")
    data = response.json()
    assert "message" in data


def test_upload_calls_vector_store_add_documents(client, mock_vector_store, sample_text_bytes):
    client.post(
        "/documents/upload",
        files={"file": ("test.txt", io.BytesIO(sample_text_bytes), "text/plain")},
    )
    mock_vector_store.add_documents.assert_called_once()


def test_delete_collection_calls_vector_store(client, mock_vector_store):
    client.delete("/documents/collection")
    mock_vector_store.delete_collection.assert_called_once()
