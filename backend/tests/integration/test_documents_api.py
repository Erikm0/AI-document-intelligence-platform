"""End-to-end document lifecycle through the HTTP API."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.container import Container
from tests.factories import SAMPLE_DOCX_PAGES, SAMPLE_TXT, make_docx, make_pdf

pytestmark = pytest.mark.integration

API = "/api/v1"


def upload(
    client: TestClient, name: str, data: bytes, content_type: str = "application/octet-stream"
):  # type: ignore[no-untyped-def]
    return client.post(f"{API}/documents", files={"file": (name, data, content_type)})


def test_health_and_system_info(client: TestClient) -> None:
    assert client.get(f"{API}/health").json()["status"] == "ok"
    ready = client.get(f"{API}/health/ready")
    assert ready.status_code == 200
    body = ready.json()
    assert body["components"]["database"]["status"] == "ok"
    assert body["components"]["llm"]["status"] == "ok"
    info = client.get(f"{API}/system/info").json()
    assert info["llm_model"] == "fake-extractive" and info["reranker"] == "lexical-coverage"
    assert ".pdf" in info["allowed_extensions"]


def test_request_id_header_is_propagated(client: TestClient) -> None:
    response = client.get(f"{API}/health", headers={"X-Request-ID": "abc123"})
    assert response.headers["X-Request-ID"] == "abc123"


def test_upload_txt_lifecycle(client: TestClient, container: Container) -> None:
    response = upload(client, "orion_report.txt", SAMPLE_TXT.encode(), "text/plain")
    assert response.status_code == 202, response.text
    doc = response.json()
    assert doc["filename"] == "orion_report.txt" and doc["file_type"] == "txt"

    # Ingestion runs synchronously in tests, so the document is ready now.
    detail = client.get(f"{API}/documents/{doc['id']}").json()
    assert detail["status"] == "ready", detail
    assert detail["page_count"] == 2
    assert detail["chunk_count"] >= 2
    assert detail["title"] == "Orion Space Agency Annual Report"
    assert detail["metadata"]["page_numbering"] == "form_feed"

    chunks = client.get(f"{API}/documents/{doc['id']}/chunks").json()
    assert chunks["total"] == detail["chunk_count"]
    shackleton = next(c for c in chunks["items"] if "Shackleton" in c["text"])
    assert shackleton["page_start"] == 2 and shackleton["section"] == "3. Results"

    assert container.vector_store.count() == detail["chunk_count"]
    assert len(container.lexical_index) == detail["chunk_count"]

    listing = client.get(f"{API}/documents").json()
    assert listing["total"] == 1 and listing["items"][0]["id"] == doc["id"]
    stats = client.get(f"{API}/documents/stats").json()
    assert stats["ready_documents"] == 1 and stats["total_chunks"] == detail["chunk_count"]

    download = client.get(f"{API}/documents/{doc['id']}/file")
    assert download.status_code == 200 and download.content == SAMPLE_TXT.encode()

    assert client.delete(f"{API}/documents/{doc['id']}").status_code == 204
    assert client.get(f"{API}/documents/{doc['id']}").status_code == 404
    assert container.vector_store.count() == 0 and len(container.lexical_index) == 0


def test_upload_pdf_and_docx(client: TestClient) -> None:
    pdf = make_pdf(
        [["# Intro", "Alpha project launched in 2024."], ["# Costs", "It cost 9 million."]]
    )
    docx = make_docx(SAMPLE_DOCX_PAGES, title="Retention Policy")
    pdf_doc = upload(client, "alpha.pdf", pdf, "application/pdf").json()
    docx_doc = upload(client, "policy.docx", docx).json()
    pdf_detail = client.get(f"{API}/documents/{pdf_doc['id']}").json()
    docx_detail = client.get(f"{API}/documents/{docx_doc['id']}").json()
    assert pdf_detail["status"] == "ready" and pdf_detail["page_count"] == 2
    assert docx_detail["status"] == "ready" and docx_detail["title"] == "Retention Policy"


def test_list_filters_and_pagination(client: TestClient) -> None:
    for i in range(3):
        upload(
            client,
            f"note_{i}.txt",
            f"Note number {i}. It has some content about topic {i}.".encode(),
        )
    page = client.get(f"{API}/documents", params={"limit": 2, "offset": 0}).json()
    assert page["total"] == 3 and len(page["items"]) == 2
    assert client.get(f"{API}/documents", params={"search": "note_1"}).json()["total"] == 1
    assert client.get(f"{API}/documents", params={"status": "failed"}).json()["total"] == 0
    assert client.get(f"{API}/documents", params={"status": "bogus"}).status_code == 422


@pytest.mark.parametrize(
    ("name", "data", "status", "code"),
    [
        ("malware.exe", b"MZ\x90\x00", 415, "unsupported_file_type"),
        ("noext", b"hello", 415, "unsupported_file_type"),
        ("fake.pdf", b"just text pretending to be a pdf", 415, "unsupported_file_type"),
        ("fake.docx", b"%PDF-1.4 not a docx", 415, "unsupported_file_type"),
        ("empty.txt", b"", 415, "unsupported_file_type"),
    ],
)
def test_upload_validation_errors(
    client: TestClient, name: str, data: bytes, status: int, code: str
) -> None:
    response = upload(client, name, data)
    assert response.status_code == status
    body = response.json()
    assert body["error"]["code"] == code
    assert body["request_id"]


def test_upload_too_large(client: TestClient, container: Container) -> None:
    container.storage.max_bytes = 100
    response = upload(client, "big.txt", b"a" * 101)
    assert response.status_code == 413
    assert response.json()["error"]["code"] == "file_too_large"


def test_duplicate_upload_conflict(client: TestClient) -> None:
    first = upload(client, "a.txt", b"Same content in both files.")
    second = upload(client, "b.txt", b"Same content in both files.")
    assert second.status_code == 409
    error = second.json()["error"]
    assert error["code"] == "duplicate_document"
    assert error["details"]["document_id"] == first.json()["id"]


def test_corrupt_pdf_is_marked_failed_and_can_be_reprocessed(client: TestClient) -> None:
    doc = upload(client, "broken.pdf", b"%PDF-1.4\nthis is not a real pdf body").json()
    detail = client.get(f"{API}/documents/{doc['id']}").json()
    assert detail["status"] == "failed"
    assert detail["error_message"]
    reprocessed = client.post(f"{API}/documents/{doc['id']}/reprocess")
    assert reprocessed.status_code == 202
    assert client.get(f"{API}/documents/{doc['id']}").json()["status"] == "failed"


def test_not_found_and_invalid_ids(client: TestClient) -> None:
    missing = "00000000-0000-0000-0000-000000000000"
    for method, path in [
        ("get", f"/documents/{missing}"),
        ("get", f"/documents/{missing}/chunks"),
        ("get", f"/documents/{missing}/file"),
        ("delete", f"/documents/{missing}"),
        ("post", f"/documents/{missing}/reprocess"),
    ]:
        response = getattr(client, method)(f"{API}{path}")
        assert response.status_code == 404, path
        assert response.json()["error"]["code"] == "document_not_found"
    invalid = client.get(f"{API}/documents/not-a-uuid")
    assert invalid.status_code == 422
    assert invalid.json()["error"]["code"] == "validation_error"


def test_unknown_route_uses_error_envelope(client: TestClient) -> None:
    response = client.get(f"{API}/nope")
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "not_found"


def test_bm25_index_is_rebuilt_on_startup(settings, fake_llm) -> None:  # type: ignore[no-untyped-def]
    from app.container import build_container
    from app.main import create_app

    first = build_container(settings, llm=fake_llm, synchronous_ingestion=True)
    with TestClient(create_app(settings, container=first)) as c:
        upload(c, "orion.txt", SAMPLE_TXT.encode())
        expected = len(first.lexical_index)
    assert expected > 0

    second = build_container(settings, llm=fake_llm, synchronous_ingestion=True)
    with TestClient(create_app(settings, container=second)):
        assert len(second.lexical_index) == expected
