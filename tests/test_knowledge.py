"""
Knowledge ingestion, end to end through the HTTP layer.

PDF and DOCX fixtures are real files built in-memory, so pypdf and
python-docx do real parsing rather than being mocked.
"""

import io

import pytest

from app.core.rag.utils import get_collection

DOCX_CONTENT_TYPE = (
    "application/vnd.openxmlformats-officedocument"
    ".wordprocessingml.document"
)


def build_pdf(text: str) -> bytes:
    """A minimal single-page PDF with one extractable text string."""
    content = f"BT /F1 24 Tf 72 720 Td ({text}) Tj ET".encode()
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
        b"/Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>",
        b"<< /Length "
        + str(len(content)).encode()
        + b" >>\nstream\n"
        + content
        + b"\nendstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]

    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for number, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += str(number).encode() + b" 0 obj\n" + body + b"\nendobj\n"

    xref_at = len(out)
    size = str(len(objects) + 1).encode()
    out += b"xref\n0 " + size + b"\n0000000000 65535 f \n"
    for offset in offsets:
        out += ("%010d 00000 n \n" % offset).encode()
    out += (
        b"trailer\n<< /Size "
        + size
        + b" /Root 1 0 R >>\nstartxref\n"
        + str(xref_at).encode()
        + b"\n%%EOF\n"
    )
    return bytes(out)


def build_docx(paragraphs: list[str]) -> bytes:
    import docx

    document = docx.Document()
    for paragraph in paragraphs:
        document.add_paragraph(paragraph)

    buffer = io.BytesIO()
    document.save(buffer)
    return buffer.getvalue()


# --- Raw text ingestion ---


def test_ingest_text_returns_201(client):
    response = client.post(
        "/knowledge",
        data={"doc_id": "policy", "text": "Refunds within 30 days."},
    )

    assert response.status_code == 201
    assert response.json()["doc_id"] == "policy"
    assert response.json()["status"] == "ingested"


def test_ingest_text_stores_chunks(client):
    client.post(
        "/knowledge",
        data={"doc_id": "policy", "text": "Refunds within 30 days."},
    )

    assert get_collection("test_tenant").count() > 0


def test_ingest_accepts_json_metadata(client):
    response = client.post(
        "/knowledge",
        data={
            "doc_id": "policy",
            "text": "Refunds within 30 days.",
            "metadata": '{"source": "handbook", "version": 2}',
        },
    )

    assert response.status_code == 201

    stored = get_collection("test_tenant").get()
    assert stored["metadatas"][0]["source"] == "handbook"


def test_ingest_rejects_malformed_metadata(client):
    response = client.post(
        "/knowledge",
        data={
            "doc_id": "policy",
            "text": "Some text.",
            "metadata": "{not valid json",
        },
    )

    assert response.status_code == 422
    assert "metadata" in response.json()["detail"]


# --- Required input ---


def test_ingest_requires_text_or_file(client):
    response = client.post("/knowledge", data={"doc_id": "policy"})

    assert response.status_code == 422
    assert "either" in response.json()["detail"].lower()


def test_ingest_rejects_both_text_and_file(client):
    response = client.post(
        "/knowledge",
        data={"doc_id": "policy", "text": "Some text."},
        files={"file": ("a.txt", b"File content", "text/plain")},
    )

    assert response.status_code == 422
    assert "not both" in response.json()["detail"].lower()


def test_ingest_requires_doc_id(client):
    response = client.post("/knowledge", data={"text": "Some text."})

    assert response.status_code == 422


# --- File uploads ---


def test_ingest_plain_text_file(client):
    response = client.post(
        "/knowledge",
        data={"doc_id": "notes"},
        files={
            "file": (
                "notes.txt",
                b"Shipping is free above fifty euros.",
                "text/plain",
            )
        },
    )

    assert response.status_code == 201
    assert get_collection("test_tenant").count() > 0


def test_ingest_pdf_file(client):
    pdf = build_pdf("The refund window is thirty days")

    response = client.post(
        "/knowledge",
        data={"doc_id": "policy-pdf"},
        files={"file": ("policy.pdf", pdf, "application/pdf")},
    )

    assert response.status_code == 201

    stored = get_collection("test_tenant").get()
    assert "thirty days" in " ".join(stored["documents"])


def test_ingest_docx_file(client):
    docx_bytes = build_docx(
        ["Support is available on weekdays.", "Escalations go to the lead."]
    )

    response = client.post(
        "/knowledge",
        data={"doc_id": "support-docx"},
        files={"file": ("support.docx", docx_bytes, DOCX_CONTENT_TYPE)},
    )

    assert response.status_code == 201

    stored = " ".join(get_collection("test_tenant").get()["documents"])
    assert "weekdays" in stored
    assert "Escalations" in stored


def test_ingest_rejects_unsupported_content_type(client):
    response = client.post(
        "/knowledge",
        data={"doc_id": "image"},
        files={"file": ("photo.png", b"\x89PNG\r\n", "image/png")},
    )

    assert response.status_code == 422
    assert "image/png" in response.json()["detail"]


@pytest.mark.parametrize(
    "filename,content_type",
    [
        ("a.txt", "text/plain"),
        ("a.pdf", "application/pdf"),
        ("a.docx", DOCX_CONTENT_TYPE),
    ],
)
def test_supported_types_are_accepted(client, filename, content_type):
    payload = {
        "text/plain": b"Plain text content here.",
        "application/pdf": build_pdf("PDF content here"),
        DOCX_CONTENT_TYPE: build_docx(["Docx content here."]),
    }[content_type]

    response = client.post(
        "/knowledge",
        data={"doc_id": f"doc-{filename}"},
        files={"file": (filename, payload, content_type)},
    )

    assert response.status_code == 201


# --- Tenant scoping ---


def test_ingestion_is_scoped_to_the_calling_tenant(client):
    client.post(
        "/knowledge",
        data={"doc_id": "policy", "text": "Tenant-specific content."},
    )

    assert get_collection("test_tenant").count() > 0
    assert get_collection("another_tenant").count() == 0


def test_ingest_requires_authentication(raw_client):
    response = raw_client.post(
        "/knowledge", data={"doc_id": "policy", "text": "Some text."}
    )

    assert response.status_code == 422


def test_ingest_rejects_an_invalid_api_key(raw_client):
    response = raw_client.post(
        "/knowledge",
        headers={"x-api-key": "not-a-real-key"},
        data={"doc_id": "policy", "text": "Some text."},
    )

    assert response.status_code == 401
