import io
import docx
import pypdf
from fastapi import UploadFile, HTTPException, status


SUPPORTED_CONTENT_TYPES = {
    "application/pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "text/plain",
}


async def extract_text_from_file(file: UploadFile) -> str:
    content_type = file.content_type

    if content_type not in SUPPORTED_CONTENT_TYPES:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Unsupported file type '{content_type}'. Accepted: pdf, docx, txt.",
        )

    raw = await file.read()

    if content_type == "text/plain":
        return raw.decode("utf-8")

    if content_type == "application/pdf":
        try:

            reader = pypdf.PdfReader(io.BytesIO(raw))
            return "\n".join(
                page.extract_text() or "" for page in reader.pages
            )
        except ImportError:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="pypdf is not installed. Run: pip install pypdf",
            )

    if "wordprocessingml" in content_type:
        try:

            doc = docx.Document(io.BytesIO(raw))
            return "\n".join(p.text for p in doc.paragraphs)
        except ImportError:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="python-docx is not installed. Run: pip install python-docx",
            )
