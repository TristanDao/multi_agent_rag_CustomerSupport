"""Load markdown documents and split into sections/chunks."""
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)


@dataclass
class DocumentChunk:
    doc_id: str
    source: str
    section: str
    content: str
    chunk_index: int


def _split_into_sections(text: str) -> List[tuple]:
    """Split a markdown file into (section_title, body) tuples based on `##` headers."""
    lines = text.splitlines()
    sections: List[tuple] = []
    current_title = "General"
    current_body: List[str] = []

    def flush() -> None:
        body = "\n".join(current_body).strip()
        if body:
            sections.append((current_title, body))

    for line in lines:
        m = re.match(r"^##\s+(.+)$", line)
        if m:
            flush()
            current_title = m.group(1).strip()
            current_body = []
        else:
            current_body.append(line)
    flush()
    return sections


def _word_chunks(text: str, chunk_size: int, overlap: int) -> List[str]:
    """Word-based chunker. Approximates tokens via word count (heuristic)."""
    words = text.split()
    if not words:
        return []
    chunks: List[str] = []
    step = max(1, chunk_size - overlap)
    for start in range(0, len(words), step):
        end = start + chunk_size
        piece = " ".join(words[start:end])
        if piece:
            chunks.append(piece)
        if end >= len(words):
            break
    return chunks


def load_markdown_dir(
    docs_dir: str | Path,
    chunk_size: int = 600,
    chunk_overlap: int = 100,
) -> List[DocumentChunk]:
    docs_dir = Path(docs_dir)
    if not docs_dir.exists():
        raise FileNotFoundError(f"docs dir not found: {docs_dir}")
    chunks: List[DocumentChunk] = []
    for path in sorted(docs_dir.glob("*.md")):
        try:
            text = path.read_text(encoding="utf-8")
        except Exception as e:
            logger.warning("read_failed path=%s err=%s", path.name, str(e))
            continue
        sections = _split_into_sections(text)
        for sec_title, body in sections:
            for idx, piece in enumerate(_word_chunks(body, chunk_size, chunk_overlap)):
                doc_id = f"{path.stem}#{sec_title}#{idx}"
                chunks.append(
                    DocumentChunk(
                        doc_id=doc_id,
                        source=path.name,
                        section=sec_title,
                        content=piece,
                        chunk_index=idx,
                    )
                )
    logger.info("docs_chunked count=%s", len(chunks))
    return chunks


def load_single_markdown(
    path: str | Path,
    chunk_size: int = 600,
    chunk_overlap: int = 100,
) -> List[DocumentChunk]:
    return load_markdown_dir(Path(path).parent, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
