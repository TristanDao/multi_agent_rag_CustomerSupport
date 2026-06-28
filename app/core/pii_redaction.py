"""PII detection and redaction utilities."""
import re
from typing import Tuple

PHONE_RE = re.compile(r"(\+?84|0)\d{9,10}\b")
EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
CARD_RE = re.compile(r"\b(?:\d[ -]*?){13,19}\b")
ADDRESS_HINT_RE = re.compile(
    r"(\d+\s+[A-Za-zÀ-ỹ][^\n,]{2,80}(?:đường|phố|phường|quận|huyện|tỉnh|thành phố|tp\.?|q\.?|p\.?|street|st\.?|road|rd\.?|district)\b[^\n,]*)",
    re.IGNORECASE,
)
API_KEY_RE = re.compile(r"(sk-[A-Za-z0-9]{20,}|AIza[0-9A-Za-z\-_]{20,})")
ID_NUMBER_RE = re.compile(r"\b\d{9,12}\b")

REDACTION_LABELS = {
    "phone": "[PHONE]",
    "email": "[EMAIL]",
    "card": "[CARD]",
    "address": "[ADDRESS]",
    "api_key": "[API_KEY]",
    "id_number": "[ID_NUMBER]",
}


def redact_pii(text: str) -> Tuple[str, bool]:
    if not text:
        return text, False
    redacted = text
    changed = False
    for pattern, label in (
        (API_KEY_RE, REDACTION_LABELS["api_key"]),
        (EMAIL_RE, REDACTION_LABELS["email"]),
        (CARD_RE, REDACTION_LABELS["card"]),
        (PHONE_RE, REDACTION_LABELS["phone"]),
        (ADDRESS_HINT_RE, REDACTION_LABELS["address"]),
    ):
        new_text, n = pattern.subn(label, redacted)
        if n:
            redacted = new_text
            changed = True
    return redacted, changed


def detect_pii(text: str) -> dict:
    if not text:
        return {"has_pii": False, "types": []}
    types = []
    if PHONE_RE.search(text):
        types.append("phone")
    if EMAIL_RE.search(text):
        types.append("email")
    if CARD_RE.search(text):
        types.append("card")
    if ADDRESS_HINT_RE.search(text):
        types.append("address")
    if API_KEY_RE.search(text):
        types.append("api_key")
    return {"has_pii": bool(types), "types": types}
