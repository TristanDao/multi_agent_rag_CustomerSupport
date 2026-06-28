"""Security helpers: request id, basic secret/header handling."""
import secrets
from typing import Optional

from fastapi import Header


def new_request_id() -> str:
    return f"req_{secrets.token_hex(8)}"


def get_request_id(x_request_id: Optional[str] = Header(default=None)) -> str:
    if x_request_id and x_request_id.startswith("req_"):
        return x_request_id
    return new_request_id()


def mask_secret(value: str) -> str:
    if not value:
        return ""
    if len(value) <= 8:
        return "***"
    return f"{value[:4]}***{value[-4:]}"
