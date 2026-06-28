"""Test PII redaction."""
from app.core.pii_redaction import detect_pii, redact_pii


def test_redact_phone():
    text = "SĐT tôi là 0909123456, kiểm tra đơn giúp tôi."
    redacted, changed = redact_pii(text)
    assert changed
    assert "[PHONE]" in redacted
    assert "0909123456" not in redacted


def test_redact_email():
    text = "Email của tôi là nguyenvana@gmail.com"
    redacted, changed = redact_pii(text)
    assert changed
    assert "[EMAIL]" in redacted


def test_detect_pii():
    text = "SĐT 0909123456 và email abc@gmail.com"
    res = detect_pii(text)
    assert res["has_pii"] is True
    assert "phone" in res["types"]
    assert "email" in res["types"]
