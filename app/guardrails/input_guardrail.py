"""Input guardrail: combines prompt-injection detection and PII check."""
from typing import Dict, Tuple

from app.core.pii_redaction import detect_pii, redact_pii
from app.guardrails.prompt_injection import detect_prompt_injection


SAFE_RESPONSE_INJECTION = (
    "Tôi có thể hỗ trợ các câu hỏi về sản phẩm, đơn hàng, hoặc chính sách của cửa hàng, "
    "nhưng tôi không thể thực hiện các yêu cầu vượt qua hệ thống hoặc chính sách kinh doanh."
)


def run_input_guardrail(message: str) -> Tuple[Dict, str]:
    """Returns (guardrail_result, processed_message).

    guardrail_result = {
        "blocked": bool,
        "reason": str|None,
        "safe_response": str|None,
        "pii_redacted": bool,
        "injection_detected": bool,
    }
    """
    if not message or not message.strip():
        return (
            {
                "blocked": True,
                "reason": "empty_message",
                "safe_response": "Vui lòng nhập nội dung câu hỏi.",
                "pii_redacted": False,
                "injection_detected": False,
            },
            message or "",
        )

    inj = detect_prompt_injection(message)
    redacted, pii_changed = redact_pii(message)

    if inj["is_injection"]:
        return (
            {
                "blocked": True,
                "reason": "Possible prompt injection attempt",
                "safe_response": SAFE_RESPONSE_INJECTION,
                "pii_redacted": pii_changed,
                "injection_detected": True,
                "injection_score": inj["score"],
                "injection_matches": inj["matched"],
            },
            redacted,
        )

    pii_info = detect_pii(redacted)
    return (
        {
            "blocked": False,
            "reason": None,
            "safe_response": None,
            "pii_redacted": pii_changed,
            "injection_detected": False,
            "pii_detected": pii_info["types"],
        },
        redacted,
    )
