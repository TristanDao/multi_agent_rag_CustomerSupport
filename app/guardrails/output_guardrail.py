"""Output guardrail: claim checking + safety rewrites."""
from typing import Dict

from app.guardrails.claim_checker import check_unsupported_claims


SAFE_FALLBACK = (
    "Tôi chưa có đủ thông tin chính xác để trả lời câu hỏi này. "
    "Vui lòng cung cấp thêm chi tiết hoặc tôi sẽ chuyển cho nhân viên hỗ trợ."
)


def run_output_guardrail(answer: str) -> Dict:
    """Returns a guardrail result dict; may rewrite the answer if needed."""
    if not answer:
        return {
            "passed": False,
            "blocked": True,
            "rewritten_answer": SAFE_FALLBACK,
            "reason": "empty_answer",
        }
    check = check_unsupported_claims(answer)
    if not check["ok"]:
        return {
            "passed": False,
            "blocked": True,
            "rewritten_answer": SAFE_FALLBACK,
            "reason": "unsupported_claims",
            "violations": check["violations"],
        }
    return {
        "passed": True,
        "blocked": False,
        "rewritten_answer": answer,
        "reason": None,
    }
