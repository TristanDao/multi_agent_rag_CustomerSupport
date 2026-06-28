"""Output claim checker: prevent unsupported business/financial claims."""
import re
from typing import Dict, List

UNSUPPORTED_CLAIM_PATTERNS: List[re.Pattern] = [
    re.compile(r"\b(đảm bảo|chắc chắn)\s+(100%|tuyệt đối)\b", re.IGNORECASE),
    re.compile(r"\bgiá\s+(?:rẻ\s+)?nhất\s+(?:thị\s+trường|toàn\s+quốc|cả\s+nước)\b", re.IGNORECASE),
    re.compile(r"\b(roi|lợi\s+nhuận|biên\s+lợi\s+nhuận)\s*(nội\s+bộ|thực|đích|thực\s+tế)\b", re.IGNORECASE),
    re.compile(r"\bsecret\s+recipe\b", re.IGNORECASE),
    re.compile(r"\bwholesale\s+margin\b", re.IGNORECASE),
    re.compile(r"\bchính\s+sách\s+bí\s+mật\b", re.IGNORECASE),
    re.compile(r"\binternal\s+(policy|policy|price|margin)\b", re.IGNORECASE),
]


FORBIDDEN_PROMISE_PATTERNS: List[re.Pattern] = [
    re.compile(r"\b(tôi|chúng\s+tôi)\s+(?:sẽ\s+)?(?:cam\s+kết|đảm\s+bảo)\s+(?:hoàn\s+tiền\s+100|không\s+rủi\s+ro)\b", re.IGNORECASE),
]


def check_unsupported_claims(text: str) -> Dict:
    if not text:
        return {"ok": True, "violations": []}
    violations: List[str] = []
    for p in UNSUPPORTED_CLAIM_PATTERNS:
        m = p.search(text)
        if m:
            violations.append(m.group(0)[:80])
    for p in FORBIDDEN_PROMISE_PATTERNS:
        m = p.search(text)
        if m:
            violations.append(m.group(0)[:80])
    return {"ok": not violations, "violations": violations}
