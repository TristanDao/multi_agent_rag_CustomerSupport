"""Prompt-injection detection patterns and helpers."""
import re
from typing import Dict, List

INJECTION_PATTERNS: List[re.Pattern] = [
    re.compile(r"ignore (?:all|any|the|previous|above) (?:instructions?|prompts?|rules?)", re.IGNORECASE),
    re.compile(r"forget (?:all|everything|the) (?:instructions?|rules?|context)", re.IGNORECASE),
    re.compile(r"reveal (?:the |your )?(?:system|internal|hidden) ?prompt", re.IGNORECASE),
    re.compile(r"bypass (?:the |any )?(?:policy|policies|guardrails?|rules?)", re.IGNORECASE),
    re.compile(r"act as (?:a |an )?(?:developer|admin|root|system|dan|jailbreak)", re.IGNORECASE),
    re.compile(r"disable (?:safety|guardrails?|filters?)", re.IGNORECASE),
    re.compile(r"return (?:the |your )?(?:system|internal|secret) ?prompt", re.IGNORECASE),
    re.compile(r"you are now (?:free|unrestricted|unfiltered)", re.IGNORECASE),
    re.compile(r"pretend (?:to be|you are) (?:an? )?(?:admin|root|developer)", re.IGNORECASE),
    re.compile(r"do anything (?:now|without) (?:restrictions?|rules?)", re.IGNORECASE),
    re.compile(r"simulate (?:a )?(?:developer|debug) mode", re.IGNORECASE),
    re.compile(r"\btool\s*call[:=]\s*\{", re.IGNORECASE),
    re.compile(r"\bfunction\s*call[:=]\s*\{", re.IGNORECASE),
    re.compile(r"<\|im_start\|>", re.IGNORECASE),
    re.compile(r"<\|im_end\|>", re.IGNORECASE),
    # Vietnamese patterns
    re.compile(r"(?:quên|bỏ\s+qua|phớt\s+lờ)\s+(?:tất\s+cả\s+)?(?:hướng\s+dẫn|chỉ\s+dẫn|quy\s+tắc|nội\s+quy)", re.IGNORECASE),
    re.compile(r"(?:tiết\s+lộ|cho\s+(?:tôi|biết)\s+)?(?:system\s+prompt|internal\s+prompt|secret\s+prompt)", re.IGNORECASE),
    re.compile(r"(?:nói|hành\s+động)\s+như\s+(?:một\s+)?(?:nhân\s+viên|admin|root)\s+nội\s+bộ", re.IGNORECASE),
    re.compile(r"(?:vượt|bypass|phá)\s+(?:qua\s+)?(?:chính\s+sách|quy\s+tắc|hệ\s+thống)", re.IGNORECASE),
    re.compile(r"(?:trở\s+thành|trở\s+nên)\s+(?:admin|root|developer|developer\s+mode)", re.IGNORECASE),
]


SUSPICIOUS_INSTRUCTION_KEYWORDS = [
    "system prompt",
    "internal prompt",
    "internal wholesale margin",
    "doanh thu nội bộ",
    "lợi nhuận nội bộ",
    "show me your instructions",
]


def detect_prompt_injection(text: str) -> Dict:
    if not text:
        return {"is_injection": False, "matched": [], "score": 0.0}
    matched = []
    for p in INJECTION_PATTERNS:
        m = p.search(text)
        if m:
            matched.append(m.group(0)[:80])
    lowered = text.lower()
    for kw in SUSPICIOUS_INSTRUCTION_KEYWORDS:
        if kw in lowered:
            matched.append(kw)
    score = min(1.0, 0.4 * len({m for m in matched}))
    return {
        "is_injection": bool(matched),
        "matched": matched,
        "score": score,
    }
