"""Test intent classifier (heuristic fallback path)."""
import asyncio

import pytest

from app.agents.intent_classifier import classify_intent


@pytest.mark.parametrize(
    "query,expected",
    [
        ("Tìm giấy A4 giá dưới 400k", "product_search"),
        ("Đơn DH00001 đang ở đâu?", "order_tracking"),
        ("Tôi nhận hàng 10 ngày rồi, còn đổi được không?", "return_refund"),
        ("Mua 50 thùng giấy A4 thì giá sỉ bao nhiêu?", "wholesale_pricing"),
        ("Bảo hành máy tính Casio bao lâu?", "warranty_policy"),
    ],
)
def test_heuristic_intent(query, expected):
    result = asyncio.run(classify_intent(query))
    assert result["intent"] == expected
    assert "required_agents" in result
