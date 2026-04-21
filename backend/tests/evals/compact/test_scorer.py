
from tests.evals.compact.scorer import score_response


def test_score_perfect_match_gets_10():
    result = score_response(
        response="涉及变量、控制流、循环、函数",
        must_contain=["变量", "控制流", "循环", "函数"],
    )
    assert result["score"] == 10


def test_score_missing_half_gets_5():
    result = score_response(
        response="变量和循环",
        must_contain=["变量", "控制流", "循环", "函数"],
    )
    assert 4 <= result["score"] <= 6


def test_score_missing_all_gets_0():
    result = score_response(
        response="完全无关的内容",
        must_contain=["变量", "控制流"],
    )
    assert result["score"] == 0
