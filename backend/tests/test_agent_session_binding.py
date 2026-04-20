import pytest
from fastapi import HTTPException

from src.models.response_session import ResponseSession
from src.services.responses_service import assert_agent_matches_session


def make_session(agent_id: str | None = None) -> ResponseSession:
    return ResponseSession(
        id="session-test",
        instance_id=1,
        model="qwen3.5",
        agent_id=agent_id,
    )


def test_assert_agent_matches_none_request_uses_session_binding():
    sess = make_session(agent_id="tutor")
    result = assert_agent_matches_session(sess, request_agent=None)
    assert result == "tutor"


def test_assert_agent_matches_identical_ok():
    sess = make_session(agent_id="tutor")
    result = assert_agent_matches_session(sess, request_agent="tutor")
    assert result == "tutor"


def test_assert_agent_mismatch_raises_400():
    sess = make_session(agent_id="tutor")
    with pytest.raises(HTTPException) as exc_info:
        assert_agent_matches_session(sess, request_agent="writer")
    assert exc_info.value.status_code == 400
    assert "agent_session_mismatch" in str(exc_info.value.detail)


def test_assert_agent_on_unbound_session_raises_400():
    sess = make_session(agent_id=None)
    with pytest.raises(HTTPException) as exc_info:
        assert_agent_matches_session(sess, request_agent="tutor")
    assert exc_info.value.status_code == 400


def test_assert_no_agent_on_unbound_session_returns_none():
    sess = make_session(agent_id=None)
    result = assert_agent_matches_session(sess, request_agent=None)
    assert result is None
