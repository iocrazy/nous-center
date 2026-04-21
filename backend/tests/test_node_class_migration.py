import pytest

from src.services.nodes.base import InvokableNode
from src.services.nodes.text_io import TextInputNode, TextOutputNode, PassthroughNode


@pytest.mark.asyncio
async def test_text_input_node_returns_data_text():
    node = TextInputNode()
    assert isinstance(node, InvokableNode)
    result = await node.invoke({"text": "hello"}, {})
    assert result == {"text": "hello"}


@pytest.mark.asyncio
async def test_text_input_node_defaults_empty():
    node = TextInputNode()
    result = await node.invoke({}, {})
    assert result == {"text": ""}


@pytest.mark.asyncio
async def test_text_output_node_returns_inputs_text():
    node = TextOutputNode()
    result = await node.invoke({}, {"text": "out"})
    assert result == {"text": "out"}


@pytest.mark.asyncio
async def test_passthrough_node_returns_inputs():
    node = PassthroughNode()
    result = await node.invoke({}, {"key": "value"})
    assert result == {"key": "value"}


# -------- Subtask 4.4: equivalence tests (audio / io group) --------


@pytest.mark.asyncio
async def test_multimodal_input_node_equivalence():
    from src.services.workflow_executor import _exec_multimodal_input
    from src.services.nodes.text_io import MultimodalInputNode

    data = {
        "text": "hi",
        "images": ["data:image/png;base64,AAA"],
        "audio_data": "data:audio/wav;base64,BBB",
    }
    inputs = {}
    old = await _exec_multimodal_input(data, inputs)
    new = await MultimodalInputNode().invoke(data, inputs)
    assert old == new


@pytest.mark.asyncio
async def test_multimodal_input_node_equivalence_single_image():
    from src.services.workflow_executor import _exec_multimodal_input
    from src.services.nodes.text_io import MultimodalInputNode

    data = {
        "text": "hello",
        "image": "data:image/png;base64,CCC",
    }
    inputs = {}
    old = await _exec_multimodal_input(data, inputs)
    new = await MultimodalInputNode().invoke(data, inputs)
    assert old == new


@pytest.mark.asyncio
async def test_ref_audio_node_equivalence():
    from src.services.workflow_executor import _exec_ref_audio
    from src.services.nodes.audio import RefAudioNode

    data = {
        "path": "/tmp/ref.wav",
        "audio_data": "data:audio/wav;base64,DDD",
        "ref_text": "hello world",
    }
    inputs = {}
    old = await _exec_ref_audio(data, inputs)
    new = await RefAudioNode().invoke(data, inputs)
    assert old == new


@pytest.mark.asyncio
async def test_tts_engine_node_equivalence(monkeypatch):
    """TTS equivalence: both old and new should call the same adapter with the same args."""
    import base64
    from unittest.mock import MagicMock

    from src.services import workflow_executor as we
    from src.services.nodes.audio import TTSEngineNode

    # Build a fake TTS adapter that returns a deterministic result.
    fake_result = MagicMock()
    fake_result.audio_bytes = b"fake-wav-bytes"
    fake_result.sample_rate = 24000
    fake_result.duration_seconds = 1.5
    fake_result.format = "wav"

    fake_adapter = MagicMock()
    fake_adapter.is_loaded = True
    fake_adapter.synthesize = MagicMock(return_value=fake_result)

    fake_mgr = MagicMock()
    fake_mgr.get_adapter = MagicMock(return_value=fake_adapter)

    monkeypatch.setattr(we, "_model_manager", fake_mgr)

    data = {
        "engine": "cosyvoice2",
        "voice": "default",
        "speed": 1.0,
        "sample_rate": 24000,
    }
    inputs = {"text": "hello"}

    old = await we._exec_tts_engine(data, inputs)
    new = await TTSEngineNode().invoke(data, inputs)
    # Both should produce identical dicts (same fake adapter).
    assert old == new
    assert old == {
        "audio": base64.b64encode(b"fake-wav-bytes").decode(),
        "sample_rate": 24000,
        "duration_seconds": 1.5,
        "format": "wav",
    }


@pytest.mark.asyncio
async def test_output_node_equivalence():
    from src.services.workflow_executor import _exec_output
    from src.services.nodes.text_io import OutputNode

    data = {}
    inputs = {"text": "hello", "audio": "data:audio/wav;base64,XXX"}
    old = await _exec_output(data, inputs)
    new = await OutputNode().invoke(data, inputs)
    assert old == new


# -------- Subtask 4.4: equivalence tests (logic group) --------


@pytest.mark.asyncio
async def test_prompt_template_node_equivalence():
    from src.services.workflow_executor import _exec_prompt_template
    from src.services.nodes.logic import PromptTemplateNode

    data = {"template": "Hello {name}, today is {day}."}
    inputs = {"name": "heygo", "day": "Tuesday"}
    old = await _exec_prompt_template(data, inputs)
    new = await PromptTemplateNode().invoke(data, inputs)
    assert old == new
    assert old == {"text": "Hello heygo, today is Tuesday."}


@pytest.mark.asyncio
async def test_agent_node_equivalence(monkeypatch):
    """Agent equivalence: stub agent_manager + llm/tools so both paths run identically."""
    from src.services import workflow_executor as we
    from src.services import agent_manager
    from src.services import llm_service
    from src.services.nodes.logic import AgentNode

    # Stub the agent registry: one agent with no skills and a trivial prompt.
    fake_agent = {
        "prompts": {"IDENTITY.md": "You are a test agent."},
        "skills": [],
        "model": {"base_url": "http://localhost:8000", "model": "test-model"},
    }
    monkeypatch.setattr(agent_manager, "get_agent", lambda name: fake_agent)

    async def fake_call_llm_with_tools(**kwargs):
        return {"content": "hi from agent", "tool_calls": None}

    monkeypatch.setattr(llm_service, "call_llm_with_tools", fake_call_llm_with_tools)

    data = {"agent_name": "test"}
    inputs = {"text": "hello"}

    old = await we._exec_agent(data, inputs)
    new = await AgentNode().invoke(data, inputs)
    assert old == new
    assert old == {"text": "hi from agent"}


@pytest.mark.asyncio
async def test_python_code_node_equivalence(monkeypatch):
    """Python code equivalence: stub skill_tools._execute_python to avoid sandbox exec."""
    from src.services import skill_tools
    from src.services.workflow_executor import _exec_python_code
    from src.services.nodes.logic import PythonCodeNode

    async def fake_execute_python(code: str) -> str:
        return f"ran: {code}"

    monkeypatch.setattr(skill_tools, "_execute_python", fake_execute_python)

    data = {"code": "print('hi')"}
    inputs = {}
    old = await _exec_python_code(data, inputs)
    new = await PythonCodeNode().invoke(data, inputs)
    assert old == new
    assert old == {"text": "ran: print('hi')"}


@pytest.mark.asyncio
async def test_if_else_node_equivalence_contains():
    from src.services.workflow_executor import _exec_if_else
    from src.services.nodes.logic import IfElseNode

    data = {"condition": "hello", "match_type": "contains"}
    inputs = {"text": "hello world"}
    old = await _exec_if_else(data, inputs)
    new = await IfElseNode().invoke(data, inputs)
    assert old == new
    assert old == {"true": "hello world", "false": ""}


@pytest.mark.asyncio
async def test_if_else_node_equivalence_regex_fail():
    """Regex branch equivalence — valid regex that does not match."""
    from src.services.workflow_executor import _exec_if_else
    from src.services.nodes.logic import IfElseNode

    data = {"condition": r"^\d+$", "match_type": "regex"}
    inputs = {"text": "no digits here"}
    old = await _exec_if_else(data, inputs)
    new = await IfElseNode().invoke(data, inputs)
    assert old == new
    assert old == {"true": "", "false": "no digits here"}
