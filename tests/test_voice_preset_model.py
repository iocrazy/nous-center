from src.models.voice_preset import VoicePreset, VoicePresetGroup


def test_voice_preset_creation():
    preset = VoicePreset(
        name="test-voice",
        engine="cosyvoice2",
        params={"voice": "default", "speed": 1.0, "sample_rate": 24000},
    )
    assert preset.name == "test-voice"
    assert preset.engine == "cosyvoice2"
    assert preset.params["speed"] == 1.0


def test_voice_preset_group_creation():
    group = VoicePresetGroup(
        name="test-group",
        presets=[{"role": "host", "voice_preset_id": "some-id"}],
    )
    assert group.name == "test-group"
    assert len(group.presets) == 1
