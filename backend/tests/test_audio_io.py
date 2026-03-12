from src.services.audio_io import AudioIOClient


def test_audio_io_client_has_methods():
    """AudioIOClient should expose info, resample, concat, split, convert."""
    client = AudioIOClient(base_url="http://localhost:8001")
    assert hasattr(client, "info")
    assert hasattr(client, "resample")
    assert hasattr(client, "concat")
    assert hasattr(client, "split")
    assert hasattr(client, "convert")
