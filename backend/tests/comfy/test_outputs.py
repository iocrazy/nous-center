from src.services.comfy.outputs import classify_ext, collect_outputs


def test_classify_ext():
    assert classify_ext("a.mp4") == "video"
    assert classify_ext("a.PNG") == "image"
    assert classify_ext("a.flac") == "audio"
    assert classify_ext("a.txt") == "text"
    assert classify_ext("a.bin") == "file"


GRAPH = {
    "92": {"class_type": "SaveVideo", "inputs": {}},
    "50": {"class_type": "PreviewImage", "inputs": {}},
    "51": {"class_type": "SaveImage", "inputs": {}},
}
HISTORY = {"outputs": {
    "92": {"images": [{"filename": "out.mp4", "subfolder": "", "type": "output"}]},
    "50": {"images": [{"filename": "prev.png", "subfolder": "", "type": "temp"}]},
    "51": {"images": [{"filename": "final.png", "subfolder": "", "type": "output"}]},
}}


def test_preview_dropped_when_primary_image_exists():
    items = collect_outputs(HISTORY, GRAPH)
    names = {i.filename for i in items}
    assert names == {"out.mp4", "final.png"}
    assert next(i for i in items if i.filename == "out.mp4").kind == "video"


def test_preview_kept_when_only_output():
    hist = {"outputs": {"50": HISTORY["outputs"]["50"]}}
    items = collect_outputs(hist, GRAPH)
    assert [i.filename for i in items] == ["prev.png"]
