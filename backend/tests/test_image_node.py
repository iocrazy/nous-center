"""Tests for image_output node.

收敛后(spec 2026-05-21):全家桶 image_generate 已删除,图像生成走细粒度图
(flux2-components)。仅 image_output 终端节点保留。
"""
from __future__ import annotations


async def test_image_output_passes_through_image_envelope():
    from src.services.nodes.image import ImageOutputNode

    node = ImageOutputNode()
    out = await node.invoke(
        data={},
        inputs={
            "image_url": "/files/images/2026-05-04/abcd.png?token=t&expires=1",
            "media_type": "image/png",
            "width": 1024,
            "height": 1024,
            "stray": "ignored",
        },
    )
    assert out == {
        "image_url": "/files/images/2026-05-04/abcd.png?token=t&expires=1",
        "media_type": "image/png",
        "width": 1024,
        "height": 1024,
    }


async def test_image_output_defaults_when_input_missing():
    from src.services.nodes.image import ImageOutputNode

    out = await ImageOutputNode().invoke(data={}, inputs={})
    assert out["image_url"] is None
    assert "image" not in out
    assert out["media_type"] == "image/png"
    assert out["width"] is None


def test_image_output_registered_and_image_generate_gone():
    """image_output 仍注册;image_generate 收敛后已删除(走 flux2 细粒度图)。"""
    from src.services.nodes.registry import get_node_class

    assert get_node_class("image_output") is not None
    assert get_node_class("image_generate") is None
