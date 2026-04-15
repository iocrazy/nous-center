# Step 9 · 多模态 VL 接入

## 现状

Step 5 已通 `input_image.file_id` → base64 data URL → vLLM `/v1/chat/completions` 链路。
缺的就是 a) 一个真的 VL 模型，b) vLLM 启动时正确开多模态。

## 决策

1. **首选模型**：Qwen2.5-VL-7B-Instruct（FP16 ~16GB，单 3090 装得下，社区验证最稳）
2. **VLLMAdapter 自动检测**：读 `config.json.architectures` / `vision_config`，命中就追加 `--limit-mm-per-prompt image=4`
3. **路径约定**：`LOCAL_MODELS_PATH/llm/Qwen2.5-VL-7B-Instruct`，下载脚本兜底
4. **API 透传**：用户侧 SDK 调用方式不变 —— `/v1/chat/completions` 直接走，`/v1/responses` 借 Step 5 的 file_id 通道

## 改动

- `backend/src/services/inference/llm_vllm.py`：`_auto_configure` 增加 `is_multimodal` 字段；`load()` 命中时追加 mm flag
- `backend/configs/models.yaml`：新增 `qwen2_5_vl_7b_instruct` 条目
- `backend/scripts/download_vl_model.py`：HF snapshot_download 直装到本地

## 验证（待模型下载完成）

```bash
# 1. 下载
python backend/scripts/download_vl_model.py
# 2. 启动接入点（自动加载 VL 时 vLLM 命令含 --limit-mm-per-prompt）
# 3. /v1/responses 多模态调用
KEY=sk-step4-xxx
FID=$(curl -F file=@cat.jpg -F purpose=vision -H "Authorization: Bearer $KEY" \
  http://localhost:8000/v1/files | jq -r .id)
curl -X POST http://localhost:8000/v1/responses \
  -H "Authorization: Bearer $KEY" -H "Content-Type: application/json" \
  -d "{
    \"model\":\"qwen2_5_vl_7b_instruct\",
    \"input\":[
      {\"type\":\"input_text\",\"text\":\"describe this image in one sentence\"},
      {\"type\":\"input_image\",\"file_id\":\"$FID\"}
    ]
  }"
# → output[0].content[0].text 应包含图像描述
```

## 不做的事

- 不做视频输入（vLLM 仍未稳定支持 Qwen2.5-VL 的 video token）
- 不做 32B / 72B 变体（先用 7B 跑通，更大模型按需）
- 不做前端图片预览（webrtc/canvas，未来再加）
