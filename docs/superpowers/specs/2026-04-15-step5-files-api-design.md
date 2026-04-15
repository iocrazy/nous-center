# Step 5 · Files API 设计

## Context

Step 4 里把 `input_image.file_id` 留了 501 桩。Step 5 落地 Files API，让 Responses / Chat / Batch 都能引用上传文件。对齐 OpenAI/Ark。

## 决策

1. **存储**：内容寻址本地文件系统 `backend/data/files/{sha256[:2]}/{sha256}`。按 hash 去重，不用对象存储（v2 打算换 MinIO，但不在这步）。
2. **表**：`files(id="file-xxx", instance_id, purpose, filename, bytes, mime_type, sha256, storage_path, created_at, expires_at nullable)`。`(instance_id, sha256)` 有 UNIQUE，复用同一内容不会重复存。
3. **purpose**：枚举 `user_data | assistants | batch | vision`。不校验语义，纯透传（Ark 也是这么做的）。
4. **端点**：
   - `POST /v1/files` multipart：fields = `file, purpose`；返回 `{id, bytes, filename, purpose, created_at}`
   - `GET /v1/files/{id}`、`GET /v1/files?purpose=&limit=&after=` 游标分页
   - `DELETE /v1/files/{id}` → 软删除（标 `deleted_at`，磁盘上的对象延后 GC，因为可能被别的 instance 共享 hash）—— 其实 v1 就硬删数据库行、磁盘对象留着由定期清理扫
   - `GET /v1/files/{id}/content` 二进制下载
5. **大小上限**：50 MB，超限 413。可配置但硬默认先卡死。
6. **鉴权**：`verify_bearer_token`（和 Responses 一致），instance-scoped：别的 instance 看不到也 GET 不到。
7. **Responses 集成**：`resolve_image(item)` 的 `file_id` 分支：读文件、base64 data URL、作为 `image_url` 传给 vLLM（vLLM `/v1/chat/completions` 接受 data URL）。非 image MIME 抛 400。
8. **清理**：新增轻量后台 loop，扫孤儿磁盘对象（DB 里没有行指向）每 6h 一次 —— v1 跳过，列出文件夹大小看需要再做。

## Schema

```python
class File(Base):
    __tablename__ = "files"
    id = Column(String(64), primary_key=True)  # file-{token_urlsafe(12)}
    instance_id = Column(BigInteger, ForeignKey("service_instances.id", ondelete="CASCADE"), index=True)
    purpose = Column(String(32), nullable=False)
    filename = Column(String(512), nullable=False)
    bytes = Column(BigInteger, nullable=False)
    mime_type = Column(String(128), nullable=False)
    sha256 = Column(String(64), nullable=False)
    storage_path = Column(String(1024), nullable=False)
    created_at = Column(DateTime(timezone=True), default=utcnow)
    __table_args__ = (
        UniqueConstraint("instance_id", "sha256", name="uq_files_instance_sha256"),
        Index("ix_files_instance_created", "instance_id", "created_at"),
    )
```

## 实现要点

- 上传走 `fastapi.UploadFile`，边读边 `hashlib.sha256` 累加（不全量入内存）。超限立刻中断。
- 临时文件写到 `{FILES_ROOT}/tmp/{uuid}`，sha256 算完后 `os.rename` 到终点（原子）。
- 如果 `(instance_id, sha256)` 已存在 → 返回既有 id，不创建新行（幂等）。磁盘 tmp 文件删除。
- MIME 用 `magic` 或仅信任 client Content-Type —— v1 信任 client，vision 路径自己判 image/* 前缀。

## 验证

```bash
# 1. 上传
curl -F file=@cat.png -F purpose=vision -H "Authorization: Bearer $KEY" \
  http://localhost:8000/v1/files
# → {"id":"file-xxx","bytes":...,"filename":"cat.png","purpose":"vision"}

# 2. GET / DOWNLOAD
curl -H "Authorization: Bearer $KEY" /v1/files/$FID
curl -H "Authorization: Bearer $KEY" /v1/files/$FID/content -o out.png
diff cat.png out.png  # 应相同

# 3. 幂等上传（同内容）→ 同 id
curl -F file=@cat.png -F purpose=vision ... # id 不变

# 4. 超限
dd if=/dev/urandom of=big bs=1M count=60
curl -F file=@big ... # 413

# 5. Responses 集成
curl -X POST /v1/responses -d "{\"model\":\"...vl\",\"input\":[
  {\"type\":\"input_text\",\"text\":\"what is this?\"},
  {\"type\":\"input_image\",\"file_id\":\"$FID\"}
]}"
# → 返回图描述
```

## 不做的事（YAGNI）

- 不做 S3/MinIO（v2 再换，留 storage 抽象点）
- 不做病毒扫描
- 不做签名下载 URL（直连后端即可）
- 不做 expires_at（除非 batch 需要，下一阶段再加字段）
