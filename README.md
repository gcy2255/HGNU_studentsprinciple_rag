# 学生手册 RAG 问答（LangChain + Chroma + 小米 MiMo）

对《黄冈师范学院学生手册（2022 年版）》做检索增强问答：问政策问题 → 在 PDF 里召回相关条文 → 由 MiMo v2.6 依据条文作答并标注页码。

## 环境

已在 `C:\ProgramData\anaconda3`（PyCharm 里这个项目的解释器）装好依赖，重装用：

```bash
py -3.12 -m pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

## 运行

```bash
python rag_pdf.py                              # 交互式问答
python rag_pdf.py "考试作弊会受到什么处理？"      # 单次提问
python rag_pdf.py --rebuild                    # 重建向量库
```

交互模式下输入 `exit` 退出，输入 `rebuild` 重建索引。

## 文件

| 文件 | 作用 |
| --- | --- |
| `rag_pdf.py` | 主程序：加载 → 清洗 → 切分 → 向量化 → 入库 → 检索 → 生成 |
| `rag.py` | 最初的小米 MiMo 接口连通性测试（纯对话，无 RAG） |
| `.env` | 存 `MIMO_API_KEY`（已被 .gitignore 忽略，不要提交） |
| `chroma_db/` | Chroma 向量库落盘目录，119 个文本块，可随时删除重建 |
| `.fastembed_cache/` | 中文向量模型 bge-small-zh-v1.5 缓存（约 91MB） |

## 流程与关键参数

1. **加载**：`PyPDFLoader` 读 50 页 PDF，页码写进 `metadata["page"]`（从 1 开始）。
2. **清洗**：`clean_text()` 去掉拼音水印 `XUESHENGSHOUCE`、页码，并把 PDF 硬换行切断的中文句子接回整句。
3. **切分**：`RecursiveCharacterTextSplitter(chunk_size=450, chunk_overlap=80)`，分隔符优先按「第X条」切，避免一条规定被拆断 → 118~119 块。
4. **向量化**：`fastembed` + `BAAI/bge-small-zh-v1.5`（512 维，onnxruntime，不需要 PyTorch）。
   提问走 `query_embed`（自动加 BGE 问题前缀），入库走 `passage_embed`。
   模型从 `HF_ENDPOINT=https://hf-mirror.com` 下载（huggingface.co 国内直连会超时）。
5. **向量库**：`langchain-chroma` 持久化到 `chroma_db/`，第二次启动直接复用，不再重新切分和向量化。
6. **检索**：`similarity_search_with_score(k=5)`，保留前 3 条 + 距离 ≤0.85 的，同一页去重。
7. **生成**：`ChatOpenAI(model="mimo-v2.6-pro", base_url="https://api.xiaomimimo.com/v1")`，
   prompt 要求「只用参考资料、没有就说没有、结论标 [第X页]」。

## 为什么向量化不用小米的接口

`https://api.xiaomimimo.com/v1/embeddings` 返回 404，MiMo 目前只提供
`mimo-v2.6-pro / flash / pro-ultraspeed` 等对话模型（还有 asr/tts），**没有 embeddings 接口**，
所以检索侧的向量化用本地 BGE 中文小模型，生成侧仍用 MiMo。

## 常见可调项

- 换 PDF：改 `PDF_PATH`，然后 `python rag_pdf.py --rebuild`。
- 想召回更多：`ask(..., k=8)`；想更严格：把 `0.85` 调小。
- 换成知识库更大的场景：`chunk_size` 调到 600~800，`chunk_overlap` 100。
- 想用别的 embedding：在 `BgeZhEmbeddings` 里换模型名，或换成 `langchain_huggingface` 的 `HuggingFaceEmbeddings`。