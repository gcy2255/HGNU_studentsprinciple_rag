"""
学生手册 RAG 问答系统
------------------------------------------------------------------
流程：PyPDFLoader 读取 PDF -> 文本清洗 -> 递归切分 -> BGE 中文向量化
      -> 存入 Chroma 向量库 -> 检索 top-k 片段 -> 小米 MiMo-v2.6 生成答案

用到的库：
    pip install langchain-community langchain-chroma langchain-openai
                langchain-text-splitters pypdf chromadb fastembed

用法：
    python rag_pdf.py                # 交互式问答
    python rag_pdf.py "考试作弊怎么处理"   # 单次提问
    python rag_pdf.py --rebuild      # 重建向量库（换了 PDF 或改了切分参数时用）
"""

import os
import re
import sys
import shutil
import warnings
from pathlib import Path

# ---------- 国内网络环境下的必要设置（要在导入 fastembed 之前设置） ----------
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")   # HuggingFace 镜像
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["ANONYMIZED_TELEMETRY"] = "False"                     # 关闭 Chroma 遥测

# langchain-community 已进入维护模式，导入时会提示，这里屏蔽掉免得干扰输出
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

from fastembed import TextEmbedding
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.embeddings import Embeddings
from langchain_core.prompt_values import ChatPromptValue
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ============================== 一、配置 ==============================
BASE_DIR = Path(__file__).resolve().parent

# 学生手册 PDF（桌面 downloads 文件夹里）
PDF_PATH = Path(r"C:\Users\gcy22\Desktop\downloads\黄冈师范学院学生手册(2022年版).pdf")

DB_DIR = BASE_DIR / "chroma_db"                 # 向量库落盘位置
CACHE_DIR = BASE_DIR / ".fastembed_cache"       # 向量模型缓存位置
COLLECTION = "student_handbook"

# ---- 大模型：小米 MiMo v2.6（OpenAI 兼容接口）----
MIMO_BASE_URL = "https://api.xiaomimimo.com/v1"
MIMO_MODEL = "mimo-v2.6-pro"


def load_api_key() -> str:
    """优先读环境变量，其次读同目录 .env 文件，最后从 rag.py 里取。"""
    key = os.environ.get("MIMO_API_KEY")
    if key:
        return key
    env_file = BASE_DIR / ".env"
    if env_file.exists():
        m = re.search(r'MIMO_API_KEY\s*=\s*"?([^"\s]+)"?', env_file.read_text(encoding="utf-8"))
        if m:
            return m.group(1)
    legacy = BASE_DIR / "rag.py"
    if legacy.exists():
        m = re.search(r'MIMO_API_KEY\s*=\s*"([^"]+)"', legacy.read_text(encoding="utf-8"))
        if m:
            return m.group(1)
    raise SystemExit("未找到 MIMO_API_KEY：请在环境变量或 .env 中设置")


# ============================== 二、向量化（Embedding） ==============================
class BgeZhEmbeddings(Embeddings):
    """把 fastembed 的 BGE 中文模型包装成 LangChain 的 Embeddings 接口。

    fastembed 走的是 onnxruntime，不需要装 PyTorch，模型只有几十 MB。
    """

    def __init__(self, model_name: str = "BAAI/bge-small-zh-v1.5"):
        self.model = TextEmbedding(model_name=model_name, cache_dir=str(CACHE_DIR))

    def embed_documents(self, texts):
        # passage_embed：给文档片段用，会自动加 BGE 的文档前缀
        return [vec.tolist() for vec in self.model.passage_embed(list(texts))]

    def embed_query(self, text):
        # query_embed：给问题用（BGE 要求问题侧加"为这个句子生成表示…"前缀）
        return [vec.tolist() for vec in self.model.query_embed([text])][0]


# ============================== 三、PDF 读取 + 切分 ==============================
def clean_text(text: str) -> str:
    """PDF 抽出来的中文会被硬换行切断，这里把被切断的行接回去。"""
    text = text.replace("\r", "")
    # 去掉每页的拼音水印（XUESHENGSHOUCE）和纯数字页码
    text = re.sub(r"XUE\s*SHENG\s*SHOU\s*CE", "", text, flags=re.IGNORECASE)
    text = re.sub(r"(?m)^\s*\d{1,3}\s*$", "", text)
    text = re.sub(r"[ \t\u3000]+", " ", text)
    lines = [line.strip() for line in text.split("\n")]

    out: list[str] = []
    for line in lines:
        if not line:
            continue
        prev_ok = out and not re.search(r"[。！？；：!?;:”』）】]$", out[-1])
        is_new_item = re.match(
            r"^(第[一二三四五六七八九十百零]+[条章节]|[一二三四五六七八九十]+、|（[一二三四五六七八九十]+）|\d+[.、])",
            line,
        )
        if prev_ok and not is_new_item:
            out[-1] += line            # 接着上一行
        else:
            out.append(line)           # 新的一段/新的一条
    return "\n".join(out)


def load_and_split():
    """PyPDFLoader 加载 -> 清洗 -> RecursiveCharacterTextSplitter 切分。"""
    if not PDF_PATH.exists():
        raise SystemExit(f"找不到 PDF：{PDF_PATH}")

    print(f"[1/3] 加载 PDF：{PDF_PATH.name}")
    pages = PyPDFLoader(str(PDF_PATH)).load()
    print(f"      共 {len(pages)} 页")

    # 把页码写进正文和 metadata，方便回答时标注出处
    for doc in pages:
        page_no = int(doc.metadata.get("page", 0)) + 1     # PyPDFLoader 页码从 0 开始
        doc.metadata["page"] = page_no
        doc.metadata["source_file"] = PDF_PATH.name
        doc.page_content = clean_text(doc.page_content)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=450,          # 中文政策条文，450 字左右一块够用
        chunk_overlap=80,        # 相邻块重叠，避免一条规定被切断后检索不到
        separators=["\n第", "\n（", "\n一、", "\n\n", "\n", "。", "；", "，", ""],
        length_function=len,
    )
    chunks = splitter.split_documents(pages)
    chunks = [c for c in chunks if len(c.page_content.strip()) >= 30]   # 丢掉封面之类的碎片
    print(f"[2/3] 切分完成：{len(chunks)} 个文本块")
    return chunks


# ============================== 四、向量库（Chroma） ==============================
def get_vectorstore(rebuild: bool = False) -> Chroma:
    embeddings = BgeZhEmbeddings()

    if rebuild and DB_DIR.exists():
        shutil.rmtree(DB_DIR)
        print("      已删除旧向量库")

    if DB_DIR.exists() and any(DB_DIR.iterdir()):
        print(f"[3/3] 复用已存在的向量库：{DB_DIR.name}")
        return Chroma(
            collection_name=COLLECTION,
            persist_directory=str(DB_DIR),
            embedding_function=embeddings,
        )

    chunks = load_and_split()
    print(f"[3/3] 向量化并写入 Chroma（第一次会下载向量模型 + 建索引，稍等）")
    store = Chroma(
        collection_name=COLLECTION,
        persist_directory=str(DB_DIR),
        embedding_function=embeddings,
    )
    store.add_documents(chunks)      # langchain-chroma 会自动落盘
    print(f"      完成，向量库共 {store._collection.count()} 条")
    return store


# ============================== 五、检索 + 生成 ==============================
PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "你是《黄冈师范学院学生手册（2022年版）》的问答助手。"
     "请严格依据【参考资料】回答学生的问题，遵守以下规则：\n"
     "1. 只使用参考资料中的信息作答，不得编造、不得补充手册以外的规定；\n"
     "2. 如果参考资料中没有相关内容，直接说明“手册中未找到相关规定”，再指出最接近可参考的章节；\n"
     "3. 回答用中文，条理清晰，涉及流程/条件/时限时用分点列出；\n"
     "4. 每条结论后面用 [第X页] 标注依据的页码。"),
    ("user", "【参考资料】\n{context}\n\n【问题】\n{question}"),
])


def format_docs(docs) -> str:
    parts = []
    for d in docs:
        parts.append(f"（第{d.metadata.get('page')}页）\n{d.page_content}")
    return "\n\n".join(parts)


def ask(store: Chroma, llm: ChatOpenAI, question: str, k: int = 5) -> str:
    # 召回：先取 Top-k，按相关性过滤掉明显不相关的（余弦距离 > 0.85）
    # 但无论分数如何都保留前 3 条，避免把答案所在页滤掉
    hits = store.similarity_search_with_score(question, k=k)
    good, seen = [], set()
    for idx, (doc, score) in enumerate(hits):
        if (idx < 3 or score <= 0.85) and doc.metadata.get("page") not in seen:
            good.append(doc)
            seen.add(doc.metadata.get("page"))
    good = good or [doc for doc, _ in hits[:2]]

    print("\n--- 召回片段 ---")
    for i, (doc, score) in enumerate(hits, 1):
        flag = "√" if doc in good else "×"
        preview = doc.page_content[:60].replace("\n", " ")
        print(f"{flag} {i}. 距离={score:.3f} 第{doc.metadata.get('page')}页 | {preview}...")

    context = format_docs(good)
    messages = PROMPT.format_messages(context=context, question=question)
    resp = llm.invoke(messages)
    return resp.content


def build_llm() -> ChatOpenAI:
    return ChatOpenAI(
        model=MIMO_MODEL,
        api_key=load_api_key(),
        base_url=MIMO_BASE_URL,
        temperature=0.3,
        timeout=120,
        max_retries=2,
        max_completion_tokens=1024,
    )


# ============================== 六、主程序 ==============================
def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    args = [a for a in sys.argv[1:]]
    rebuild = "--rebuild" in args
    args = [a for a in args if a != "--rebuild"]

    store = get_vectorstore(rebuild=rebuild)
    llm = build_llm()

    # 单次提问模式
    if args:
        question = " ".join(args)
        print(f"\n【问题】{question}")
        print(f"\n【回答】{ask(store, llm, question)}")
        return

    # 交互式模式
    print("\n" + "=" * 60)
    print(" 学生手册智能问答（输入 exit 退出，输入 rebuild 重建向量库）")
    print("=" * 60)
    while True:
        try:
            question = input("\n【问题】").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not question:
            continue
        if question.lower() in {"exit", "quit", "q"}:
            break
        if question.lower() == "rebuild":
            store = get_vectorstore(rebuild=True)
            continue
        try:
            answer = ask(store, llm, question)
            print(f"\n【回答】{answer}")
        except Exception as exc:
            print(f"\n[出错] {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    main()