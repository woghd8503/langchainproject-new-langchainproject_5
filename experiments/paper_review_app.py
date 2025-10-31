"""Streamlit 앱: 논문 리뷰 RAG & 에이전트 (최소 기능 버전)

구성 개요 (최소 단위):
- 웹 검색(duckduckgo, arXiv)을 통해 논문 후보 탐색
- 선택한 논문 텍스트를 분할/임베딩하여 로컬 FAISS 벡터DB(`/data/vectordb`)에 저장
- 메타데이터는 SQLite(`/data/rdbms/papers.db`)로 관리
- 용어집(자주 등장 용어) 자동 추출 → RAG시 참조
- 질문 응답(RAG) 시 EZ(초심자)/HARD(대학원 수준) 모드로 설명
- LangChain 도구화: retriever, glossary, explain (웹검색/파일저장은 제외하고 신규 도구 중심)

주의사항:
- 외부 LLM 키가 없을 경우 HuggingFace 임베딩 + 로컬/대체 LLM 시도로 동작(가능하면 OPENAI 또는 GROQ 또는 Ollama)
- 본 앱은 학습/실험용 최소구현이며, 대규모 데이터/고성능 설정은 별도 확장 필요

Project 규칙 반영:
- 데이터는 `/data` 하위 사용, 재사용 코드는 여기 파일에 최소화로 포함(실험 폴더)
- 함수/클래스는 snake_case/PascalCase, Google 스타일 docstring 사용
"""

from __future__ import annotations

import os
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

# 환경 변수 설정 (최우선 - 모든 임포트 전에 설정)
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
# TORCH_LOGS는 제거 (유효한 값이 아니면 에러 발생)
# 대신 logging 모듈로 로깅 레벨 제어
if "TORCH_LOGS" in os.environ and os.environ["TORCH_LOGS"] == "+error":
    del os.environ["TORCH_LOGS"]  # 잘못된 값 제거
# PyTorch 중복 로드 방지
if "TORCH_LOADED" not in os.environ:
    os.environ["TORCH_LOADED"] = "1"

import streamlit as st
from dotenv import load_dotenv
import warnings
import logging
import sys

# PyTorch 경고 메시지 숨김 (가장 먼저)
warnings.filterwarnings("ignore", message=".*Examining the path of torch.classes.*")
warnings.filterwarnings("ignore", message=".*torch.classes.*")
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message=".*TORCH_LIBRARY.*")

# .env 파일에서 환경 변수 로드 (명시적 경로 지정) - OpenAI API 키 로드
env_path = Path(__file__).resolve().parents[1] / ".env"
if env_path.exists():
    load_dotenv(dotenv_path=env_path, override=True)
else:
    load_dotenv(override=True)

# 로깅 레벨 조정
logging.getLogger("torch").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)

# PyTorch 중복 임포트 방지 - 이미 로드된 경우 스킵
if "torch" in sys.modules:
    # 이미 로드되어 있으면 transformers만 안전하게 로드
    pass

# LangChain 관련 (PyTorch 에러 방지를 위해 핵심 임포트만)
from langchain_core.documents import Document
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda

# PyTorch 관련 임포트는 필요할 때만 (에러 처리 강화)
RecursiveCharacterTextSplitter = None
FAISS = None
HuggingFaceEmbeddings = None

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except Exception as e:
    # PyTorch 에러를 포함한 모든 임포트 에러를 무시하고 지연 로딩으로 처리
    if "TORCH_LIBRARY" in str(e) or "torch" in str(e).lower():
        pass  # 나중에 지연 로딩으로 시도

try:
    from langchain_community.vectorstores import FAISS
except Exception as e:
    if "TORCH_LIBRARY" in str(e) or "torch" in str(e).lower():
        pass

try:
    # langchain-huggingface로 마이그레이션 시도 (새 패키지)
    try:
        from langchain_huggingface import HuggingFaceEmbeddings
    except ImportError:
        # 대체: 기존 패키지 사용 (deprecated)
        from langchain_community.embeddings import HuggingFaceEmbeddings
except Exception as e:
    if "TORCH_LIBRARY" in str(e) or "torch" in str(e).lower():
        pass

try:
    from langchain_community.chat_models import ChatOllama
except Exception:  # noqa: BLE001
    ChatOllama = None  # type: ignore[assignment, misc]

# 검색 도구(키 불필요 우선)과 arXiv
from langchain_community.tools import DuckDuckGoSearchRun
try:
    # 구조화 결과를 제공하는 도구 (가능 시 사용)
    from langchain_community.tools import DuckDuckGoSearchResults
except Exception:  # noqa: BLE001
    DuckDuckGoSearchResults = None  # type: ignore[assignment]

# Google Scholar 간단 검색을 위한 임포트
try:
    from googlesearch import search as google_search
except ImportError:
    try:
        from googlesearch_python import search as google_search
    except ImportError:
        google_search = None
from langchain_community.document_loaders import ArxivLoader

try:
    from langchain_openai import ChatOpenAI
except Exception:  # noqa: BLE001
    ChatOpenAI = None  # type: ignore[assignment]

try:
    from langchain_groq import ChatGroq
except Exception:  # noqa: BLE001
    ChatGroq = None  # type: ignore[assignment]


# ========= 경로/상수 설정 =========
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
VDB_DIR = DATA_DIR / "vectordb"
RDBMS_DIR = DATA_DIR / "rdbms"
RDBMS_DIR.mkdir(parents=True, exist_ok=True)
VDB_DIR.mkdir(parents=True, exist_ok=True)

SQLITE_PATH = RDBMS_DIR / "papers.db"
FAISS_DIR = VDB_DIR / "papers_faiss"


# ========= 유틸리티 =========
def ensure_sqlite_schema(db_path: Path) -> None:
    """SQLite 스키마를 초기화한다.

    Args:
        db_path: SQLite 파일 경로
    """
    with sqlite3.connect(db_path.as_posix()) as conn:
        cur = conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS papers (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source TEXT NOT NULL,
                title TEXT,
                url TEXT,
                paper_id TEXT,
                summary TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS glossary (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                term TEXT NOT NULL,
                definition TEXT,
                score REAL DEFAULT 0.0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            """
        )
        conn.commit()


def insert_paper(
    source: str,
    title: Optional[str],
    url: Optional[str],
    paper_id: Optional[str],
    summary: Optional[str],
) -> None:
    """논문 메타데이터를 DB에 저장한다.

    Args:
        source: 수집 출처(e.g., "arxiv", "web")
        title: 논문 제목
        url: 원문 URL
        paper_id: 식별자(e.g., arXiv ID)
        summary: 간단 요약
    """
    ensure_sqlite_schema(SQLITE_PATH)
    with sqlite3.connect(SQLITE_PATH.as_posix()) as conn:
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO papers (source, title, url, paper_id, summary) VALUES (?, ?, ?, ?, ?)",
            (source, title, url, paper_id, summary),
        )
        conn.commit()


def upsert_glossary(terms: List[Tuple[str, str, float]]) -> None:
    """용어집 용어를 upsert에 준하게 단순 삽입(중복 무시)한다.

    Args:
        terms: (term, definition, score) 리스트
    """
    ensure_sqlite_schema(SQLITE_PATH)
    with sqlite3.connect(SQLITE_PATH.as_posix()) as conn:
        cur = conn.cursor()
        cur.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_glossary_term ON glossary(term)")
        for term, definition, score in terms:
            try:
                cur.execute(
                    "INSERT OR IGNORE INTO glossary (term, definition, score) VALUES (?, ?, ?)",
                    (term, definition, score),
                )
            except sqlite3.Error:
                continue
        conn.commit()


def load_glossary() -> List[Tuple[str, str, float]]:
    """저장된 용어집을 불러온다.

    Returns:
        용어, 정의, 점수 튜플 리스트
    """
    ensure_sqlite_schema(SQLITE_PATH)
    with sqlite3.connect(SQLITE_PATH.as_posix()) as conn:
        cur = conn.cursor()
        cur.execute("SELECT term, definition, score FROM glossary ORDER BY score DESC, term ASC")
        rows = cur.fetchall()
    return [(r[0], r[1] or "", float(r[2] or 0.0)) for r in rows]


# ========= 임베딩/LLM 선택 =========
def get_embeddings_model():
    """키가 필요 없는 기본 임베딩 모델을 반환한다.

    Returns:
        HuggingFaceEmbeddings 인스턴스
    """
    # 지연 로딩 처리
    global HuggingFaceEmbeddings
    if HuggingFaceEmbeddings is None:
        try:
            # langchain-huggingface로 마이그레이션 시도 (새 패키지)
            try:
                from langchain_huggingface import HuggingFaceEmbeddings
            except ImportError:
                # 대체: 기존 패키지 사용 (deprecated)
                from langchain_community.embeddings import HuggingFaceEmbeddings
        except Exception as e:
            raise RuntimeError(
                f"임베딩 모델을 임포트할 수 없습니다: {e}\n"
                f"PyTorch 충돌 문제일 수 있습니다. 앱을 재시작해보세요."
            )
    
    try:
        return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    except Exception as e:
        # TensorFlow 충돌 시 대체 방법
        st.error(f"임베딩 모델 로딩 실패: {e}")
        st.info("임베딩 모델을 다시 시도합니다...")
        import time
        time.sleep(1)
        # 재시도
        try:
            return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        except Exception:
            raise RuntimeError("임베딩 모델을 로드할 수 없습니다. TensorFlow/sentence-transformers 설치를 확인하세요.")


def get_chat_model() -> BaseChatModel:
    """OpenAI LLM을 반환한다.

    Returns:
        ChatOpenAI 인스턴스

    Raises:
        RuntimeError: OpenAI API 키가 없거나 초기화 실패 시
    """
    if ChatOpenAI is None:
        raise RuntimeError("langchain_openai가 설치되지 않았습니다. pip install langchain-openai")
    
    # .env 파일에서 다시 로드 확인
    env_path = Path(__file__).resolve().parents[1] / ".env"
    if env_path.exists():
        load_dotenv(dotenv_path=env_path, override=True)
    else:
        load_dotenv(override=True)
    
    openai_key = os.getenv("OPENAI_API_KEY")
    if not openai_key:
        env_path = Path(__file__).resolve().parents[1] / ".env"
        raise RuntimeError(
            f"OPENAI_API_KEY가 설정되지 않았습니다.\n"
            f".env 파일 위치: {env_path}\n"
            f".env 파일에 다음을 추가하세요:\nOPENAI_API_KEY=sk-proj-..."
        )
    
    openai_key = openai_key.strip()
    # API 키 형식 검증
    if not (openai_key.startswith("sk-") or openai_key.startswith("sk-proj-")):
        raise RuntimeError(
            f"OPENAI_API_KEY 형식이 올바르지 않습니다.\n"
            f"키는 'sk-' 또는 'sk-proj-'로 시작해야 합니다.\n"
            f"현재 키 시작: {openai_key[:10]}"
        )
    
    if len(openai_key) < 20:
        raise RuntimeError(
            f"OPENAI_API_KEY가 너무 짧습니다.\n"
            f"올바른 키인지 확인하세요. (현재 길이: {len(openai_key)}자)"
        )
    
    try:
        model_name = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        model = ChatOpenAI(
            model=model_name,
            temperature=0.2,
            timeout=30,
            max_retries=2,  # 재시도 설정
        )
        return model
    except Exception as e:
        error_msg = str(e)
        if "api_key" in error_msg.lower() or "authentication" in error_msg.lower():
            raise RuntimeError(
                f"OpenAI API 키 인증 실패.\n"
                f"오류: {error_msg}\n"
                f".env 파일의 OPENAI_API_KEY를 확인하세요."
            )
        else:
            raise RuntimeError(
                f"OpenAI 초기화 실패: {error_msg}\n"
                f"인터넷 연결 및 API 키를 확인하세요."
            )


# ========= 수집/분할/색인 =========
def chunk_documents(texts: List[str], source: str) -> List[Document]:
    """텍스트 리스트를 Document로 변환하고 분할한다.

    Args:
        texts: 원문 텍스트 리스트
        source: 출처 식별 문자열

    Returns:
        Document 분할 리스트
    """
    # 지연 로딩 처리
    global RecursiveCharacterTextSplitter
    if RecursiveCharacterTextSplitter is None:
        try:
            from langchain_text_splitters import RecursiveCharacterTextSplitter
        except Exception as e:
            raise RuntimeError(f"텍스트 스플리터를 임포트할 수 없습니다: {e}")
    
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=120)
    docs: List[Document] = []
    for idx, text in enumerate(texts):
        base = Document(page_content=text, metadata={"source": source, "idx": idx})
        # split_documents는 Document 리스트를 반환하므로 extend 사용
        split_docs = splitter.split_documents([base])
        docs.extend(split_docs)
    return docs


def build_or_load_faiss(docs: List[Document]):
    """FAISS 벡터스토어를 생성하거나 기존 인덱스를 불러온다.

    Args:
        docs: 색인할 문서 리스트(없으면 기존 로딩만)

    Returns:
        FAISS 벡터스토어
    """
    # 지연 로딩 처리
    global FAISS
    if FAISS is None:
        try:
            from langchain_community.vectorstores import FAISS
        except Exception as e:
            raise RuntimeError(f"FAISS를 임포트할 수 없습니다: {e}")
    
    try:
        embeddings = get_embeddings_model()
    except Exception as e:
        st.error(f"임베딩 모델 로딩 실패: {e}")
        raise
    
    # Document 타입 검증
    if docs:
        valid_docs = []
        for d in docs:
            if isinstance(d, Document) and hasattr(d, 'page_content'):
                valid_docs.append(d)
            else:
                st.warning(f"잘못된 문서 형식: {type(d)}")
        docs = valid_docs
    
    try:
        if FAISS_DIR.exists():
            # 기존 인덱스 로드
            vs = FAISS.load_local(FAISS_DIR.as_posix(), embeddings, allow_dangerous_deserialization=True)
            if docs:
                # 새 문서 추가
                vs.add_documents(docs)
                vs.save_local(FAISS_DIR.as_posix())
        elif docs:
            # 문서가 있을 때 새 인덱스 생성
            vs = FAISS.from_documents(docs, embeddings)
            vs.save_local(FAISS_DIR.as_posix())
        else:
            # 문서도 없고 기존 인덱스도 없을 때는 더미 문서로 초기화
            dummy_doc = Document(page_content="dummy", metadata={})
            vs = FAISS.from_documents([dummy_doc], embeddings)
        return vs
    except Exception as e:
        st.error(f"FAISS 색인 생성 실패: {e}")
        import traceback
        st.code(traceback.format_exc())
        raise


# ========= 간단 검색(DuckDuckGo + arXiv) =========
def search_papers(query: str, num_results: int = 5) -> List[Tuple[str, str]]:
    """arXiv와 DuckDuckGo로 논문을 검색한다.

    Args:
        query: 검색 질의
        num_results: 최대 결과 수

    Returns:
        (title, url) 튜플 리스트
    """
    results: List[Tuple[str, str]] = []
    
    # 1) arXiv 우선 검색 (안정적)
    try:
        arxiv_docs = load_arxiv(query, max_results=3)
        for d in arxiv_docs:
            title = d.metadata.get("Title") or d.metadata.get("title") or "arXiv 논문"
            url = d.metadata.get("pdf_url") or d.metadata.get("entry_id") or ""
            if url and url.startswith("http"):
                results.append((title, url))
                if len(results) >= num_results:
                    break
    except Exception as e:
        st.write(f"arXiv 검색 실패: {e}")
    
    # 2) DuckDuckGo 추가 검색 (보조)
    if len(results) < num_results:
        try:
            search = DuckDuckGoSearchRun()
            search_query = f"{query} arxiv research paper"
            search_text = search.run(search_query)
            
            # URL 추출 및 arxiv 필터링
            urls = list({u for u in re.findall(r"https?://[^\s)]+", search_text) if "arxiv" in u.lower()})
            for url in urls[:num_results - len(results)]:
                results.append((f"DuckDuckGo Result {len(results)+1}", url))
        except Exception as e:
            st.write(f"DuckDuckGo 검색 실패: {e}")
    
    return results[:num_results]


def load_arxiv(query: str, max_results: int = 3) -> List[Document]:
    """arXiv에서 간단히 논문 메타데이터를 불러온다.

    Args:
        query: 검색 질의
        max_results: 결과 수

    Returns:
        Document 리스트
    """
    try:
        # PDF 없이 메타데이터만 로드
        import arxiv
        client = arxiv.Client()
        search = arxiv.Search(query=query, max_results=max_results, sort_by=arxiv.SortCriterion.Relevance)
        results = list(client.results(search))
        
        docs: List[Document] = []
        for result in results:
            # 메타데이터만 사용
            metadata = {
                "Title": result.title,
                "Authors": ", ".join([str(a) for a in result.authors]),
                "Summary": result.summary,
                "Published": str(result.published),
                "id": result.entry_id,
                "pdf_url": result.pdf_url if hasattr(result, 'pdf_url') else result.entry_id,
            }
            # 제목과 요약만으로 Document 생성
            page_content = f"Title: {result.title}\n\nSummary: {result.summary}"
            docs.append(Document(page_content=page_content, metadata=metadata))
        
        return docs
    except Exception as e:
        st.write(f"arXiv 검색 오류: {e}")
        import traceback
        traceback.print_exc()
        return []


# ========= 용어집 추출 =========
def extract_glossary_terms(llm: BaseChatModel, texts: List[str], top_k: int = 20) -> List[Tuple[str, str, float]]:
    """텍스트에서 자주 등장/중요한 용어를 간단히 추출한다.

    Args:
        llm: 사용 LLM (OpenAI)
        texts: 원문 텍스트 리스트
        top_k: 최대 용어 수

    Returns:
        (term, definition, score) 리스트
    """
    if not texts:
        return []
    
    joined = "\n\n".join(texts)[:120000]
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
                너는 과학/공학 논문의 핵심 용어를 추출하는 조수이다.
                - 전문 용어를 한국어로 적고, 가능한 경우 간략 정의를 1-2문장으로 제공하라.
                - 중요도 점수(0-1)를 함께 추정하라.
                - JSON 배열로만 응답하라. 각 항목 형식: term, definition, score
                """.strip(),
            ),
            (
                "human",
                """
                다음 텍스트에서 핵심 용어를 {k}개까지 뽑아서 JSON 배열로 반환하라.
                텍스트:
                {text}
                """.strip(),
            ),
        ]
    )
    
    try:
        chain = prompt | llm
        resp = chain.invoke({"text": joined, "k": top_k})
        content = resp.content if hasattr(resp, "content") else str(resp)
    except Exception as e:
        error_str = str(e)
        
        # 프롬프트 템플릿 오류
        if isinstance(e, KeyError) or "missing variables" in error_str.lower():
            st.error(f"❌ 프롬프트 템플릿 오류: {error_str}")
            return []
        
        # OpenAI API 오류
        if "AuthenticationError" in error_str or "invalid_api_key" in error_str.lower():
            st.error("❌ OpenAI API 키 인증 실패")
            st.info("💡 .env 파일의 OPENAI_API_KEY를 확인하세요.")
        elif "timeout" in error_str.lower():
            st.error("❌ OpenAI API 타임아웃")
            st.info("네트워크 연결을 확인하고 다시 시도하세요.")
        elif "RateLimit" in error_str or "429" in error_str:
            st.error("❌ OpenAI API 사용량 제한")
            st.info("잠시 후 다시 시도하세요.")
        else:
            st.error(f"❌ OpenAI 호출 실패: {error_str[:300]}")
        
        return []
    
    # 매우 관대한 JSON 추출
    json_blob = re.findall(r"\[(?:.|\n)*?\]", content)
    items: List[Tuple[str, str, float]] = []
    if json_blob:
        import json
        try:
            parsed = json.loads(json_blob[0])
            for it in parsed:
                term = str(it.get("term", "")).strip()
                definition = str(it.get("definition", "")).strip()
                score = float(it.get("score", 0.0))
                if term:
                    items.append((term, definition, max(0.0, min(1.0, score))))
        except Exception:
            items = []

    # LLM 결과가 없으면 휴리스틱(빈도 기반)으로 대체 추출
    if not items:
        raw = joined.lower()
        # 간단 토크나이즈: 한글/영문/숫자 조합 토큰만 남김
        tokens = re.findall(r"[\w가-힣]+", raw)
        stop = {
            "the","and","for","that","with","this","from","have","has","are","was","were","can","will","into","your","you","our","their","but","not","all","any","each","other","more","most","some","such","no","nor","too","very","of","in","to","on","by","as","at","an","a","is","it","be","or","we",
            "그리고","또는","하지만","그러나","그리고","이는","있는","없는","에서","으로","에게","그","이","저","것","수","등","및","하면","하였다","하는","하였다","대한","까지","에서","으로","하는","하는데"
        }
        freq: dict[str, int] = {}
        for t in tokens:
            if len(t) < 2 or t in stop:
                continue
            freq[t] = freq.get(t, 0) + 1
        # 상위 토큰 선정
        top = sorted(freq.items(), key=lambda x: x[1], reverse=True)[: top_k]
        if top:
            max_c = max(c for _, c in top) or 1
            for term, c in top:
                score = c / max_c
                items.append((term, "", float(score)))
            # 휴리스틱 사용 안내
            st.info("LLM 응답이 불안정하여 빈도 기반으로 용어를 추출했습니다.")

    return items[:top_k]


# ========= RAG 체인 =========
def build_rag_chain(llm: BaseChatModel, retriever) -> RunnableLambda:
    """RAG 체인을 구성한다.

    Args:
        llm: 사용 LLM
        retriever: langchain retriever

    Returns:
        Runnable 체인
    """
    system_tmpl = (
        """
        너는 논문 리뷰 도우미이다. 주어진 문서 컨텍스트만 신뢰하고 한국어로 답한다.
        - EZ 모드: 초심자도 이해할 수 있도록 비유/예시와 함께 쉽게 설명
        - HARD 모드: 대학원 수준으로 수식/정의/근거를 간결히 포함
        - 출처 문맥 없으면 모른다고 말하고, 추측하지 말 것
        """.strip()
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_tmpl),
            (
                "human",
                """
                모드: {mode}
                질문: {question}
                문맥:
                {context}
                """.strip(),
            ),
        ]
    )

    def format_docs(docs: List[Document]) -> str:
        return "\n\n".join([d.page_content for d in docs])

    return ({"context": retriever | format_docs} | prompt | llm)  # type: ignore[return-value]


# ========= 간단 에이전트형 도구 =========
def tool_retrieve(vs: FAISS, query: str, k: int = 5) -> List[Document]:
    """벡터스토어에서 관련 문서를 조회한다.

    Args:
        vs: FAISS 벡터스토어
        query: 질의
        k: 상위 문서 수

    Returns:
        관련 문서 리스트
    """
    return vs.as_retriever(search_kwargs={"k": k}).get_relevant_documents(query)


def tool_explain_with_mode(llm: BaseChatModel, context_docs: List[Document], question: str, mode: str) -> str:
    """문맥과 모드에 따라 설명을 생성한다.

    Args:
        llm: 사용 LLM
        context_docs: 문맥 문서들
        question: 질문 텍스트
        mode: "EZ" 또는 "HARD"

    Returns:
        한국어 설명 문자열
    """
    rag = build_rag_chain(llm, retriever=context_docs)
    # retriever 자리에 문서 리스트를 직접 넘기기 위해 간단 어댑터
    def _retrieve(_: str) -> List[Document]:
        return context_docs

    rag = build_rag_chain(llm, retriever=RunnableLambda(_retrieve))
    ans = rag.invoke({"mode": mode, "question": question})
    return ans.content if hasattr(ans, "content") else str(ans)


# ========= Streamlit UI =========
def ui_search_and_ingest(llm: BaseChatModel) -> None:
    """탭: 검색 및 수집/색인.

    Args:
        llm: 사용 LLM (요약 등에 이용)
    """
    st.subheader("1) 논문 검색 → 선택 수집/색인")
    q = st.text_input("검색 질의(키워드)", value="Large Language Model alignment survey")
    if st.button("논문 검색"):
        with st.spinner("논문 검색 중..."):
            results = search_papers(q, num_results=10)
        st.session_state["web_results"] = results
        if not results:
            st.info("검색 결과가 없습니다. 다른 키워드로 시도해주세요.")

    web_results = st.session_state.get("web_results", [])

    if web_results:
        st.write("웹 검색 결과:")
        for title, url in web_results:
            st.markdown(f"- [{title}]({url})")

    # Scholar만 사용하므로 arXiv 문서 표시는 제거

    st.markdown("---")
    st.write("선택 수집/색인")
    with st.form("ingest_form"):
        use_web = st.checkbox("검색 결과(URL)만 색인")
        submitted = st.form_submit_button("색인 실행")

    if submitted:
        texts: List[str] = []
        if use_web and web_results:
            # 최소구현: URL 자체를 문서로 저장(실서버라면 크롤링/파싱 필요)
            texts.extend([f"URL 참고: {u}" for _, u in web_results])
            for _, u in web_results:
                insert_paper(source="web", title=None, url=u, paper_id=None, summary=None)

        # Scholar만 사용하므로 arXiv 색인 로직 제거

        if not texts:
            st.warning("색인할 텍스트가 없습니다. 검색 후 체크박스를 선택하세요.")
            return

        docs = chunk_documents(texts, source="ingested")
        with st.spinner("임베딩/색인 중..."):
            try:
                vs = build_or_load_faiss(docs)
                st.success(f"색인 완료! 현재 벡터DB 크기: {vs.index.ntotal}")
            except RuntimeError as e:
                st.error(f"색인 실패: {e}")
                st.info("임베딩 모델 로딩 문제입니다. TensorFlow/sentence-transformers 설치를 확인하세요.")
            except Exception as e:
                st.error(f"색인 중 오류 발생: {e}")
                import traceback
                st.code(traceback.format_exc())


def ui_glossary(llm: BaseChatModel) -> None:
    """탭: 용어집 생성/보기.

    Args:
        llm: 사용 LLM
    """
    st.subheader("2) 용어집 생성")
    top_k = st.slider("최대 용어 수", 5, 50, 20)
    if st.button("현재 코퍼스에서 용어집 추출"):
        if not FAISS_DIR.exists():
            st.warning("먼저 색인을 생성하세요.")
            return
        
        try:
            vs = build_or_load_faiss([])
        except Exception as e:
            st.error(f"벡터스토어 로딩 실패: {e}")
            return
        
        # 간단히 상위 임의 문서 일부만 취해 용어 추출(최소 구현)
        try:
            sample_docs = vs.similarity_search("overview", k=20)
            texts = [d.page_content for d in sample_docs]
        except Exception as e:
            st.error(f"문서 검색 실패: {e}")
            return
        
        if not texts:
            st.warning("색인된 문서가 없습니다.")
            return
        
        with st.spinner("용어 추출 중..."):
            terms = extract_glossary_terms(llm, texts, top_k=top_k)
        
        if not terms:
            st.warning("용어를 추출하지 못했습니다.")
            st.info("💡 OpenAI API 키가 올바른지 확인하고 다시 시도하세요.")
            return
        
        try:
            upsert_glossary(terms)
            st.success(f"{len(terms)}개 용어를 저장했습니다.")
        except Exception as e:
            st.error(f"용어 저장 실패: {e}")

    st.markdown("---")
    st.write("저장된 용어집:")
    rows = load_glossary()
    if not rows:
        st.info("아직 용어집이 비어 있습니다.")
    else:
        for term, definition, score in rows[:200]:
            st.markdown(f"- **{term}** (점수 {score:.2f}) — {definition}")


def ui_rag_qa(llm: BaseChatModel) -> None:
    """탭: RAG 질문 응답(EZ/HARD 모드).

    Args:
        llm: 사용 LLM
    """
    st.subheader("3) RAG 질문 응답")
    if not FAISS_DIR.exists():
        st.warning("먼저 색인을 생성하세요.")
        return

    try:
        vs = build_or_load_faiss([])
        retriever = vs.as_retriever(search_kwargs={"k": 6})
        chain = build_rag_chain(llm, retriever)
    except Exception as e:
        st.error(f"RAG 체인 생성 실패: {e}")
        return

    mode = st.radio("설명 모드", options=["EZ", "HARD"], horizontal=True)
    question = st.text_area("질문을 입력하세요", "이 논문의 핵심 기여는 무엇인가요?")
    if st.button("답변 생성"):
        if not question.strip():
            st.warning("질문을 입력하세요.")
            return
        
        try:
            with st.spinner("생성 중..."):
                ans = chain.invoke({"mode": mode, "question": question})
            st.markdown(ans.content if hasattr(ans, "content") else str(ans))
        except Exception as e:
            error_msg = str(e)
            if "AuthenticationError" in error_msg or "invalid_api_key" in error_msg.lower():
                st.error("❌ OpenAI API 키 인증 실패")
                st.info("💡 .env 파일의 OPENAI_API_KEY를 확인하세요.")
            else:
                st.error(f"❌ 답변 생성 실패: {error_msg[:300]}")


def main() -> None:
    """Streamlit 메인 진입점."""
    st.set_page_config(page_title="논문 리뷰 RAG/Agent (실험)", page_icon="📄", layout="wide")
    st.title("논문 리뷰 RAG & 에이전트 - 최소 기능 데모")
    st.caption("웹검색/색인 → 용어집 → RAG(EZ/HARD)")

    with st.sidebar:
        st.markdown("### 설정")
        
        # LLM 상태 표시
        try:
            llm = get_chat_model()
            st.success(f"✅ OpenAI 연결됨")
            
            # API 키 상태 표시
            openai_key = os.getenv("OPENAI_API_KEY", "")
            if openai_key:
                key_preview = openai_key[:7] + "..." + openai_key[-4:] if len(openai_key) > 11 else "설정됨"
                st.caption(f"🔑 API 키: {key_preview}")
                # 모델 정보 표시
                model_name = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
                st.caption(f"🤖 모델: {model_name}")
            else:
                st.caption("🔑 API 키: 미설정")
        except RuntimeError as e:
            st.error(f"❌ OpenAI 초기화 실패")
            st.code(str(e))
            env_path = Path(__file__).resolve().parents[1] / ".env"
            if env_path.exists():
                st.info(f"💡 .env 파일 위치: {env_path}")
                # .env 파일 내용 확인 (키만)
                try:
                    with open(env_path, "r", encoding="utf-8") as f:
                        content = f.read()
                        if "OPENAI_API_KEY" in content:
                            st.success("✅ .env 파일에 OPENAI_API_KEY가 있습니다.")
                        else:
                            st.warning("⚠️ .env 파일에 OPENAI_API_KEY가 없습니다.")
                except Exception:
                    pass
            else:
                st.warning(f"⚠️ .env 파일이 없습니다: {env_path}")
            llm = None
        
        st.markdown("---")
        st.markdown("데이터 경로")
        st.code(f"VDB: {FAISS_DIR}\nDB: {SQLITE_PATH}")

    if llm is None:
        return

    tabs = st.tabs(["검색/색인", "용어집", "RAG Q&A"])
    with tabs[0]:
        ui_search_and_ingest(llm)
    with tabs[1]:
        ui_glossary(llm)
    with tabs[2]:
        ui_rag_qa(llm)


if __name__ == "__main__":
    main()


