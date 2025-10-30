# 담당역할: 신준엽 - RAG 검색 도구 & 용어집 도구

## 담당자 정보
- **이름**: 신준엽
- **역할**: RAG 시스템 전문 담당
- **참여 기간**: 10/28 ~ 11/6 (전체 기간)
- **핵심 역할**: RAG 파이프라인 구현, Vector DB 검색, 용어집 시스템

---

## 담당 모듈 및 도구

### 1. RAG 시스템 (`src/rag/`)
- Langchain PGVector (PostgreSQL + pgvector) 연동
- VectorStoreRetriever 구현 (similarity, mmr)
- MultiQueryRetriever 구현 (쿼리 확장)
- ContextualCompressionRetriever (문맥 압축, 선택)
- 임베딩 관리 (OpenAIEmbeddings)
- 용어집 RAG 통합 (별도 pgvector 컬렉션)

### 2. 도구 1: RAG 검색 도구 (`src/tools/rag_search.py`)
- Langchain @tool 데코레이터로 search_paper_database 구현
- Retriever.invoke() 호출
- 메타데이터 필터링 (년도, 저자, 카테고리)
- 유사도 점수 반환
- PostgreSQL 메타데이터 조회

### 3. 도구 3: 용어집 도구 (`src/tools/glossary.py`)
- Langchain @tool 데코레이터로 search_glossary 구현
- 용어집 전용 VectorStore 검색
- PostgreSQL glossary 테이블 직접 검색
- 난이도별 설명 반환 (Easy/Hard)
- 하이브리드 검색 (PostgreSQL + Vector DB)

---

## 도구 1: RAG 검색 도구

### 기능 설명
로컬 Vector DB와 PostgreSQL에 저장된 논문 데이터베이스에서 관련 논문을 검색하는 도구

### Langchain 구현

#### 1. VectorStore 및 Retriever 초기화
```python
# src/rag/retriever.py

from langchain_postgres.vectorstores import PGVector
from langchain_openai import OpenAIEmbeddings
from langchain.retrievers import MultiQueryRetriever, ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
import os

class RAGRetriever:
    """논문 검색을 위한 RAG Retriever"""

    def __init__(self, llm):
        # OpenAI Embeddings 초기화
        self.embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small",
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )

        # PostgreSQL + pgvector VectorStore 초기화
        self.vectorstore = PGVector(
            collection_name="paper_chunks",
            embedding_function=self.embeddings,
            connection_string="postgresql://user:password@localhost:5432/papers"
        )

        # 기본 Retriever (MMR 방식)
        self.base_retriever = self.vectorstore.as_retriever(
            search_type="mmr",  # Maximal Marginal Relevance
            search_kwargs={
                "k": 5,  # 최종 반환 문서 수
                "fetch_k": 20,  # MMR 후보 문서 수
                "lambda_mult": 0.5  # 관련성 vs 다양성 균형
            }
        )

        # MultiQuery Retriever (쿼리 확장)
        self.multi_query_retriever = MultiQueryRetriever.from_llm(
            retriever=self.base_retriever,
            llm=llm
        )

    def retrieve(self, query: str, use_multi_query: bool = True):
        """문서 검색"""
        if use_multi_query:
            # 쿼리 확장 사용
            docs = self.multi_query_retriever.invoke(query)
        else:
            # 기본 검색
            docs = self.base_retriever.invoke(query)

        return docs

    def retrieve_with_filter(self, query: str, filter_dict: dict):
        """메타데이터 필터링을 포함한 검색"""
        docs = self.vectorstore.similarity_search(
            query,
            k=5,
            filter=filter_dict  # 예: {"year": {"$gte": 2020}}
        )
        return docs

    def retrieve_with_scores(self, query: str):
        """유사도 점수를 포함한 검색"""
        docs_with_scores = self.vectorstore.similarity_search_with_relevance_scores(
            query,
            k=5
        )
        return docs_with_scores
```

#### 2. RAG 검색 도구 구현
```python
# src/tools/rag_search.py

from langchain.tools import tool
from langchain.schema import Document
import psycopg2

@tool
def search_paper_database(query: str, year_filter: int = None) -> str:
    """
    논문 데이터베이스에서 관련 논문을 검색합니다.

    Args:
        query: 검색할 질문 또는 키워드
        year_filter: 년도 필터 (예: 2020 이상)

    Returns:
        관련 논문 내용 및 메타데이터
    """
    # 1. Vector DB에서 유사도 검색
    if year_filter:
        docs = rag_retriever.retrieve_with_filter(
            query,
            filter_dict={"year": {"$gte": year_filter}}
        )
    else:
        docs = rag_retriever.retrieve(query, use_multi_query=True)

    # 2. PostgreSQL에서 메타데이터 조회
    conn = psycopg2.connect("postgresql://user:password@localhost/papers")
    cursor = conn.cursor()

    results = []
    for doc in docs:
        paper_id = doc.metadata.get("paper_id")

        # 메타데이터 조회
        cursor.execute(
            "SELECT title, authors, publish_date, url FROM papers WHERE paper_id = %s",
            (paper_id,)
        )
        meta = cursor.fetchone()

        if meta:
            results.append({
                "title": meta[0],
                "authors": meta[1],
                "publish_date": meta[2],
                "url": meta[3],
                "content": doc.page_content,
                "section": doc.metadata.get("section", "본문")
            })

    cursor.close()
    conn.close()

    # 3. 결과 포맷팅
    formatted_results = format_search_results(results)
    return formatted_results


def format_search_results(results: list) -> str:
    """검색 결과를 LLM에 전달할 수 있는 형식으로 포맷팅"""
    if not results:
        return "관련 논문을 찾을 수 없습니다."

    output = "## 검색된 논문\n\n"

    for i, result in enumerate(results, 1):
        output += f"### {i}. {result['title']}\n"
        output += f"- **저자**: {result['authors']}\n"
        output += f"- **출판일**: {result['publish_date']}\n"
        output += f"- **URL**: {result['url']}\n"
        output += f"- **섹션**: {result['section']}\n\n"
        output += f"**내용**:\n{result['content']}\n\n"
        output += "---\n\n"

    return output
```

#### 3. MultiQueryRetriever (쿼리 확장)
```python
# MultiQueryRetriever 동작 방식

# 원본 쿼리: "Transformer 논문 설명해줘"
# → LLM이 자동으로 3-5개 변형 쿼리 생성:
#   1. "Transformer 아키텍처란?"
#   2. "Attention Is All You Need 논문 내용"
#   3. "Transformer 모델의 핵심 메커니즘"
# → 각 쿼리로 검색 → 결과 통합 → 중복 제거 → 최종 반환
```

#### 4. ContextualCompressionRetriever (선택 사항)
```python
# src/rag/compression.py

from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor

def create_compression_retriever(base_retriever, llm):
    """문맥 압축 Retriever 생성"""
    compressor = LLMChainExtractor.from_llm(llm)

    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=base_retriever
    )

    return compression_retriever

# 사용 예시
# 긴 문서를 검색 후, 질문과 관련된 부분만 추출하여 컨텍스트 크기 감소
```

### 사용하는 DB

#### PostgreSQL + pgvector (Vector DB)
- **컬렉션**: `paper_chunks`
- **저장 데이터**: 논문 본문을 청크로 나눈 임베딩 벡터 (pgvector extension 사용)
- **메타데이터**:
  - `paper_id`: 논문 ID (PostgreSQL과 연결)
  - `section`: 논문 섹션 (Abstract, Introduction 등)
  - `page_num`: 페이지 번호
  - `title`: 논문 제목
  - `authors`: 저자
  - `year`: 출판 년도
- **검색 방식**:
  - Cosine Similarity (기본)
  - L2 Distance
  - MMR (Maximal Marginal Relevance) - 관련성 + 다양성
  - MultiQuery (쿼리 확장)

#### PostgreSQL
- **테이블**: `papers`
  ```sql
  CREATE TABLE papers (
      paper_id SERIAL PRIMARY KEY,
      title VARCHAR(500) NOT NULL,
      authors TEXT,
      publish_date DATE,
      source VARCHAR(100),
      url TEXT UNIQUE,
      category VARCHAR(100),
      citation_count INT DEFAULT 0,
      abstract TEXT,
      created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
  );

  -- 인덱스
  CREATE INDEX idx_papers_title ON papers USING GIN (to_tsvector('english', title));
  CREATE INDEX idx_papers_category ON papers(category);
  CREATE INDEX idx_papers_date ON papers(publish_date);
  ```
- **역할**: 논문 메타데이터 저장 및 조회
- **쿼리**: paper_id로 제목, 저자, 년도, URL 등 조회

---

## 도구 3: 용어집 도구

### 기능 설명
논문에 자주 등장하는 전문 용어(Attention, Fine-tuning, BLEU Score 등)를 검색하여 난이도별 설명을 제공하는 도구

### Langchain 구현

#### 1. 용어집 VectorStore 초기화
```python
# src/rag/glossary_retriever.py

from langchain_postgres.vectorstores import PGVector
from langchain_openai import OpenAIEmbeddings

class GlossaryRetriever:
    """용어집 검색을 위한 Retriever"""

    def __init__(self):
        # OpenAI Embeddings 초기화
        self.embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small"
        )

        # 용어집 전용 VectorStore 초기화 (pgvector)
        self.glossary_vectorstore = PGVector(
            collection_name="glossary_embeddings",
            embedding_function=self.embeddings,
            connection_string="postgresql://user:password@localhost:5432/papers"
        )

        # Retriever 설정
        self.retriever = self.glossary_vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 3}
        )

    def search(self, term: str):
        """용어 검색"""
        docs = self.retriever.invoke(term)
        return docs
```

#### 2. 용어집 검색 도구 구현
```python
# src/tools/glossary.py

from langchain.tools import tool
import psycopg2

@tool
def search_glossary(term: str, difficulty: str = "easy") -> str:
    """
    논문 용어집에서 전문 용어를 검색하여 설명합니다.

    Args:
        term: 검색할 용어
        difficulty: 'easy' (초심자) 또는 'hard' (전문가)

    Returns:
        용어 정의 및 설명
    """
    # 1. PostgreSQL glossary 테이블에서 직접 검색 (빠름)
    conn = psycopg2.connect("postgresql://user:password@localhost/papers")
    cursor = conn.cursor()

    cursor.execute("""
        SELECT term, definition, easy_explanation, hard_explanation, category
        FROM glossary
        WHERE term ILIKE %s
    """, (f"%{term}%",))

    result = cursor.fetchone()

    if result:
        # PostgreSQL에서 찾은 경우
        term_name, definition, easy_exp, hard_exp, category = result

        if difficulty == "easy":
            explanation = easy_exp if easy_exp else definition
        else:
            explanation = hard_exp if hard_exp else definition

        output = f"## 📚 용어: {term_name}\n\n"
        output += f"**카테고리**: {category}\n\n"
        output += f"**설명**:\n{explanation}\n"

        cursor.close()
        conn.close()
        return output

    cursor.close()
    conn.close()

    # 2. PostgreSQL에 없으면 Vector DB에서 검색 (유연함)
    glossary_docs = glossary_retriever.search(term)

    if glossary_docs:
        # Vector DB에서 찾은 경우
        top_doc = glossary_docs[0]
        return f"## 📚 용어 관련 내용\n\n{top_doc.page_content}"

    # 3. 용어집에도 없으면 논문 본문에서 검색
    paper_docs = rag_retriever.retrieve(f"{term} 정의", use_multi_query=False)

    if paper_docs:
        context = paper_docs[0].page_content
        return f"## 📚 '{term}'에 대한 논문 내용\n\n{context}"

    return f"'{term}'에 대한 정보를 찾을 수 없습니다."
```

#### 3. 하이브리드 검색 (PostgreSQL + Vector DB)
```python
def hybrid_glossary_search(term: str, difficulty: str = "easy") -> str:
    """
    PostgreSQL과 Vector DB를 동시에 검색하여 최상의 결과 반환
    """
    results = {
        "postgres": None,
        "vector_db": None
    }

    # PostgreSQL 검색
    conn = psycopg2.connect("postgresql://user:password@localhost/papers")
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM glossary WHERE term ILIKE %s", (f"%{term}%",))
    results["postgres"] = cursor.fetchone()
    cursor.close()
    conn.close()

    # Vector DB 검색
    glossary_docs = glossary_retriever.search(term)
    if glossary_docs:
        results["vector_db"] = glossary_docs[0]

    # 결과 통합
    if results["postgres"]:
        # PostgreSQL 우선 (정확도 높음)
        return format_postgres_result(results["postgres"], difficulty)
    elif results["vector_db"]:
        # Vector DB (유연성 높음)
        return format_vector_result(results["vector_db"])
    else:
        return f"'{term}'에 대한 정보를 찾을 수 없습니다."
```

#### 4. 질문 분석 시 용어 자동 추출 및 컨텍스트 추가
```python
# src/rag/context_enhancer.py

def extract_and_add_glossary_context(user_query: str) -> str:
    """
    사용자 질문에서 전문 용어를 추출하여 프롬프트에 추가
    """
    conn = psycopg2.connect("postgresql://user:password@localhost/papers")
    cursor = conn.cursor()

    # 질문에서 용어 찾기 (PostgreSQL ILIKE 사용)
    cursor.execute("""
        SELECT term, definition, easy_explanation
        FROM glossary
        WHERE %s ILIKE '%' || term || '%'
    """, (user_query,))

    terms_found = cursor.fetchall()
    cursor.close()
    conn.close()

    if not terms_found:
        return ""

    # 용어 정의를 컨텍스트에 추가
    glossary_context = "\n\n## 📚 관련 용어 정의\n\n"
    for term, definition, easy_exp in terms_found:
        explanation = easy_exp if easy_exp else definition
        glossary_context += f"- **{term}**: {explanation}\n"

    return glossary_context
```

### 사용하는 DB

#### PostgreSQL
- **테이블**: `glossary`
  ```sql
  CREATE TABLE glossary (
      term_id SERIAL PRIMARY KEY,
      term VARCHAR(200) NOT NULL UNIQUE,
      definition TEXT NOT NULL,
      easy_explanation TEXT,  -- 초심자용 설명
      hard_explanation TEXT,  -- 전문가용 설명
      category VARCHAR(100),  -- ML, NLP, CV, RL 등
      difficulty_level VARCHAR(20),  -- beginner, intermediate, advanced
      related_terms TEXT[],  -- 관련 용어
      examples TEXT,  -- 사용 예시
      created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
  );

  -- 인덱스
  CREATE INDEX idx_glossary_term ON glossary(term);
  CREATE INDEX idx_glossary_category ON glossary(category);

  -- 예시 데이터
  INSERT INTO glossary (term, definition, easy_explanation, hard_explanation, category, difficulty_level)
  VALUES (
      'Attention Mechanism',
      'A technique that allows models to focus on specific parts of the input when generating output.',
      '책을 읽을 때 중요한 부분에 집중하는 것처럼, AI가 입력 데이터에서 중요한 부분에 집중하는 기술입니다.',
      'A weighted sum mechanism that computes attention scores between query and key vectors, allowing the model to dynamically focus on relevant input positions during sequence processing.',
      'Deep Learning',
      'intermediate'
  );
  ```
- **역할**: 용어 정의 및 난이도별 설명 저장

#### PostgreSQL + pgvector (Vector DB)
- **컬렉션**: `glossary_embeddings`
- **저장 데이터**: 용어 + 정의를 임베딩한 벡터 (pgvector extension 사용)
- **메타데이터**:
  - `term`: 용어명
  - `category`: 카테고리 (ML, NLP, CV 등)
  - `difficulty_level`: 난이도
- **검색 방식**: Cosine Similarity, L2 Distance (유사 용어 검색)

---

## 개발 일정

### Phase 1: RAG 시스템 기초 구현 (10/28~10/30)
- PostgreSQL + pgvector VectorStore 연동
- OpenAI Embeddings 초기화
- 기본 Retriever 구현 (similarity)
- search_paper_database 도구 기본 구현

### Phase 2: 고급 검색 기능 구현 (10/31~11/02)
- MultiQueryRetriever 구현 (쿼리 확장)
- MMR 검색 방식 적용
- 메타데이터 필터링
- 유사도 점수 반환

### Phase 3: 용어집 시스템 구현 (11/01~11/02)
- 용어집 전용 VectorStore 초기화
- search_glossary 도구 구현
- PostgreSQL glossary 테이블 연동
- 하이브리드 검색 구현

### Phase 4: 통합 및 최적화 (11/03~11/04)
- ContextualCompressionRetriever 구현 (선택)
- 검색 결과 포맷팅 개선
- PostgreSQL 연동 최적화
- 단위 테스트

---

## RAG 노드 구현 (LangGraph 통합)

```python
# src/agent/nodes.py

def search_paper_node(state: AgentState):
    """RAG 검색 노드"""
    question = state["question"]

    # RAG 검색 도구 호출
    search_result = search_paper_database.invoke({
        "query": question,
        "year_filter": None
    })

    # 검색 결과를 상태에 저장
    state["tool_result"] = search_result

    # LLM에 전달하여 최종 답변 생성
    difficulty = state.get("difficulty", "easy")

    prompt = f"""
    다음 논문 검색 결과를 바탕으로 사용자 질문에 답변해주세요.

    검색 결과:
    {search_result}

    사용자 질문: {question}

    난이도: {difficulty}

    답변:
    """

    response = llm.invoke([
        SystemMessage(content="당신은 논문 리뷰 전문가입니다."),
        HumanMessage(content=prompt)
    ])

    state["final_answer"] = response.content
    return state


def glossary_node(state: AgentState):
    """용어집 검색 노드"""
    question = state["question"]
    difficulty = state.get("difficulty", "easy")

    # 질문에서 용어 추출 (간단한 방법)
    term = question.replace("이 뭐야?", "").replace("란?", "").strip()

    # 용어집 검색 도구 호출
    glossary_result = search_glossary.invoke({
        "term": term,
        "difficulty": difficulty
    })

    state["final_answer"] = glossary_result
    return state
```

---

## Feature 브랜치

- `feature/rag-system` - RAG 시스템 기초 구현
- `feature/tool-rag-search` - RAG 검색 도구
- `feature/tool-glossary` - 용어집 도구
- `feature/rag-optimization` - 검색 최적화 (MultiQuery, MMR)

---

## 테스트 코드

```python
# tests/test_rag.py

import pytest
from src.rag.retriever import RAGRetriever
from src.tools.rag_search import search_paper_database

def test_rag_retriever():
    """RAG Retriever 테스트"""
    retriever = RAGRetriever(llm)

    # 기본 검색
    docs = retriever.retrieve("Transformer architecture")
    assert len(docs) > 0

    # 필터링 검색
    docs_filtered = retriever.retrieve_with_filter(
        "BERT",
        filter_dict={"year": {"$gte": 2018}}
    )
    assert len(docs_filtered) > 0

def test_search_paper_database():
    """RAG 검색 도구 테스트"""
    result = search_paper_database.invoke({
        "query": "Attention mechanism"
    })

    assert "검색된 논문" in result
    assert len(result) > 0
```

---

## 참고 자료

- Langchain Retrieval: https://python.langchain.com/docs/tutorials/rag/
- Langchain Vector Stores: https://python.langchain.com/docs/integrations/vectorstores/
- Langchain PGVector: https://python.langchain.com/docs/integrations/vectorstores/pgvector/
- Langchain Retrievers: https://python.langchain.com/docs/modules/data_connection/retrievers/
- MultiQueryRetriever: https://python.langchain.com/docs/modules/data_connection/retrievers/multi_query/
- ContextualCompressionRetriever: https://python.langchain.com/docs/modules/data_connection/retrievers/contextual_compression/
