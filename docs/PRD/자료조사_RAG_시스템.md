# 자료조사: RAG 시스템 설계

## 문서 정보
- **작성일**: 2025-10-29
- **프로젝트**: 논문 리뷰 챗봇 (AI Agent + RAG)
- **팀명**: 연결의 민족

---

## 1. RAG (Retrieval-Augmented Generation) 개요

### 1.1 RAG란?

**RAG**는 외부 지식 베이스(Knowledge Base)에서 관련 정보를 검색(Retrieval)하여 LLM의 답변 생성(Generation)을 보강하는 기술입니다.

### 1.2 RAG의 필요성

1. **LLM의 한계 극복**
   - LLM은 학습 데이터 기준 시점 이후의 정보를 모름
   - 특정 도메인(논문)에 대한 깊이 있는 지식 부족
   - Hallucination(환각) 문제 해결

2. **논문 리뷰 챗봇에서의 중요성**
   - 수천 개의 논문 정보를 LLM이 직접 학습 불가능
   - 정확한 출처와 인용 제공 필요
   - 최신 논문까지 커버 가능

---

## 2. RAG 시스템 구조

### 2.1 전체 파이프라인

```
[데이터 수집] → [전처리] → [청크 분할] → [임베딩] → [Vector DB 저장]
                                                            ↓
[사용자 질문] → [질문 임베딩] → [유사도 검색] → [관련 문서 조회] → [LLM에 전달] → [답변 생성]
```

### 2.2 Mermaid 다이어그램

```mermaid
graph TB
    subgraph Prepare["🔸 단계 1: 데이터 준비 & 저장"]
        direction LR
        A[논문 데이터 수집<br/>📊 arXiv, Scholar] --> B[데이터 전처리<br/>PDF → 텍스트]
        B --> C[텍스트 청크 분할<br/>1000자 단위]
        C --> D[임베딩 벡터 생성<br/>OpenAI Embeddings]
        D --> E[💾 Vector DB 저장<br/>ChromaDB]
    end

    subgraph Search["🔹 단계 2: 검색 & 조회"]
        direction LR
        F[사용자 질문 입력] --> G[질문 임베딩<br/>벡터 변환]
        G --> H[유사도 검색<br/>Cosine Similarity]
        H --> I[관련 문서 Top-K<br/>상위 5개 조회]
    end

    subgraph Generate["🔺 단계 3: 답변 생성"]
        direction LR
        J[컨텍스트 구성<br/>문서 + 질문] --> K[LLM 답변 생성<br/>GPT-4]
        K --> L[✅ 사용자 응답<br/>난이도 적용]
    end

    %% 단계 간 연결
    Prepare --> Search
    Search --> Generate
    E -.-> H
    I --> J

    %% Subgraph 스타일
    style Prepare fill:#e1f5ff,stroke:#01579b,stroke-width:3px,color:#000
    style Search fill:#f3e5f5,stroke:#4a148c,stroke-width:3px,color:#000
    style Generate fill:#e8f5e9,stroke:#1b5e20,stroke-width:3px,color:#000

    %% 노드 스타일 (데이터 준비 - 파랑 계열)
    style A fill:#90caf9,stroke:#1976d2,color:#000
    style B fill:#81d4fa,stroke:#0288d1,color:#000
    style C fill:#64b5f6,stroke:#1976d2,color:#000
    style D fill:#42a5f5,stroke:#1565c0,color:#000
    style E fill:#1e88e5,stroke:#0d47a1,color:#fff

    %% 노드 스타일 (검색 - 보라 계열)
    style F fill:#ce93d8,stroke:#7b1fa2,color:#000
    style G fill:#ba68c8,stroke:#7b1fa2,color:#fff
    style H fill:#ab47bc,stroke:#4a148c,color:#fff
    style I fill:#9c27b0,stroke:#4a148c,color:#fff

    %% 노드 스타일 (답변 생성 - 녹색 계열)
    style J fill:#a5d6a7,stroke:#388e3c,color:#000
    style K fill:#81c784,stroke:#2e7d32,color:#000
    style L fill:#66bb6a,stroke:#2e7d32,color:#fff
```

---

## 3. 데이터 수집 및 전처리

### 3.1 논문 데이터 수집

**데이터 소스:**
1. arXiv API (최신 논문)
2. Google Scholar (인용 정보)
3. Semantic Scholar API (논문 메타데이터)
4. 수동 업로드 (PDF 파일)

### 3.2 데이터 텍스트화

**지원 형식:**
- PDF → `PyPDF2`, `pdfplumber`
- HTML → `BeautifulSoup4`
- LaTeX → `pandoc`

**구현 예시:**
```python
from langchain.document_loaders import PyPDFLoader, ArxivLoader

# PDF 파일 로드
pdf_loader = PyPDFLoader("data/raw/transformer_paper.pdf")
documents = pdf_loader.load()

# arXiv에서 직접 로드
arxiv_loader = ArxivLoader(query="attention mechanism", max_docs=10)
arxiv_docs = arxiv_loader.load()
```

### 3.3 메타데이터 추출

**추출 정보:**
- 제목 (Title)
- 저자 (Authors)
- 출판일 (Publication Date)
- 출처 (Source: arXiv, IEEE, ACL 등)
- 키워드/카테고리 (Keywords/Category)
- DOI/URL
- 초록 (Abstract)

**PostgreSQL 저장:**
```python
import psycopg2

def save_paper_metadata(paper_data):
    conn = psycopg2.connect("postgresql://user:password@localhost/papers")
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO papers (title, authors, publish_date, source, url, abstract, category)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
        RETURNING paper_id
    """, (
        paper_data['title'],
        paper_data['authors'],
        paper_data['publish_date'],
        paper_data['source'],
        paper_data['url'],
        paper_data['abstract'],
        paper_data['category']
    ))

    paper_id = cursor.fetchone()[0]
    conn.commit()
    return paper_id
```

---

## 4. 텍스트 청크 분할 (Text Splitting)

### 4.1 청크 분할의 필요성

- LLM의 컨텍스트 윈도우 제한
- 임베딩 모델의 토큰 제한
- 검색 정확도 향상 (작은 청크가 더 관련성 높은 정보 포함)

### 4.2 청크 분할 전략

**RecursiveCharacterTextSplitter (추천)**

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,  # 청크 크기 (문자 수)
    chunk_overlap=200,  # 청크 간 중복 (맥락 유지)
    separators=["\n\n", "\n", ". ", " ", ""],  # 분할 우선순위
    length_function=len
)

chunks = text_splitter.split_documents(documents)
```

**논문 구조 기반 분할 (고급)**

논문의 섹션 구조를 활용한 분할:
- Abstract (초록)
- Introduction (서론)
- Related Work (관련 연구)
- Method (방법론)
- Experiments (실험)
- Conclusion (결론)

```python
def split_by_sections(paper_text):
    sections = {
        "abstract": extract_section(paper_text, "Abstract"),
        "introduction": extract_section(paper_text, "Introduction"),
        "method": extract_section(paper_text, "Method"),
        "experiments": extract_section(paper_text, "Experiments"),
        "conclusion": extract_section(paper_text, "Conclusion")
    }
    return sections
```

**청크 크기 권장사항:**
- **Small (500-800자)**: 정확한 검색, 작은 질문에 적합
- **Medium (1000-1500자)**: 균형잡힌 선택 (추천)
- **Large (2000-3000자)**: 넓은 맥락, 요약 작업에 적합

---

## 5. 임베딩 (Embedding)

### 5.1 임베딩 모델 선택

**OpenAI Embedding Models:**

| 모델 | 차원 | 비용 (1M tokens) | 사용 권장 |
|------|------|------------------|-----------|
| text-embedding-3-small | 1536 | $0.02 | ✅ 권장 (비용 효율) |
| text-embedding-3-large | 3072 | $0.13 | 높은 정확도 필요 시 |
| text-embedding-ada-002 | 1536 | $0.10 | 구버전 |

**구현 예시:**
```python
from langchain.embeddings import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    openai_api_key="your-api-key"
)

# 텍스트 임베딩
vector = embeddings.embed_query("Transformer architecture")
print(len(vector))  # 1536
```

### 5.2 임베딩 저장

**Vector DB에 저장:**

```python
from langchain.vectorstores import Chroma

# ChromaDB 초기화
vectorstore = Chroma(
    collection_name="paper_embeddings",
    embedding_function=embeddings,
    persist_directory="data/vectordb"
)

# 문서 추가
vectorstore.add_documents(chunks)
```

---

## 6. Vector Database 설계

### 6.1 컬렉션 구조

**3개의 컬렉션 운영:**

#### 1. `paper_chunks` (논문 본문)
- **용도**: 논문 전체 내용 검색
- **메타데이터**: paper_id, section, page_num

#### 2. `paper_abstracts` (논문 초록)
- **용도**: 빠른 논문 개요 검색
- **메타데이터**: paper_id, title, authors

#### 3. `glossary_embeddings` (용어집)
- **용도**: 전문 용어 정의 검색
- **메타데이터**: term, category, difficulty_level

### 6.2 컬렉션별 구현

```python
# 1. 논문 본문 컬렉션
paper_chunks_store = Chroma(
    collection_name="paper_chunks",
    embedding_function=embeddings,
    persist_directory="data/vectordb/chunks"
)

# 2. 논문 초록 컬렉션
abstract_store = Chroma(
    collection_name="paper_abstracts",
    embedding_function=embeddings,
    persist_directory="data/vectordb/abstracts"
)

# 3. 용어집 컬렉션
glossary_store = Chroma(
    collection_name="glossary_embeddings",
    embedding_function=embeddings,
    persist_directory="data/vectordb/glossary"
)
```

---

## 7. 용어집(Glossary) 관리 시스템 ★ 중요

### 7.1 용어집의 필요성

논문에는 전문 용어가 많이 등장하며, 초심자는 이해하기 어려움:
- "Attention Mechanism"
- "Fine-tuning"
- "BLEU Score"
- "Backpropagation"

### 7.2 용어집 데이터 구조

**PostgreSQL 테이블:**

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

### 7.3 용어집 RAG 활용 방안

#### 방안 1: 용어집을 Vector DB에 임베딩 저장

**장점:**
- 사용자 질문에 용어가 포함되면 자동으로 검색됨
- 유사한 용어도 함께 찾아줌 (예: "어텐션" → "Attention")

**구현:**
```python
# 용어집 데이터를 Vector DB에 저장
def add_glossary_to_vectordb():
    conn = psycopg2.connect("postgresql://user:password@localhost/papers")
    cursor = conn.cursor()

    cursor.execute("SELECT term, definition, easy_explanation, category FROM glossary")
    glossary_items = cursor.fetchall()

    documents = []
    for term, definition, easy_exp, category in glossary_items:
        doc_content = f"용어: {term}\n정의: {definition}\n쉬운 설명: {easy_exp}"
        documents.append(Document(
            page_content=doc_content,
            metadata={"term": term, "category": category, "type": "glossary"}
        ))

    glossary_store.add_documents(documents)

add_glossary_to_vectordb()
```

#### 방안 2: 질문 분석 시 용어 자동 추출 및 컨텍스트 추가

**구현:**
```python
def extract_and_add_glossary_context(user_query):
    """
    사용자 질문에서 전문 용어를 추출하여 용어집 정의를 프롬프트에 추가
    """
    # 용어집에서 용어 검색
    conn = psycopg2.connect("postgresql://user:password@localhost/papers")
    cursor = conn.cursor()

    # 질문에서 용어 찾기 (간단한 매칭)
    cursor.execute("""
        SELECT term, definition, easy_explanation
        FROM glossary
        WHERE %s ILIKE '%' || term || '%'
    """, (user_query,))

    terms_found = cursor.fetchall()

    if terms_found:
        glossary_context = "\n\n[용어 정의]\n"
        for term, definition, easy_exp in terms_found:
            glossary_context += f"- **{term}**: {easy_exp}\n"

        return glossary_context
    return ""

# 사용 예시
user_query = "Attention Mechanism이 뭐야?"
glossary_context = extract_and_add_glossary_context(user_query)

final_prompt = f"""
{glossary_context}

사용자 질문: {user_query}

답변:
"""
```

#### 방안 3: 하이브리드 검색 (Hybrid Search)

**용어집 + 논문 본문 동시 검색:**

```python
def hybrid_search(query, difficulty="easy"):
    """
    용어집과 논문 본문을 동시에 검색하여 최적의 답변 생성
    """
    # 1. 용어집 검색
    glossary_results = glossary_store.similarity_search(query, k=2)

    # 2. 논문 본문 검색
    paper_results = paper_chunks_store.similarity_search(query, k=3)

    # 3. 결과 결합
    combined_context = "### 용어 정의:\n"
    for doc in glossary_results:
        combined_context += doc.page_content + "\n\n"

    combined_context += "### 논문 내용:\n"
    for doc in paper_results:
        combined_context += doc.page_content + "\n\n"

    # 4. LLM에 전달
    if difficulty == "easy":
        prompt = f"{combined_context}\n\n질문: {query}\n\n초심자도 이해할 수 있도록 쉽게 설명해주세요."
    else:
        prompt = f"{combined_context}\n\n질문: {query}\n\n전문가 수준으로 자세히 설명해주세요."

    return llm.invoke(prompt)
```

### 7.4 용어집 자동 생성

**논문에서 자동으로 용어 추출:**

```python
def auto_generate_glossary_from_papers():
    """
    논문에서 중요 용어를 자동 추출하여 용어집에 추가
    """
    # 1. 논문에서 주요 용어 추출 (NER 또는 LLM 활용)
    extraction_prompt = """
    다음 논문에서 중요한 기술 용어 5개를 추출하고 간단히 정의해주세요:

    논문 내용: {paper_content}

    출력 형식:
    1. 용어: 정의
    2. 용어: 정의
    ...
    """

    # 2. LLM으로 용어 추출
    terms = llm.invoke(extraction_prompt)

    # 3. PostgreSQL 용어집에 추가
    # (중복 체크 후 추가)
```

---

## 8. 검색 (Retrieval) 전략

### 8.1 기본 유사도 검색

```python
# Top-K 검색
results = vectorstore.similarity_search(
    query="Transformer architecture",
    k=5  # 상위 5개 문서 조회
)
```

### 8.2 MMR (Maximal Marginal Relevance) 검색

**목적:** 관련성 높으면서도 다양한 문서 검색

```python
results = vectorstore.max_marginal_relevance_search(
    query="Transformer architecture",
    k=5,
    fetch_k=20,  # 먼저 20개 후보 검색
    lambda_mult=0.5  # 관련성 vs 다양성 균형
)
```

### 8.3 메타데이터 필터링

```python
# 특정 년도 논문만 검색
results = vectorstore.similarity_search(
    query="attention mechanism",
    k=5,
    filter={"year": {"$gte": 2020}}  # 2020년 이후 논문만
)
```

### 8.4 Reranking (재순위화)

**Cohere Rerank API 사용:**

```python
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CohereRerank

# Reranker 설정
compressor = CohereRerank(model="rerank-english-v2.0")

retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=vectorstore.as_retriever(search_kwargs={"k": 10})
)

# 재순위화된 결과
compressed_docs = retriever.get_relevant_documents(
    "Explain transformer architecture"
)
```

---

## 9. LLM 프롬프트 구성

### 9.1 RAG 프롬프트 템플릿

```python
RAG_PROMPT_TEMPLATE = """
당신은 논문 리뷰 전문가입니다.

아래 논문 내용을 참고하여 사용자의 질문에 답변해주세요.

[참고 논문]
{context}

[사용자 질문]
{question}

[답변 규칙]
- 참고 논문의 내용을 기반으로 답변하세요
- 출처를 명시하세요 (논문 제목, 저자)
- 논문에 없는 내용은 추측하지 마세요
- 난이도: {difficulty} 모드

답변:
"""
```

### 9.2 용어집 포함 프롬프트

```python
RAG_WITH_GLOSSARY_PROMPT = """
당신은 논문 리뷰 전문가입니다.

[용어 정의]
{glossary_context}

[참고 논문]
{paper_context}

[사용자 질문]
{question}

답변:
"""
```

---

## 10. RAG 체인 구현

### 10.1 Langchain RAG Chain

```python
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# 프롬프트 템플릿
prompt = PromptTemplate(
    template=RAG_PROMPT_TEMPLATE,
    input_variables=["context", "question", "difficulty"]
)

# RAG 체인 구성
rag_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",  # 모든 문서를 하나의 프롬프트로
    retriever=vectorstore.as_retriever(search_kwargs={"k": 5}),
    chain_type_kwargs={"prompt": prompt}
)

# 실행
response = rag_chain.run(
    query="Transformer 논문 설명해줘",
    difficulty="easy"
)
```

### 10.2 LangGraph를 활용한 복잡한 RAG

```python
from langgraph.graph import StateGraph

class RAGState(TypedDict):
    question: str
    difficulty: str
    glossary_context: str
    paper_context: str
    final_answer: str

def glossary_search_node(state: RAGState):
    """용어집 검색"""
    glossary_docs = glossary_store.similarity_search(state["question"], k=2)
    state["glossary_context"] = "\n".join([doc.page_content for doc in glossary_docs])
    return state

def paper_search_node(state: RAGState):
    """논문 검색"""
    paper_docs = paper_chunks_store.similarity_search(state["question"], k=3)
    state["paper_context"] = "\n".join([doc.page_content for doc in paper_docs])
    return state

def generate_answer_node(state: RAGState):
    """최종 답변 생성"""
    prompt = RAG_WITH_GLOSSARY_PROMPT.format(
        glossary_context=state["glossary_context"],
        paper_context=state["paper_context"],
        question=state["question"]
    )
    state["final_answer"] = llm.invoke(prompt)
    return state

# 그래프 구성
workflow = StateGraph(RAGState)
workflow.add_node("glossary_search", glossary_search_node)
workflow.add_node("paper_search", paper_search_node)
workflow.add_node("generate_answer", generate_answer_node)

workflow.set_entry_point("glossary_search")
workflow.add_edge("glossary_search", "paper_search")
workflow.add_edge("paper_search", "generate_answer")
workflow.add_edge("generate_answer", END)

rag_graph = workflow.compile()
```

---

## 11. 참고 자료

- Langchain RAG 튜토리얼: https://python.langchain.com/docs/tutorials/rag/
- Langchain Vector Stores: https://python.langchain.com/docs/integrations/vectorstores/
- ChromaDB 문서: https://docs.trychroma.com/
- OpenAI Embeddings: https://platform.openai.com/docs/guides/embeddings
- Text Splitters: https://docs.langchain.com/oss/python/integrations/splitters
