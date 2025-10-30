# 13. RAG 시스템 설계

## 문서 정보
- **작성일**: 2025-10-30
- **프로젝트명**: 논문 리뷰 챗봇 (AI Agent + RAG)
- **팀명**: 연결의 민족

---

## 1. RAG 파이프라인

### 1.1 전체 흐름

```mermaid
graph TB
    subgraph Prepare["🔸 데이터 준비"]
        direction LR
        A[논문 수집<br/>arXiv] --> B[텍스트 추출<br/>PDF → Text]
        B --> C[청크 분할<br/>1000자 단위]
        C --> D[임베딩<br/>OpenAI]
        D --> E[💾 VectorDB<br/>pgvector]
    end

    subgraph Search["🔹 검색"]
        direction LR
        F[사용자 질문] --> G[질문 임베딩]
        G --> H[유사도 검색<br/>Top-K]
        H --> I[관련 문서]
    end

    subgraph Generate["🔺 답변 생성"]
        direction LR
        J[컨텍스트 구성] --> K[LLM 생성<br/>GPT-4]
        K --> L[✅ 답변]
    end

    Prepare --> Search
    Search --> Generate
    E -.-> H
    I --> J

    %% Subgraph 스타일
    style Prepare fill:#e1f5ff,stroke:#01579b,stroke-width:3px,color:#000
    style Search fill:#f3e5f5,stroke:#4a148c,stroke-width:3px,color:#000
    style Generate fill:#e8f5e9,stroke:#1b5e20,stroke-width:3px,color:#000
```

---

## 2. 문서 처리

### 2.1 텍스트 분할

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    separators=["\n\n", "\n", ". ", " ", ""]
)

chunks = text_splitter.split_documents(documents)
```

### 2.2 임베딩 생성

```python
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small"
)

vectors = embeddings.embed_documents([chunk.page_content for chunk in chunks])
```

---

## 3. Vector Store

### 3.1 pgvector 연동

```python
from langchain_postgres.vectorstores import PGVector

vectorstore = PGVector(
    collection_name="paper_chunks",
    embedding_function=embeddings,
    connection_string="postgresql://user:password@localhost:5432/papers"
)

# 문서 추가
vectorstore.add_documents(chunks)
```

### 3.2 검색

```python
# Similarity Search
docs = vectorstore.similarity_search(query, k=5)

# MMR (Maximal Marginal Relevance)
docs = vectorstore.max_marginal_relevance_search(query, k=5, fetch_k=20)
```

---

## 4. Retriever

### 4.1 기본 Retriever

```python
retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 5, "fetch_k": 20}
)

docs = retriever.invoke(query)
```

### 4.2 MultiQueryRetriever

```python
from langchain.retrievers import MultiQueryRetriever

retriever = MultiQueryRetriever.from_llm(
    retriever=vectorstore.as_retriever(),
    llm=llm
)
```

---

## 5. 용어집 통합

### 5.1 용어집 검색

```python
# 용어집 전용 Vector Store
glossary_store = PGVector(
    collection_name="glossary_embeddings",
    embedding_function=embeddings,
    connection_string=CONNECTION_STRING
)

# 용어 검색
glossary_docs = glossary_store.similarity_search(term, k=3)
```

### 5.2 하이브리드 검색

```python
def hybrid_search(query, difficulty="easy"):
    """용어집 + 논문 본문 동시 검색"""
    # 용어집 검색
    glossary_results = glossary_store.similarity_search(query, k=2)
    
    # 논문 검색
    paper_results = vectorstore.similarity_search(query, k=3)
    
    # 결합
    combined_context = "### 용어:\n" + glossary_results + "\n### 논문:\n" + paper_results
    
    return combined_context
```

---

## 6. 프롬프트 구성

```python
RAG_PROMPT = """
당신은 논문 리뷰 전문가입니다.

[참고 논문]
{context}

[질문]
{question}

[답변 규칙]
- 참고 논문의 내용을 기반으로 답변
- 출처를 명시 (논문 제목, 저자)
- 논문에 없는 내용은 추측하지 말 것
- 난이도: {difficulty}

답변:
"""
```

---

## 7. 참고 자료

- Langchain RAG: https://python.langchain.com/docs/tutorials/rag/
- Langchain VectorStores: https://python.langchain.com/docs/integrations/vectorstores/
