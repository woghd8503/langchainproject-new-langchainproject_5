# 담당역할: 신준엽 - RAG 검색 도구 & 용어집 도구

## 담당자 정보
- **이름**: 신준엽
- **역할**: RAG 시스템 전문 담당
- **참여 기간**: 전체 기간
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

**파일 경로**: `src/rag/retriever.py`

**구현 방법**:
1. `RAGRetriever` 클래스 정의 (논문 검색을 위한 RAG Retriever)
2. OpenAI Embeddings 초기화
   - 모델: `text-embedding-3-small`
   - API 키를 환경변수에서 로드
3. PostgreSQL + pgvector VectorStore 초기화
   - 컬렉션명: `paper_chunks`
   - 임베딩 함수 설정
   - PostgreSQL 연결 문자열 설정
4. 기본 Retriever 설정 (MMR 방식)
   - 검색 타입: MMR (Maximal Marginal Relevance)
   - 최종 반환 문서 수: 5개
   - MMR 후보 문서 수: 20개
   - lambda_mult: 0.5 (관련성 vs 다양성 균형)
5. MultiQuery Retriever 구현 (쿼리 확장)
   - LLM을 사용하여 기본 Retriever에서 쿼리를 확장
6. `retrieve` 메서드 구현
   - 쿼리 확장 사용 여부에 따라 적절한 Retriever 선택
   - 문서 검색 후 반환
7. `retrieve_with_filter` 메서드 구현
   - 메타데이터 필터링을 포함한 검색
   - 예: 년도 필터 {"year": {"$gte": 2020}}
8. `retrieve_with_scores` 메서드 구현
   - 유사도 점수를 포함한 검색 결과 반환

#### 2. RAG 검색 도구 구현

**파일 경로**: `src/tools/rag_search.py`

**구현 방법**:
1. Langchain `@tool` 데코레이터로 `search_paper_database` 함수 정의
   - 인자: query (검색 질문), year_filter (선택적 년도 필터)
   - 반환: 관련 논문 내용 및 메타데이터
2. Vector DB에서 유사도 검색 수행
   - year_filter가 있으면 필터링 검색 실행
   - year_filter가 없으면 MultiQuery 검색 실행
3. PostgreSQL 연결 및 메타데이터 조회
   - psycopg2로 PostgreSQL 연결
   - 각 검색된 문서의 paper_id로 메타데이터 조회
   - 제목, 저자, 출판일, URL 등을 가져옴
4. 검색 결과를 딕셔너리 리스트로 구성
   - title, authors, publish_date, url, content, section 포함
5. `format_search_results` 함수 구현
   - 검색 결과를 LLM에 전달할 수 있는 Markdown 형식으로 포맷팅
   - 논문별로 제목, 저자, 출판일, URL, 섹션, 내용을 포함
   - 결과가 없으면 "관련 논문을 찾을 수 없습니다." 반환

#### 3. MultiQueryRetriever (쿼리 확장)

**동작 방식**:
- 원본 쿼리를 LLM이 자동으로 3-5개의 변형 쿼리로 확장
- 예: "Transformer 논문 설명해줘" → "Transformer 아키텍처란?", "Attention Is All You Need 논문 내용", "Transformer 모델의 핵심 메커니즘"
- 각 쿼리로 검색 수행
- 결과 통합 및 중복 제거
- 최종 결과 반환

#### 4. ContextualCompressionRetriever (선택 사항)

**파일 경로**: `src/rag/compression.py`

**구현 방법**:
1. `create_compression_retriever` 함수 정의
   - 인자: base_retriever, llm
2. LLMChainExtractor를 사용하여 compressor 생성
3. ContextualCompressionRetriever 생성
   - base_compressor와 base_retriever 설정
4. 긴 문서를 검색 후, 질문과 관련된 부분만 추출하여 컨텍스트 크기 감소

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

**파일 경로**: `src/rag/glossary_retriever.py`

**구현 방법**:
1. `GlossaryRetriever` 클래스 정의 (용어집 검색을 위한 Retriever)
2. OpenAI Embeddings 초기화
   - 모델: `text-embedding-3-small`
3. 용어집 전용 VectorStore 초기화 (pgvector)
   - 컬렉션명: `glossary_embeddings`
   - 임베딩 함수 설정
   - PostgreSQL 연결 문자열 설정
4. Retriever 설정
   - 검색 타입: similarity
   - 반환 문서 수: 3개
5. `search` 메서드 구현
   - 용어를 받아서 Retriever로 검색
   - 검색 결과 반환

#### 2. 용어집 검색 도구 구현

**파일 경로**: `src/tools/glossary.py`

**구현 방법**:
1. Langchain `@tool` 데코레이터로 `search_glossary` 함수 정의
   - 인자: term (검색할 용어), difficulty (난이도: easy/hard)
   - 반환: 용어 정의 및 설명
2. PostgreSQL glossary 테이블에서 직접 검색 (1차 검색)
   - psycopg2로 PostgreSQL 연결
   - ILIKE를 사용한 유연한 검색
   - term, definition, easy_explanation, hard_explanation, category 조회
3. PostgreSQL에서 결과를 찾은 경우
   - 난이도에 따라 easy_explanation 또는 hard_explanation 선택
   - Markdown 형식으로 포맷팅하여 반환
4. PostgreSQL에 없으면 Vector DB에서 검색 (2차 검색)
   - glossary_retriever.search()를 사용하여 유사 용어 검색
   - 가장 유사한 문서의 내용 반환
5. Vector DB에도 없으면 논문 본문에서 검색 (3차 검색)
   - RAG Retriever로 논문에서 용어 정의 검색
   - 검색된 문서의 내용 반환
6. 모든 검색에 실패하면 "정보를 찾을 수 없습니다" 메시지 반환

#### 3. 하이브리드 검색 (PostgreSQL + Vector DB)

**구현 방법**:
1. `hybrid_glossary_search` 함수 정의
   - PostgreSQL과 Vector DB를 동시에 검색하여 최상의 결과 반환
2. PostgreSQL 검색 수행
   - ILIKE를 사용하여 용어 검색
   - 결과를 results["postgres"]에 저장
3. Vector DB 검색 수행
   - glossary_retriever.search()로 검색
   - 결과를 results["vector_db"]에 저장
4. 결과 통합 및 우선순위 처리
   - PostgreSQL 결과가 있으면 우선 반환 (정확도 높음)
   - PostgreSQL 결과가 없으면 Vector DB 결과 반환 (유연성 높음)
   - 둘 다 없으면 "정보를 찾을 수 없습니다" 반환

#### 4. 질문 분석 시 용어 자동 추출 및 컨텍스트 추가

**파일 경로**: `src/rag/context_enhancer.py`

**구현 방법**:
1. `extract_and_add_glossary_context` 함수 정의
   - 사용자 질문에서 전문 용어를 추출하여 프롬프트에 추가
2. PostgreSQL 연결 및 용어 검색
   - ILIKE를 사용하여 질문 내에 포함된 용어 찾기
   - term, definition, easy_explanation 조회
3. 검색된 용어가 있으면 컨텍스트 생성
   - 각 용어에 대해 이름과 설명을 Markdown 형식으로 포맷팅
   - "관련 용어 정의" 섹션으로 구성
4. 검색된 용어가 없으면 빈 문자열 반환

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

### Phase 1: RAG 시스템 기초 구현
- PostgreSQL + pgvector VectorStore 연동
- OpenAI Embeddings 초기화
- 기본 Retriever 구현 (similarity)
- search_paper_database 도구 기본 구현

### Phase 2: 고급 검색 기능 구현
- MultiQueryRetriever 구현 (쿼리 확장)
- MMR 검색 방식 적용
- 메타데이터 필터링
- 유사도 점수 반환

### Phase 3: 용어집 시스템 구현
- 용어집 전용 VectorStore 초기화
- search_glossary 도구 구현
- PostgreSQL glossary 테이블 연동
- 하이브리드 검색 구현

### Phase 4: 통합 및 최적화
- ContextualCompressionRetriever 구현 (선택)
- 검색 결과 포맷팅 개선
- PostgreSQL 연동 최적화
- 단위 테스트

---

## RAG 노드 구현 (LangGraph 통합)

**파일 경로**: `src/agent/nodes.py`

### 1. RAG 검색 노드

**구현 방법**:
1. `search_paper_node` 함수 정의
   - 인자: state (AgentState)
2. state에서 질문 추출
3. RAG 검색 도구 호출
   - search_paper_database.invoke()로 검색 수행
   - query와 year_filter 전달
4. 검색 결과를 state["tool_result"]에 저장
5. 난이도 정보 추출
6. 프롬프트 구성
   - 검색 결과, 사용자 질문, 난이도 포함
7. LLM 호출하여 최종 답변 생성
   - SystemMessage: "당신은 논문 리뷰 전문가입니다."
   - HumanMessage: 프롬프트
8. 최종 답변을 state["final_answer"]에 저장 후 반환

### 2. 용어집 검색 노드

**구현 방법**:
1. `glossary_node` 함수 정의
   - 인자: state (AgentState)
2. state에서 질문과 난이도 추출
3. 질문에서 용어 추출
   - "이 뭐야?", "란?" 등의 패턴 제거
4. 용어집 검색 도구 호출
   - search_glossary.invoke()로 검색 수행
   - term과 difficulty 전달
5. 검색 결과를 state["final_answer"]에 저장 후 반환

---

## Feature 브랜치

- `feature/rag-system` - RAG 시스템 기초 구현
- `feature/tool-rag-search` - RAG 검색 도구
- `feature/tool-glossary` - 용어집 도구
- `feature/rag-optimization` - 검색 최적화 (MultiQuery, MMR)

---

## 테스트 코드

**파일 경로**: `tests/test_rag.py`

### 테스트 항목

1. **test_rag_retriever**: RAG Retriever 테스트
   - RAGRetriever 인스턴스 생성
   - 기본 검색 테스트 (query: "Transformer architecture")
   - 검색 결과가 존재하는지 확인
   - 필터링 검색 테스트 (query: "BERT", filter: year >= 2018)
   - 필터링된 검색 결과가 존재하는지 확인

2. **test_search_paper_database**: RAG 검색 도구 테스트
   - search_paper_database.invoke() 호출 (query: "Attention mechanism")
   - 반환 결과에 "검색된 논문" 텍스트가 포함되어 있는지 확인
   - 결과가 비어있지 않은지 확인

---

## 참고 자료

- Langchain Retrieval: https://python.langchain.com/docs/tutorials/rag/
- Langchain Vector Stores: https://python.langchain.com/docs/integrations/vectorstores/
- Langchain PGVector: https://python.langchain.com/docs/integrations/vectorstores/pgvector/
- Langchain Retrievers: https://python.langchain.com/docs/modules/data_connection/retrievers/
- MultiQueryRetriever: https://python.langchain.com/docs/modules/data_connection/retrievers/multi_query/
- ContextualCompressionRetriever: https://python.langchain.com/docs/modules/data_connection/retrievers/contextual_compression/
