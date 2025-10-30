## 제목 : RAG 검색 시스템 및 용어집 도구 구현

---

## 📋 작업 개요
**작업 주제:** RAG 파이프라인 구축 및 용어집 하이브리드 검색 시스템 개발
**담당자:** @신준엽
**마감일:** 11/03 24:00

## 📅 기간
- 시작일: 2025-10-28
- 종료일: 2025-11-03

---

## 📌 이슈 목적

Langchain과 PostgreSQL + pgvector를 사용하여 논문 검색을 위한 RAG(Retrieval-Augmented Generation) 시스템을 구축합니다. VectorStoreRetriever, MultiQueryRetriever를 구현하고, AI/ML 용어를 설명하는 용어집 도구를 개발합니다.

**핵심 목표:**
- PostgreSQL + pgvector VectorStore 연동
- RAG Retriever 구현 (MMR, MultiQuery)
- RAG 검색 도구 개발 (search_paper_database)
- 용어집 하이브리드 검색 시스템 (PostgreSQL + Vector DB)
- 용어집 도구 개발 (search_glossary)

---

## ✅ 작업 항목 체크리스트

### Phase 1: RAG 시스템 기초 구현 (2일)
- [ ] RAGRetriever 클래스 구현 (`src/rag/retriever.py`)
  - [ ] OpenAI Embeddings 초기화 (text-embedding-3-small)
  - [ ] PGVector VectorStore 연동 (collection: paper_chunks)
  - [ ] 기본 Retriever 설정 (MMR 방식, k=5, fetch_k=20, lambda_mult=0.5)
  - [ ] retrieve() 메서드 구현
  - [ ] retrieve_with_filter() 메서드 구현 (년도, 카테고리 필터링)
  - [ ] retrieve_with_scores() 메서드 구현 (유사도 점수 포함)
- [ ] PostgreSQL 연결 및 메타데이터 조회 로직

### Phase 2: 고급 검색 기능 구현 (2일)
- [ ] MultiQueryRetriever 구현
  - [ ] LLM으로 쿼리 확장 (1개 → 3-5개 변형 쿼리)
  - [ ] 각 쿼리로 검색 후 결과 통합
  - [ ] 중복 제거 및 최종 결과 반환
- [ ] RAG 검색 도구 구현 (`src/tools/rag_search.py`)
  - [ ] @tool 데코레이터로 search_paper_database 함수 정의
  - [ ] year_filter 파라미터 지원
  - [ ] PostgreSQL에서 메타데이터 조회 (title, authors, url)
  - [ ] format_search_results() 함수 구현 (Markdown 형식)
- [ ] RAG 노드 구현 (search_paper_node, src/agent/nodes.py)

### Phase 3: 용어집 시스템 구현 (2일)
- [ ] GlossaryRetriever 클래스 구현 (`src/rag/glossary_retriever.py`)
  - [ ] 용어집 전용 VectorStore 초기화 (collection: glossary_embeddings)
  - [ ] Retriever 설정 (similarity, k=3)
  - [ ] search() 메서드 구현
- [ ] 용어집 검색 도구 구현 (`src/tools/glossary.py`)
  - [ ] @tool 데코레이터로 search_glossary 함수 정의
  - [ ] PostgreSQL glossary 테이블에서 1차 검색 (ILIKE)
  - [ ] 난이도별 설명 반환 (easy_explanation / hard_explanation)
  - [ ] Vector DB에서 2차 검색 (유사 용어)
  - [ ] 하이브리드 검색 결과 포맷팅
- [ ] 용어집 노드 구현 (glossary_node, src/agent/nodes.py)

### Phase 4: 통합 및 최적화 (1일)
- [ ] ContextualCompressionRetriever 구현 (선택 사항)
  - [ ] LLMChainExtractor로 문서 압축
  - [ ] 질문 관련 부분만 추출하여 컨텍스트 축소
- [ ] 검색 결과 포맷팅 개선
- [ ] PostgreSQL 연동 최적화 (연결 풀링)
- [ ] 단위 테스트 작성 (`tests/test_rag.py`)
  - [ ] RAG Retriever 테스트
  - [ ] search_paper_database 도구 테스트
  - [ ] Glossary 검색 테스트

### Phase 5: 로깅 및 문서화 (1일)
- [ ] Logger 클래스 적용
  - [ ] 실험 폴더 생성 (experiments/날짜/날짜_시간_rag_search/)
  - [ ] 검색 결과 로깅 (쿼리, 결과 개수, 유사도 점수)
  - [ ] config.yaml, results.json 저장
- [ ] 코드 주석 작성
- [ ] 사용 예시 문서 작성

---

## 📦 설치/실행 명령어 예시

```bash
# 가상환경 활성화
source .venv/bin/activate

# 필요한 패키지 설치
pip install langchain langchain-openai langchain-postgres pgvector psycopg2-binary

# PostgreSQL + pgvector 연결 테스트
python -c "from langchain_postgres.vectorstores import PGVector; print('pgvector 연결 성공')"

# RAG Retriever 테스트
python src/rag/retriever.py

# 도구 테스트
python -c "from src.tools.rag_search import search_paper_database; print(search_paper_database.invoke({'query': 'Transformer'}))"

# 단위 테스트 실행
pytest tests/test_rag.py -v
```

---

### ⚡️ 참고

**중요 사항:**
1. **pgvector 설정**: PostgreSQL에 pgvector extension 설치 필수
2. **MMR 검색**: 관련성(relevance)과 다양성(diversity) 균형 (lambda_mult=0.5)
3. **MultiQuery**: 쿼리 확장으로 검색 recall 향상
4. **하이브리드 검색**: PostgreSQL(정확도) + Vector DB(유사도) 조합
5. **난이도 모드**: Easy/Hard에 맞는 용어 설명 제공

**PostgreSQL 테이블:**
- `papers`: 논문 메타데이터 (paper_id, title, authors, url, publish_date)
- `glossary`: 용어집 (term, definition, easy_explanation, hard_explanation)
- `paper_chunks` (pgvector): 논문 청크 임베딩
- `glossary_embeddings` (pgvector): 용어집 임베딩

**검색 방식:**
- Cosine Similarity (기본)
- MMR (Maximal Marginal Relevance)
- MultiQuery (쿼리 확장)

---

### 유용한 링크

**필수 참고 PRD 문서:**
- [docs/PRD/01_프로젝트_개요.md](../PRD/01_프로젝트_개요.md) - 프로젝트 전체 개요
- [docs/PRD/02_프로젝트_구조.md](../PRD/02_프로젝트_구조.md) - 폴더 구조
- [docs/PRD/05_로깅_시스템.md](../PRD/05_로깅_시스템.md) ⭐ - Logger 사용법
- [docs/PRD/06_실험_추적_관리.md](../PRD/06_실험_추적_관리.md) ⭐ - 실험 폴더 구조
- [docs/PRD/10_기술_요구사항.md](../PRD/10_기술_요구사항.md) - 기술 스택
- [docs/PRD/11_데이터베이스_설계.md](../PRD/11_데이터베이스_설계.md) - DB 스키마 (papers, glossary 테이블)
- [docs/PRD/13_RAG_시스템_설계.md](../PRD/13_RAG_시스템_설계.md) - RAG 파이프라인 및 Retriever 설정

**참고 PRD 문서:**
- [docs/PRD/03_브랜치_전략.md](../PRD/03_브랜치_전략.md) - Feature 브랜치
- [docs/PRD/04_일정_관리.md](../PRD/04_일정_관리.md) - 개발 일정

**외부 링크:**
- [Langchain Retrieval](https://python.langchain.com/docs/tutorials/rag/)
- [Langchain PGVector](https://python.langchain.com/docs/integrations/vectorstores/pgvector/)
- [MultiQueryRetriever](https://python.langchain.com/docs/modules/data_connection/retrievers/multi_query/)

## 🔖 추천 라벨

`feature` `rag` `tool` `vectordb` `embedding` `high`

---
