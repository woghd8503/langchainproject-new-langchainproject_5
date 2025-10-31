# 박재홍 팀원 코드 문제점 및 TODO

## 문서 정보
- **작성일**: 2025-10-31
- **작성자**: 최현화
- **목적**: 박재홍 팀원이 병합한 코드의 문제점 파악 및 추가 작업 정리

---

## 🚨 발견된 문제점

### 1. collect_arxiv_papers.py 스크립트 누락 (Critical)
**문제**:
- 문서에는 `scripts/collect_arxiv_papers.py`가 명시되어 있음
- `run_full_pipeline.py`에서도 이 스크립트를 실행하려고 시도
- **실제 파일이 Git에 커밋되지 않음**

**증거**:
```python
# run_full_pipeline.py 50번째 줄
steps = [
    ("collect_arxiv_papers.py", "Phase 1: arXiv 논문 수집"),  # 파일 없음!
    ("setup_database.py", "Phase 2: PostgreSQL 데이터베이스 초기화"),
    ("process_documents.py", "Phase 3: PDF 문서 로드 및 청크 분할"),
    ("load_embeddings.py", "Phase 4: 임베딩 생성 및 Vector DB 저장"),
]
```

**영향**:
- 전체 파이프라인 실행 불가
- 논문 수집 자동화 불가
- 새로운 논문 추가 시 수동 작업 필요

**해결 방안**:
- [ ] `scripts/collect_arxiv_papers.py` 스크립트 구현
- [ ] ArxivClient를 래핑하는 실행 스크립트 작성
- [ ] PDF 다운로드 기능 추가
- [ ] 메타데이터 JSON 저장 기능 구현

---

### 2. PDF 파일 누락
**문제**:
- 실험 로그에는 75개 논문 수집 완료 기록됨 (2025-10-30 17:16:21~17:17:22)
- Windows 경로로 작업: `D:\Projects\Langchain\...\data\raw\pdfs\`
- 현재 Linux 환경에는 PDF 파일 없음

**수집 완료 논문**:
```
Collected unique papers: 75
Saved metadata to D:\Projects\Langchain\langchainproject-new-langchainproject_5\data\raw\arxiv_papers_metadata.json
```

**다운로드 실패 논문** (일부):
- 2510.25772v1: Invalid argument (Windows 경로 문제)
- 2510.25770v1, 2510.25765v1, 2510.25760v1, 2510.25409v1

**영향**:
- Document Loader 테스트 불가
- 임베딩 생성 불가
- RAG 시스템 구축 불가

**해결 방안**:
- [ ] 논문 재수집 필요
- [ ] Linux 호환 경로로 수정
- [ ] PDF 다운로드 재시도 로직 추가

---

### 3. 실제 구현과 문서의 불일치
**문제**:
- **문서**: `scripts/collect_arxiv_papers.py` → ArxivPaperCollector 클래스
- **실제**: `src/papers/infra/arxiv_client.py` → ArxivClient 클래스

**차이점**:

| 항목 | 문서 명세 | 실제 구현 |
|------|----------|----------|
| 파일 위치 | scripts/ | src/papers/infra/ |
| 클래스명 | ArxivPaperCollector | ArxivClient |
| PDF 다운로드 | ✅ 구현 명시 | ❌ 미구현 |
| 메타데이터 수집 | ✅ | ✅ |
| 중복 제거 | ✅ 구현 명시 | ❌ 미구현 |
| ExperimentManager 연동 | ✅ 문서에 명시 | ❌ 미구현 |

**문서에서 요구한 기능 (ArxivPaperCollector)**:
```python
class ArxivPaperCollector:
    def __init__(self, save_dir="data/raw/pdfs", exp_manager=None)
    def collect_papers(self, query, max_results=50)
    def collect_by_keywords(self, keywords, per_keyword=15)
    def remove_duplicates(self, papers)
```

**실제 구현 (ArxivClient)**:
```python
class ArxivClient:
    def __init__(self, max_results_default: int = 20)
    def search(self, query: str, max_results: int | None = None) -> List[PaperDTO]
    # PDF 다운로드 기능 없음
    # 중복 제거 기능 없음
    # ExperimentManager 연동 없음
```

**영향**:
- 문서와 코드 불일치로 혼란
- PDF 다운로드 불가
- 중복 논문 관리 불가
- 실험 추적 불가

**해결 방안**:
- [ ] ArxivPaperCollector 래퍼 클래스 구현
- [ ] ArxivClient를 내부적으로 사용
- [ ] PDF 다운로드 기능 추가
- [ ] 중복 제거 로직 추가
- [ ] ExperimentManager 통합

---

### 4. Vector DB 기술 불일치
**문제**:
- **문서**: PostgreSQL + pgvector 사용
- **실제 코드**:
  - `embeddings.py`는 pgvector 사용 ✅
  - `inspect_vector_db.py`는 FAISS 사용 ❌

**inspect_vector_db.py 코드**:
```python
from langchain_community.vectorstores import FAISS
# FAISS 벡터 데이터베이스 열람
FAISS_DIR = ROOT / "data" / "vectordb" / "papers_faiss"
```

**영향**:
- 코드 간 일관성 없음
- FAISS와 pgvector 혼용으로 혼란

**해결 방안**:
- [ ] Vector DB를 pgvector로 통일
- [ ] inspect_vector_db.py를 pgvector용으로 수정
- [ ] 또는 FAISS 검사 스크립트 별도 분리

---

### 5. 데이터 폴더 구조 불완전
**문제**:
현재 상태:
```
data/
└── raw/
    └── .gitkeep
```

필요한 구조 (PRD 02번 참조):
```
data/
├── raw/
│   ├── pdfs/              # ❌ 없음
│   └── arxiv_papers_metadata.json  # ❌ 없음
├── processed/
│   └── paper_id_mapping.json  # ❌ 없음
└── vectordb/              # ❌ 없음
```

**해결 방안**:
- [ ] 폴더 구조 생성
- [ ] .gitkeep 파일 추가

---

### 6. ExperimentManager 미적용
**문제**:
- 문서에는 ExperimentManager 사용이 명시되어 있음
- 실제 코드에는 미적용
- 실험 로그가 수동으로 생성됨 (`20251030_171621_data_collection`)

**문서 명세 (담당역할_03_박재홍_논문데이터수집.md:742-765)**:
```python
with ExperimentManager() as exp:
    collector = ArxivPaperCollector(exp_manager=exp)
    # ...
    exp.logger.write("논문 데이터 수집 시작")
```

**실제 실험 폴더**:
```
experiments/20251030/20251030_171621_data_collection/experiment.log
```
→ ExperimentManager 없이 수동 생성됨

**영향**:
- 실험 추적 불가
- metadata.json 없음
- 세션 ID 자동 부여 없음

**해결 방안**:
- [ ] collect_arxiv_papers.py에 ExperimentManager 통합
- [ ] 모든 스크립트에 ExperimentManager 적용
- [ ] 실험 로그 구조 통일

---

### 7. 환경 변수 설정 미비
**문제**:
필요한 환경 변수:
- `DATABASE_URL` - PostgreSQL 연결 문자열
- `OPENAI_API_KEY` - OpenAI API 키
- `OPENAI_EMBEDDING_MODEL` - 임베딩 모델 (기본값: text-embedding-3-small)

현재 상태:
- `.env` 파일 확인 필요
- 환경 변수 설정 가이드 없음

**해결 방안**:
- [ ] `.env.example` 파일 생성
- [ ] README에 환경 변수 설정 가이드 추가

---

### 8. 테스트 파일 위치 이동
**문제**:
- 원래 위치: `scripts/test_pipeline.py`
- 이동 위치: `tests/test_pipeline.py` ✅

**영향**:
- 테스트 실행 명령어 변경 필요

**해결 방안**:
- [x] 파일 이동 완료
- [ ] README 업데이트 (테스트 실행 명령어 수정)

---

## 📋 추가 작업 필요 항목 (PRD 분석 결과)

### PRD 분석 완료 ✅
- **PRD 11번 (데이터베이스 설계)**: papers, glossary, query_logs 테이블 스키마 확인
- **PRD 02번 (프로젝트 구조)**: 폴더 구조 및 scripts 요구사항 확인
- **PRD 13번 (RAG 시스템)**: 청크 크기, 임베딩 모델, pgvector 컬렉션 확인

### Phase 1: 데이터 수집 스크립트 구현 (Critical - P0)
- [ ] `scripts/collect_arxiv_papers.py` 작성
  - [ ] ArxivClient를 래핑하는 ArxivPaperCollector 클래스
  - [ ] PDF 다운로드 기능 (arxiv.Result.download_pdf)
    - Linux 경로 호환성 확보
    - 다운로드 실패 재시도 로직
  - [ ] 메타데이터 JSON 저장 (`data/raw/json/arxiv_papers_metadata.json`)
  - [ ] 중복 제거 (제목 기준)
  - [ ] ExperimentManager 통합
  - [ ] 키워드 리스트: transformer attention, BERT GPT, LLM, RAG, NMT, QA, AI agent
  - [ ] 키워드당 15편씩 수집 (총 ~100편 목표)
  - [ ] 진행률 표시 (tqdm 통합)

### Phase 2: Document Loader 검증
- [ ] `src/data/document_loader.py` 테스트
  - [x] 클래스 구현 확인
  - [ ] PDF 로드 테스트
  - [ ] 청크 분할 테스트
  - [ ] 메타데이터 매핑 테스트

### Phase 3: 임베딩 및 Vector DB 검증
- [ ] `src/data/embeddings.py` 테스트
  - [x] 클래스 구현 확인
  - [ ] OpenAI Embeddings 연결 테스트
  - [ ] PGVector 연결 테스트
  - [ ] 배치 처리 테스트
  - [ ] Rate Limit 대응 확인

### Phase 4: 데이터베이스 초기화
- [ ] `scripts/setup_database.py` 실행
  - [x] 스크립트 구현 확인
  - [ ] PostgreSQL 연결 테스트
  - [ ] pgvector extension 설치
  - [ ] papers 테이블 생성
  - [ ] glossary 테이블 생성
  - [ ] 인덱스 생성
  - [ ] 용어집 초기 데이터 삽입

### Phase 5: 전체 파이프라인 실행
- [ ] `scripts/run_full_pipeline.py` 수정
  - [ ] collect_arxiv_papers.py 호출 부분 수정
  - [ ] 각 단계별 오류 처리
  - [ ] 진행 상황 로깅

### Phase 6: 검증 및 테스트
- [ ] `tests/test_pipeline.py` 실행
  - [ ] ArxivPaperCollector 테스트
  - [ ] PaperDocumentLoader 테스트
  - [ ] PaperEmbeddingManager 테스트
  - [ ] 전체 통합 테스트

---

## 🔍 PRD 문서 분석 필요

다음 문서들을 분석하여 추가 요구사항 확인:

### 필수 분석 문서
1. **01_프로젝트_개요.md** - 전체 프로젝트 목표 확인
2. **02_프로젝트_구조.md** - 폴더 구조 확인
3. **11_데이터베이스_설계.md** ⭐ - DB 스키마 상세 확인
4. **13_RAG_시스템_설계.md** - Document 처리 요구사항 확인

### 선택 분석 문서
5. **05_로깅_시스템.md** - Logger 사용법
6. **06_실험_추적_관리.md** - ExperimentManager 요구사항
7. **10_기술_요구사항.md** - arXiv API, PyPDFLoader, OpenAI Embeddings

---

## 📊 우선순위

### P0 (Critical - 즉시 해결 필요)
1. `scripts/collect_arxiv_papers.py` 구현
2. 데이터 폴더 구조 생성
3. `.env.example` 파일 생성

### P1 (High - 이번 주 내 해결)
4. PRD 11번 문서 분석 및 DB 스키마 검증
5. 논문 재수집 (75~100편)
6. ExperimentManager 통합

### P2 (Medium - 다음 주)
7. Vector DB 통일 (pgvector)
8. 전체 파이프라인 테스트
9. README 업데이트

### P3 (Low - 여유 있을 때)
10. 코드 리팩토링
11. 문서 업데이트

---

---

## 📊 PRD vs 실제 구현 비교

### 데이터베이스 스키마 (PRD 11번)

| 항목 | PRD 요구사항 | 박재홍 구현 | 상태 |
|------|-------------|-----------|------|
| **papers 테이블** | | | |
| - arxiv_id | VARCHAR(64) | ✅ | ✅ |
| - updated_at | TIMESTAMP DEFAULT NOW() | ❌ 없음 | ⚠️ 추가 필요 |
| **glossary 테이블** | | | |
| - updated_at | TIMESTAMP DEFAULT NOW() | ❌ 없음 | ⚠️ 추가 필요 |
| - related_terms | TEXT[] (배열) | TEXT | ⚠️ 타입 불일치 |
| - difficulty_level | VARCHAR(20) | INT | ⚠️ 타입 불일치 |
| **pgvector 컬렉션** | | | |
| - paper_chunks | ✅ 명시 | ✅ 구현 | ✅ |
| - paper_abstracts | ✅ 명시 | ❌ 없음 | ⚠️ 추가 필요 |
| - glossary_embeddings | ✅ 명시 | ❌ 없음 | ⚠️ 추가 필요 |

### 폴더 구조 (PRD 02번)

| 폴더 | PRD 요구사항 | 현재 상태 | 상태 |
|------|-------------|----------|------|
| data/raw/pdfs/ | ✅ | ❌ 없음 | ⚠️ 생성 필요 |
| data/raw/json/ | ✅ | ❌ 없음 | ⚠️ 생성 필요 |
| data/raw/txt/ | ✅ | ❌ 없음 | ⚠️ 생성 필요 |
| data/processed/chunks/ | ✅ | ❌ 없음 | ⚠️ 생성 필요 |
| data/processed/embeddings/ | ✅ | ❌ 없음 | ⚠️ 생성 필요 |
| data/outputs/conversations/ | ✅ | ❌ 없음 | ⚠️ 생성 필요 |
| data/outputs/summaries/ | ✅ | ❌ 없음 | ⚠️ 생성 필요 |
| scripts/collect_papers.py | ✅ | ❌ 없음 | ⚠️ 구현 필요 |
| scripts/build_vectordb.py | ✅ | ❌ 없음 | ⚠️ 구현 필요 |

### RAG 설정 (PRD 13번)

| 항목 | PRD 요구사항 | 박재홍 구현 | 상태 |
|------|-------------|-----------|------|
| chunk_size | 1000 | 1000 | ✅ |
| chunk_overlap | 200 | 200 | ✅ |
| separators | ["\n\n", "\n", ". ", " ", ""] | ✅ | ✅ |
| embedding_model | text-embedding-3-small | ✅ | ✅ |
| dimension | 1536 | - | ⚠️ 확인 필요 |

---

## 🔧 즉시 수정 필요 항목

### 1. 데이터베이스 스키마 수정
```sql
-- papers 테이블에 updated_at 추가
ALTER TABLE papers ADD COLUMN IF NOT EXISTS updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP;

-- glossary 테이블 수정
ALTER TABLE glossary ADD COLUMN IF NOT EXISTS updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP;
ALTER TABLE glossary ALTER COLUMN difficulty_level TYPE VARCHAR(20);
-- related_terms는 이미 TEXT로 정의됨 (PRD는 TEXT[] 배열이지만 TEXT로 사용 가능)
```

### 2. 추가 pgvector 컬렉션 생성
```python
# paper_abstracts 컬렉션
abstract_store = PGVector(
    collection_name="paper_abstracts",
    embedding_function=embeddings,
    connection_string=CONNECTION_STRING
)

# glossary_embeddings 컬렉션
glossary_store = PGVector(
    collection_name="glossary_embeddings",
    embedding_function=embeddings,
    connection_string=CONNECTION_STRING
)
```

### 3. 폴더 구조 생성
```bash
mkdir -p data/raw/{pdfs,json,txt}
mkdir -p data/processed/{chunks,embeddings}
mkdir -p data/outputs/{conversations,summaries}
```

---

## 📅 작업 이력
- 2025-10-31 19:30: 초안 작성 (문제점 발견 및 정리)
- 2025-10-31 20:15: PRD 01~16번 문서 분석 완료
- 2025-10-31 20:30: PRD vs 실제 구현 비교표 추가
