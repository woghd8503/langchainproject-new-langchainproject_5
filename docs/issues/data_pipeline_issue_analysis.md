# 데이터 파이프라인 이슈 분석 및 현황

**작성일**: 2025-10-31  
**분석자**: 박재홍  
**참고 문서**: `docs/todo/박재홍_코드_문제점_및_TODO.md`

---

## 🔍 실제 구현 현황 확인

### 1. collect_arxiv_papers.py 스크립트 ✅ (부분 해결)

**현재 상태:**
- ❌ `scripts/collect_arxiv_papers.py` 파일 없음
- ✅ `src/papers/infra/arxiv_client.py`에 `ArxivClient` 클래스 구현됨
- ✅ PDF 파일 75개 존재 (`data/raw/pdfs/`)
- ❌ `data/raw/arxiv_papers_metadata.json` 파일 없음

**실제 구현:**
- `ArxivClient`는 메타데이터 검색 기능만 제공
- PDF 다운로드 기능 없음
- 중복 제거 기능 없음
- ExperimentManager 연동 없음

**해결 필요:**
- `scripts/data/collect_arxiv_papers.py` 구현 필요
- `ArxivClient`를 래핑하여 PDF 다운로드 추가
- 메타데이터 JSON 저장 기능 추가

---

### 2. PDF 파일 현황 ✅ (일부 해결)

**현재 상태:**
- ✅ PDF 파일 75개 존재 확인 (`data/raw/pdfs/`)
- ❌ 메타데이터 JSON 파일 없음
- ❌ Linux 환경에서의 경로 문제 가능성

**확인된 파일:**
```
data/raw/pdfs/
├── 2510.24435v1.pdf
├── 2510.24438v1.pdf
├── ...
└── 2510.25772v1.pdf (75개 파일)
```

**문제점:**
- 메타데이터 JSON 파일 누락으로 재구성 불가
- PDF는 있으나 메타데이터 정보 없음

---

### 3. ArxivPaperCollector vs ArxivClient 불일치 ✅ (확인됨)

**문서 요구사항:**
- `scripts/collect_arxiv_papers.py`에 `ArxivPaperCollector` 클래스
- PDF 다운로드, 중복 제거, ExperimentManager 연동

**실제 구현:**
- `src/papers/infra/arxiv_client.py`에 `ArxivClient` 클래스
- 메타데이터 검색만 제공 (DTO 반환)
- PDF 다운로드 없음
- 중복 제거 없음
- ExperimentManager 연동 없음

**해결 방안:**
- `ArxivClient`를 내부적으로 사용하는 `ArxivPaperCollector` 래퍼 클래스 구현
- 또는 `ArxivClient`에 필요한 기능 추가

---

### 4. Vector DB 기술 불일치 ✅ (확인됨)

**현재 상태:**
- ✅ `src/data/embeddings.py`: PGVector 사용
- ❌ `scripts/data/inspect_vector_db.py`: FAISS 사용
- ❌ FAISS 인덱스 파일 존재 (`data/vectordb/papers_faiss/`)

**문제점:**
- 두 가지 Vector DB 기술 혼용
- FAISS는 로컬 파일 시스템 사용
- PGVector는 PostgreSQL 사용

**해결 방안:**
- `inspect_vector_db.py`를 PGVector용으로 수정
- 또는 FAISS 검사 스크립트 별도 분리
- 문서화: FAISS는 예전 버전, PGVector가 최신

---

### 5. 데이터 폴더 구조 ✅ (일부 존재)

**현재 상태:**
```
data/
├── raw/
│   └── pdfs/          ✅ 75개 PDF 존재
├── processed/         ✅ 폴더 존재
└── vectordb/         ✅ FAISS 인덱스 존재
    └── papers_faiss/
```

**누락된 항목:**
- `data/raw/arxiv_papers_metadata.json` ❌
- `data/processed/paper_id_mapping.json` ❌
- `data/raw/json/` 폴더 ❌
- `data/raw/txt/` 폴더 ❌
- `data/processed/chunks/` 폴더 ❌
- `data/processed/embeddings/` 폴더 ❌
- `data/outputs/` 폴더 ❌

---

### 6. ExperimentManager 미적용 ✅ (확인됨)

**현재 상태:**
- ❌ `scripts/data/` 폴더의 스크립트들에 ExperimentManager 미적용
- ✅ 실험 로그 폴더는 수동으로 생성됨 (`experiments/20251030/`)

**문서 요구사항:**
```python
with ExperimentManager() as exp:
    collector = ArxivPaperCollector(exp_manager=exp)
    exp.logger.write("논문 데이터 수집 시작")
```

**실제 구현:**
- ExperimentManager 사용 없음
- 수동 로깅 또는 Logger만 사용

---

### 7. 환경 변수 설정 ✅ (확인 필요)

**필요한 환경 변수:**
- `DATABASE_URL`: PostgreSQL 연결
- `OPENAI_API_KEY`: OpenAI API 키
- `OPENAI_EMBEDDING_MODEL`: 임베딩 모델 (선택)

**현재 상태:**
- `.env` 파일 존재 여부 확인 필요
- `.env.example` 파일 없음

---

### 8. 스크립트 위치 변경 ✅ (확인됨)

**현재 위치:**
```
scripts/data/
├── setup_database.py
├── process_documents.py
├── load_embeddings.py
├── run_full_pipeline.py
└── inspect_vector_db.py
```

**이전 위치:**
- `scripts/` 바로 아래 있었음
- 최근 폴더 구조 개선으로 `scripts/data/`로 이동됨

**영향:**
- 실행 명령어 변경 필요: `python scripts/data/setup_database.py`
- 문서 업데이트 필요

---

## 📋 해결 우선순위

### P0 (Critical - 즉시 해결)

1. **`scripts/data/collect_arxiv_papers.py` 구현**
   - ArxivClient 래핑
   - PDF 다운로드 기능
   - 메타데이터 JSON 저장
   - 중복 제거
   - ExperimentManager 통합

2. **메타데이터 JSON 파일 재생성**
   - 기존 PDF 파일 기반으로 메타데이터 재수집
   - 또는 arXiv API로 재검색

3. **`.env.example` 파일 생성**
   - 환경 변수 설정 가이드

### P1 (High - 이번 주)

4. **`inspect_vector_db.py`를 PGVector용으로 수정**
   - FAISS 대신 PGVector 사용
   - 또는 별도 스크립트로 분리

5. **ExperimentManager 통합**
   - 모든 스크립트에 적용
   - 실험 로그 구조 통일

6. **폴더 구조 완성**
   - 필요한 폴더 생성
   - `.gitkeep` 파일 추가

### P2 (Medium - 다음 주)

7. **데이터베이스 스키마 수정**
   - `updated_at` 컬럼 추가
   - `difficulty_level` 타입 수정

8. **추가 pgvector 컬렉션 생성**
   - `paper_abstracts` (필요 시)
   - `glossary_embeddings` (필요 시)

9. **문서 업데이트**
   - 실행 명령어 경로 수정
   - 폴더 구조 반영

---

## ✅ 이미 해결된 항목

1. **PDF 파일 존재**: 75개 PDF 파일 확인됨
2. **스크립트 구조**: `scripts/data/` 폴더로 정리됨
3. **Document Loader**: 구현 완료 (`src/data/document_loader.py`)
4. **Embedding Manager**: 구현 완료 (`src/data/embeddings.py`)
5. **Database Setup**: 구현 완료 (`scripts/data/setup_database.py`)

---

## 🔧 즉시 수정 가능한 항목

### 1. 폴더 구조 생성

```bash
# 필요한 폴더 생성
mkdir -p data/raw/json
mkdir -p data/raw/txt
mkdir -p data/processed/chunks
mkdir -p data/processed/embeddings
mkdir -p data/outputs/conversations
mkdir -p data/outputs/summaries

# .gitkeep 파일 추가
touch data/raw/json/.gitkeep
touch data/raw/txt/.gitkeep
# ...
```

### 2. .env.example 생성

```bash
# .env.example
DATABASE_URL=postgresql://user:password@localhost:5432/papers
OPENAI_API_KEY=sk-...
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
```

### 3. run_full_pipeline.py 경로 수정

```python
# scripts/data/run_full_pipeline.py
steps = [
    ("scripts/data/collect_arxiv_papers.py", "Phase 1: arXiv 논문 수집"),
    ("scripts/data/setup_database.py", "Phase 2: PostgreSQL 데이터베이스 초기화"),
    ("scripts/data/process_documents.py", "Phase 3: PDF 문서 로드 및 청크 분할"),
    ("scripts/data/load_embeddings.py", "Phase 4: 임베딩 생성 및 Vector DB 저장"),
]
```

---

## 📊 현황 요약

| 항목 | 문서 요구사항 | 현재 상태 | 우선순위 |
|------|-------------|----------|---------|
| collect_arxiv_papers.py | ✅ 필요 | ❌ 없음 | P0 |
| ArxivPaperCollector | ✅ 필요 | ❌ 없음 (ArxivClient만 존재) | P0 |
| PDF 다운로드 | ✅ 필요 | ❌ 미구현 | P0 |
| 메타데이터 JSON | ✅ 필요 | ❌ 없음 | P0 |
| PDF 파일 | ✅ 필요 | ✅ 75개 존재 | - |
| ExperimentManager | ✅ 필요 | ❌ 미적용 | P1 |
| inspect_vector_db (pgvector) | ✅ 필요 | ❌ FAISS 사용 중 | P1 |
| 폴더 구조 | ✅ 필요 | ⚠️ 일부만 존재 | P1 |
| .env.example | ✅ 필요 | ❌ 없음 | P0 |

---

**다음 단계**: P0 항목부터 순차적으로 해결

