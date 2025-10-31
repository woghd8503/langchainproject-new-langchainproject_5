## 제목: arXiv 논문 수집, Document 처리, 임베딩 생성 및 PostgreSQL + pgvector 적재

---

## 📋 작업 개요

**작업 주제:** arXiv 논문 수집, Document 처리, 임베딩 생성 및 PostgreSQL + pgvector 적재

**담당자:** @박재홍

**마감일:** 10/31 24:00 (단기 집중, 4일)

## 📅 기간

- 시작일: 2025-10-28
- 종료일: 2025-10-31

---

## 📌 이슈 목적

arXiv API를 사용하여 AI/ML 관련 논문 데이터를 수집하고, Langchain Document Loader로 처리한 후 PostgreSQL + pgvector에 저장하는 데이터 파이프라인을 구축합니다. 최소 50-100편의 논문 데이터를 확보하여 RAG 시스템의 기반을 마련합니다.

**핵심 목표:**
- arXiv에서 최소 50-100편 논문 수집 (PDF 다운로드)
- Langchain PyPDFLoader로 PDF → Document 변환
- RecursiveCharacterTextSplitter로 청크 분할
- OpenAI Embeddings로 임베딩 생성
- PostgreSQL + pgvector에 임베딩 저장
- 용어집 초기 데이터 삽입

---

## ✅ 작업 항목 체크리스트

### Phase 1: 데이터 수집 스크립트 (1일)

* [ ] ArxivPaperCollector 클래스 구현 (scripts/collect_arxiv_papers.py)

* [ ] arxiv 패키지 사용하여 논문 검색

* [ ] collect_papers() 메서드 (query, max_results 파라미터)

* [ ] PDF 다운로드 및 저장 (data/raw/pdfs/)

* [ ] 메타데이터 추출 (title, authors, publish_date, summary, pdf_url, categories)

* [ ] collect_by_keywords() 메서드 (여러 키워드 반복 수집)

* [ ] remove_duplicates() 메서드 (제목 기준 중복 제거)

* [ ] AI/ML 키워드 리스트 정의
  - "transformer attention", "BERT GPT", "large language model"
  - "retrieval augmented generation", "neural machine translation"
  - "question answering", "AI agent"

* [ ] 메타데이터 JSON 파일 저장 (data/raw/arxiv_papers_metadata.json)

### Phase 2: Document Loader 구현 (1일)

* [ ] PaperDocumentLoader 클래스 구현 (src/data/document_loader.py)

* [ ] RecursiveCharacterTextSplitter 초기화 (chunk_size=1000, chunk_overlap=200)

* [ ] load_pdf() 메서드 (PyPDFLoader 사용)

* [ ] load_and_split() 메서드 (PDF → 청크 분할)

* [ ] 각 청크에 chunk_id 메타데이터 추가

* [ ] load_all_pdfs() 메서드 (디렉토리 전체 PDF 처리)

* [ ] 메타데이터 매핑 (arXiv ID → 논문 정보)

### Phase 3: 임베딩 및 Vector DB 적재 (1일)

* [ ] PaperEmbeddingManager 클래스 구현 (src/data/embeddings.py)

* [ ] OpenAI Embeddings 초기화 (text-embedding-3-small)

* [ ] PGVector VectorStore 초기화 (collection: paper_chunks)

* [ ] add_documents() 메서드 (배치 처리, batch_size=50)

* [ ] add_documents_with_paper_id() 메서드 (paper_id 메타데이터 추가)

* [ ] 진행 상황 로깅

* [ ] 오류 처리 (API 속도 제한 대응)

* [ ] 배치 처리 로직 (OpenAI API Rate Limit 고려)

### Phase 4: 데이터베이스 초기 설정 (1일)

* [ ] PostgreSQL 스키마 생성 (scripts/setup_database.py)

* [ ] create_tables() 함수
  - papers 테이블 생성 (paper_id, title, authors, publish_date, url, category, citation_count, abstract)
  - glossary 테이블 생성 (term_id, term, definition, easy_explanation, hard_explanation, category)

* [ ] 인덱스 생성 (papers.title, glossary.term)

* [ ] insert_paper_metadata() 함수
  - JSON 메타데이터를 papers 테이블에 INSERT
  - ON CONFLICT (url) DO NOTHING (중복 방지)
  - RETURNING paper_id로 ID 조회
  - arXiv ID → paper_id 매핑 딕셔너리 생성

* [ ] insert_glossary_data() 함수
  - 초기 용어집 데이터 삽입 (Attention Mechanism, Fine-tuning, BLEU Score 등)

* [ ] pgvector extension 설치 및 초기화

* [ ] paper_id_mapping.json 저장 (data/processed/)

### Phase 5: 인수인계 문서 작성 (반나절)

* [ ] 완료 항목 체크리스트

* [ ] 데이터 위치 명시 (PDF, 메타데이터, Vector DB)

* [ ] 추가 데이터 수집 방법 가이드

* [ ] DB 연결 정보 및 주의사항

* [ ] Logger 적용 및 로그 기록

---

## 📦 설치/실행 명령어 예시

```bash
# 가상환경 활성화
source .venv/bin/activate

# 필요한 패키지 설치
pip install arxiv pypdf langchain langchain-openai langchain-postgres pgvector psycopg2-binary

# PostgreSQL 연결 환경변수 설정
export DATABASE_URL="postgresql://user:password@localhost:5432/papers"
export OPENAI_API_KEY="your-openai-api-key"

# 1단계: arXiv 논문 수집 (약 20-30분 소요)
python scripts/collect_arxiv_papers.py

# 2단계: PostgreSQL 스키마 생성
python scripts/setup_database.py

# 3단계: Document 로드 및 청크 분할
# (PaperDocumentLoader 클래스를 직접 사용하거나 스크립트 작성 필요)
python -c "
from src.data.document_loader import PaperDocumentLoader
from pathlib import Path
import json

loader = PaperDocumentLoader(chunk_size=1000, chunk_overlap=200)
pdf_dir = Path('data/raw/pdfs')
metadata_path = Path('data/raw/arxiv_papers_metadata.json')

chunks = loader.load_all_pdfs(pdf_dir, metadata_path)
print(f'Loaded {len(chunks)} chunks from PDFs')
"

# 4단계: 임베딩 생성 및 Vector DB 저장 (시간 소요 큼, 배치 처리)
# (PaperEmbeddingManager 클래스를 직접 사용하거나 스크립트 작성 필요)
python -c "
from src.data.embeddings import PaperEmbeddingManager
from src.data.document_loader import PaperDocumentLoader
from pathlib import Path
import json

# Document 로드
loader = PaperDocumentLoader()
pdf_dir = Path('data/raw/pdfs')
metadata_path = Path('data/raw/arxiv_papers_metadata.json')
chunks = loader.load_all_pdfs(pdf_dir, metadata_path)

# paper_id_mapping 로드
with open('data/processed/paper_id_mapping.json', 'r') as f:
    mapping = json.load(f)

# 임베딩 및 저장
manager = PaperEmbeddingManager()
count = manager.add_documents_with_paper_id(chunks, mapping)
print(f'Saved {count} documents to vector DB')
"

# 데이터 확인
ls -lh data/raw/pdfs/  # PDF 파일 확인
cat data/raw/arxiv_papers_metadata.json | jq length  # 논문 개수 확인
```

---

### ⚡️ 참고

**중요 사항:**
- 단기 집중 작업: 4일 내에 데이터 파이프라인 완성 (최우선 과제)
- 최소 목표: 50-100편 논문, 실제로는 70-100편 권장
- 배치 처리 필수: OpenAI API 속도 제한 대응 (batch_size=50)
- 오류 처리: PDF 다운로드 실패, API 오류 시 해당 논문 건너뛰기
- 중복 제거: 논문 제목 기준으로 중복 제거

**데이터 위치:**
- `data/raw/pdfs/`: 다운로드한 PDF 파일
- `data/raw/arxiv_papers_metadata.json`: 논문 메타데이터
- `data/processed/paper_id_mapping.json`: arXiv ID → paper_id 매핑
- PostgreSQL + pgvector: 임베딩 벡터 및 메타데이터

**키워드 선택 전략:**
- 핵심 주제: Transformer, BERT, GPT, LLM
- 응용 분야: RAG, QA, NMT
- 최신 트렌드: AI Agent, Few-shot Learning

---

### 유용한 링크

**필수 참고 PRD 문서:**
- [ ] `docs/PRD/01_프로젝트_개요.md` - 프로젝트 전체 개요
- [ ] `docs/PRD/02_프로젝트_구조.md` - 폴더 구조 (data/, scripts/)
- [ ] `docs/PRD/05_로깅_시스템.md` ⭐ - Logger 사용법
- [ ] `docs/PRD/06_실험_추적_관리.md` ⭐ - 실험 폴더 구조
- [ ] `docs/PRD/10_기술_요구사항.md` - arXiv API, PyPDFLoader, OpenAI Embeddings
- [ ] `docs/PRD/11_데이터베이스_설계.md` - papers 테이블 스키마
- [ ] `docs/PRD/13_RAG_시스템_설계.md` - Document 처리 및 Text Splitting

**참고 PRD 문서:**
- [ ] `docs/PRD/03_브랜치_전략.md` - Feature 브랜치
- [ ] `docs/PRD/04_일정_관리.md` - 개발 일정

**외부 링크:**
- [ ] [arXiv API](https://info.arxiv.org/help/api/index.html)
- [ ] [Langchain Document Loaders](https://python.langchain.com/docs/integrations/document_loaders/)
- [ ] [Langchain Text Splitters](https://python.langchain.com/docs/modules/data_connection/document_transformers/)
- [ ] [pgvector 문서](https://github.com/pgvector/pgvector)

