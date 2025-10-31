## 목적
- 논문 리뷰 Streamlit 애플리케이션의 기술 스택, 데이터베이스 선택 근거, 핵심 모듈 및 동작 흐름을 ML 지식 공유 문서 형태로 정리한다.
- 현재 구현 상태(최소 기능 데모)와 문제 해결 내역을 명확히 기록한다.

## 전반 아키텍처 개요
- UI: Streamlit
- LLM: OpenAI(ChatOpenAI)만 사용
- 임베딩: HuggingFaceEmbeddings(`sentence-transformers/all-MiniLM-L6-v2`)
- Vector DB: FAISS(로컬 디렉토리 저장)
- RDBMS: SQLite(논문 메타데이터, 용어집 저장)
- 검색: 우선 arXiv 메타데이터 API, 보조 DuckDuckGo(`ddgs`) 결과에서 arXiv URL 필터링
- RAG: FAISS Retriever + Prompt 체인(EZ/HARD 모드)
- 용어집: LLM 기반 JSON 추출 + 실패 시 빈도 기반 휴리스틱
- 환경변수: `.env`로 OpenAI 키 관리(`python-dotenv`)

## 데이터베이스 설계 및 선택 근거
### 1) Relational DB: SQLite
- 파일 경로: `data/rdbms/papers.db`
- 테이블
  - `papers(id, source, title, url, paper_id, summary, created_at)`
  - `glossary(id, term, definition, score, created_at)`
- 사용 목적
  - 검색된 논문 메타데이터(제목, URL, 요약 등) 영속화
  - 추출된 용어집(용어/정의/중요도) 영속화
- 선택 이유
  - 배포/실험 단계에서 운영 부담이 없는 무설정 파일 DB
  - 트랜잭션/인덱싱/간단한 조인 제공 → 소규모 실험 데이터에 충분
  - Python 표준 라이브러리 `sqlite3`로 간단히 사용 가능

### 2) Vector DB: FAISS
- 파일 경로: `data/vectordb/papers_faiss/`
- 사용 목적
  - 논문 텍스트(제목/요약 분할 문서) 임베딩을 벡터 인덱스로 저장하여 RAG 검색 수행
- 선택 이유
  - 로컬 환경에서 빠르고 가벼움, 설치가 용이
  - LangChain 통합 지원이 안정적이며, 소규모 PoC에 적합
- 운용 포인트
  - 인덱스가 없고 신규 문서도 없을 때 에러 방지용 더미 문서로 초기화
  - 기존 인덱스 존재 시 문서 추가 후 `save_local`로 지속 저장

## 핵심 모듈 및 라이브러리
- Streamlit: UI 프레임워크
- LangChain core/community/openai
  - `langchain_core`(Document, Prompt, Runnable)
  - `langchain_openai.ChatOpenAI` (LLM 호출)
  - `langchain_community.vectorstores.FAISS` (벡터DB)
  - `langchain_community.embeddings.HuggingFaceEmbeddings` (임베딩)
  - `langchain_text_splitters.RecursiveCharacterTextSplitter` (문서 분할)
- 검색
  - `arxiv` Python 패키지: PDF 다운로드 없이 메타데이터만 가져와 Document 구성
  - `ddgs`(DuckDuckGo): 일반 검색 결과에서 arXiv 도메인 URL만 필터링하여 보조 결과 확보
- 환경/유틸
  - `python-dotenv`: `.env`에서 `OPENAI_API_KEY`, `OPENAI_MODEL` 로드
  - `warnings`, `logging`: 불필요 경고 억제, 로깅 레벨 제어
- 기계학습 스택
  - `transformers`, `torch`: 텍스트 분할/토크나이저 등 내부 의존(직접 학습 미수행, 임베딩/LLM 호출 중심)

## LLM/임베딩 정책
- LLM: OpenAI만 지원(요청에 따라 단순화). 복잡한 폴백 제거.
  - 모델 기본값: `gpt-4o-mini`
  - API 키 검증: 접두(`sk-` 또는 `sk-proj-`), 길이(>=20자), 미설정 시 명확한 오류 메시지
- 임베딩: `sentence-transformers/all-MiniLM-L6-v2`
  - HuggingFaceEmbeddings 사용(키 불필요)
  - TF/Transformers 충돌 완화: `TRANSFORMERS_NO_TF=1`, 경고 억제

## 검색 전략
- 1순위: arXiv API 메타데이터 검색(제목/요약만 사용, PDF 미다운로드)
- 2순위: DuckDuckGo 검색 결과 문자열에서 URL 정규식 추출 후 `arxiv.org` 포함 링크만 채택
- 기록: 과거 Google Scholar 시도 → 비안정, 제거. "DuckDuckGo + arXiv 필터"로 단순·안정화

## 문서 전처리 및 색인
- 분할: `RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=120)`
- 저장: `FAISS.from_documents` 생성 또는 기존 인덱스 로드 후 `add_documents`
- 예외 처리: 비어있는 입력/인덱스 상황에서 더미 문서 초기화로 안정성 확보

## RAG 체인
- Retriever: `FAISS.as_retriever(k=6)`
- Prompt: EZ/HARD 모드
  - EZ: 비전공자 대상 쉬운 설명, 비유/요약 강조
  - HARD: 대학원 수준 기술 설명, 용어/수식 직관 설명
- 실행: `chain.invoke({mode, question})` → LLM 응답 반환

## 용어집(Glossary) 생성
- LLM 추출
  - 시스템 프롬프트에 JSON 배열 형식 요구(예시 키는 이스케이프 처리)
  - 실패/비정형 응답 시 너그럽게 JSON 파싱 시도
- 휴리스틱 백업
  - 토큰화(간단 정규식), 불용어 제거 후 출현 빈도 기반 Top-K 선택
  - 정의는 빈 문자열, score는 상대 빈도로 추정
- 저장: SQLite `glossary` 테이블에 UPSERT

## 환경 변수 및 보안
- `.env` 경로: 프로젝트 루트에 배치(앱에서 명시 경로 재확인 후 로드)
- 노출 방지: 사이드바에 키 미리보기(앞 7/뒤 4 글자)만 표시
- 실패 메시지: 인증/타임아웃/RateLimit 등 OpenAI 전용 친절 메시지 제공

## 경고/에러 대응 내역
- OpenAI `AuthenticationError(401)`: 키 형식/길이 검증 및 구체적 해결 가이드 제공
- 템플릿 `KeyError`: 프롬프트 템플릿의 리터럴 JSON 예시를 `{{ ... }}`로 이스케이프
- `FAISS.from_documents([])` → `IndexError`: 더미 문서 초기화로 해결
- `tuple`에 `page_content` 없음: 분할 결과 처리 시 `extend(split_docs)`로 수정
- Transformers-Keras 충돌: TensorFlow 의존 비활성(`TRANSFORMERS_NO_TF=1`), 커뮤니티 임베딩 경로 사용
- `ddgs` 미설치: 설치 및 `requirements.txt` 반영
- `PyMuPDF` 미설치로 PDF 로딩 실패: PDF 미사용(메타데이터만)으로 전략 전환
- PyTorch 경고/충돌
  - `Examining the path of torch.classes...`: 경고 억제 필터 추가
  - `Only a single TORCH_LIBRARY...`: Streamlit 재실행 시 중복 로드 가능성 → 임포트 지연/보호, 민감 임포트 최소화
  - `TORCH_LOGS` 유효하지 않은 값 오류: 환경 변수 제거, `logging`로 제어

## 디렉터리 구조 상 연계
- 코드: `experiments/paper_review_app.py`
- 데이터
  - RDBMS: `data/rdbms/papers.db`
  - Vector DB: `data/vectordb/papers_faiss/`
- 문서(본 문서): `docs/architecture/paper_review_app_ml_brief.md`

## 실행 전제 및 요구사항
- Python 패키지(일부)
  - `streamlit`, `langchain-core`, `langchain-community`, `langchain-openai`
  - `langchain-text-splitters`, `sentence-transformers`, `faiss-cpu`
  - `ddgs`, `arxiv`, `python-dotenv`, `pymupdf`(현재는 메타데이터만 사용)
  - `transformers`, `torch`
- `.env` 설정
  - `OPENAI_API_KEY=sk-...` 또는 `OPENAI_API_KEY=sk-proj-...`
  - 선택: `OPENAI_MODEL=gpt-4o-mini`

## 운용 팁
- 키 오류 발생 시: 사이드바의 에러 메시지와 `.env` 경로 안내 확인
- 검색 결과가 비어있을 때: 키워드에 분야/연도를 포함하거나 arXiv ID 직접 입력
- 성능 이슈: 벡터DB 삭제 후 재색인(`data/vectordb/papers_faiss` 삭제) → 깨끗한 상태에서 재시작

## 향후 개선 포인트
- PDF 파서 추가(선택): `pymupdf`/`pdfminer` 등으로 전문 색인
- UI: 검색 결과 미리보기, 선택적 인덱싱, 하이라이트
- 품질: 하드모드 프롬프트에 수식/증명 템플릿 강화
- 추적: `mlflow` 또는 `wandb` 연동(실험/지표 로깅)

## 평가 기준(Assessment)
### RAG 평가
- 정답성/충실성(offline)
  - Recall@K(문서 검색): 정답 근거가 상위 K 문서에 포함되는 비율. K∈{3,5,10}
  - Precision@K, MRR: 검색 품질의 정확·순위 민감도 측정
  - EM(Exact Match), F1(토큰 단위) for 답변: 기준 정답 대비 일치도
  - Faithfulness(환각률): 답변이 제공된 컨텍스트에 근거하는 비율(=1-환각률)
  - Context Utilization(%): 답변이 실제로 어느 컨텍스트 조각을 참조했는지 매칭 비율
- 효율성(online/latency)
  - End-to-end 지연(ms): 검색+생성 총 소요시간 p50/p95
  - 토큰 비용/호출 비용(원): 질문당 평균 프롬프트/출력 토큰 수, API 비용 추정
- 강건성
  - No-answer handling: 근거 부재 시 “근거 없음” 응답 정확히 반환하는 비율
  - 길이/잡음 민감도: 매우 긴 요약/문서 섞임/중복 컨텐츠에서 성능 저하 정도
- 권장 툴/프로토콜
  - 데이터셋: 소규모 GT 쿼리-정답-근거 세트(예: 50~200개)를 CSV로 준비
  - 도구: `ragas`, `langchain` eval, 수동 판정 템플릿(스트림릿 내 표출 가능)
  - 절차: (1) 인덱스 고정 → (2) 쿼리 배치 실행 → (3) 메트릭 산출 → (4) 리그레션 추적
- 권장 수용 기준(초기 PoC)
  - Recall@5 ≥ 0.6, EM ≥ 0.4 또는 F1 ≥ 0.6, Faithfulness ≥ 0.9, p95 지연 ≤ 6s

### RDB(SQLite) 평가
- 성능/효율
  - Query Latency(ms): 주요 쿼리(papers 최신 N건, term 상위 N개 등) p50/p95
  - Insert/Upsert Throughput: 초당 처리 건수, 배치 삽입 시 평균 소요
  - 파일 크기/성장률: `papers.db` 크기, 주당 증가량, VACUUM 후 절감률
- 무결성/품질
  - 스키마 일관성: NULL/타입 위반 0건, 제약조건 위반 0건
  - 중복률: 동일 URL/ID 중복 삽입 비율(UPSERT 정책으로 0에 근접)
  - 백업/복구 테스트: 주 1회 백업 후 복원 성공률 100%
- 유지보수성
  - 인덱스 활용도: `EXPLAIN QUERY PLAN`으로 풀스캔 회피 확인
  - 마이그레이션 용이성: 컬럼 추가(예: tags) 시 기존 쿼리 호환성 유지율
- 권장 툴/프로토콜
  - `sqlite3` CLI, `EXPLAIN QUERY PLAN`, `ANALYZE`
  - 스트림릿 진단 탭(선택): 최근 레코드 수, 파일 크기, 인덱스 유무 표시
- 권장 수용 기준(초기 PoC)
  - p95 조회 지연 ≤ 50ms, p95 삽입 지연 ≤ 30ms, 중복률 ≤ 1%, 복구 성공률 100%
