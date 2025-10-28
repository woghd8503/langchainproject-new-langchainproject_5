# GitHub 이슈 라벨 정의

프로젝트 진행 시 사용할 GitHub Issue 라벨 목록 및 설명입니다.

---

## 📌 이슈 유형 (Type)

| 라벨명 | 설명 | 사용 예시 |
|--------|------|-----------|
| `bug` | 버그 발생 시 | 코드 실행 오류, 예상과 다른 동작, 기능 오작동 |
| `feature` | 새 기능 관련 | 새로운 기능 개발, 신규 모듈 추가 |
| `enhancement` | 기능 개선 / 추가 요청 | 기존 기능 성능 향상, UI/UX 개선 |
| `refactor` | 코드 리팩토링 | 코드 구조 개선, 중복 코드 제거, 가독성 향상 |
| `documentation` | 문서 추가 또는 개선 | README 작성, 주석 추가, 사용법 가이드 작성 |
| `testing` | 테스트 코드 작성 및 수정 | 단위 테스트, 통합 테스트, 평가 코드 |

---

## 🔧 기술 스택 (Tech Stack)

| 라벨명 | 설명 | 사용 예시 |
|--------|------|-----------|
| `rag` | RAG 파이프라인 관련 | VectorDB 구축, Retriever 개발, 문서 검색, 텍스트 분할 |
| `agent` | AI Agent 로직 관련 | Agent 그래프 구성, 질문 라우팅, ReAct 패턴, 노드/엣지 추가 |
| `prompt` | 프롬프트 엔지니어링 관련 | System Prompt 작성, Agent Prompt 개선, RAG Prompt 최적화 |
| `tool` | Agent 도구 개발/수정 | 웹 검색 도구, 파일 저장 도구, 커스텀 도구 개발 |
| `ui` | 챗봇 UI 관련 | Gradio 인터페이스, Streamlit 대시보드, UI 컴포넌트 개선 |
| `memory` | 멀티턴 대화 메모리 관련 | ConversationBuffer 구현, 대화 이력 관리, 세션 저장/로드 |
| `evaluation` | 성능 평가 시스템 | 평가 메트릭 개발, 성능 측정 코드, 결과 분석 도구 |
| `text2sql` | Text2SQL 기능 관련 | SQL 쿼리 생성, DB 스키마 로드, 쿼리 실행 로직 |
| `vectordb` | VectorDB 구축 및 관리 | ChromaDB 설정, 임베딩 저장, DB 인덱싱 |
| `embedding` | 임베딩 모델 관련 | OpenAI Embeddings, Upstage Embeddings, 성능 비교 |
| `model` | LLM 모델 관련 | 모델 선택 및 설정, API 연동, 파라미터 튜닝 |
| `api` | API 통합 관련 | OpenAI/Upstage API 연동, API 키 관리, Rate limit 처리 |

---

## 📊 데이터 관련 (Data)

| 라벨명 | 설명 | 사용 예시 |
|--------|------|-----------|
| `data` | 데이터 관련 | 데이터 수집, 전처리, 형식 변환, 검증 |
| `pipeline` | 전반적인 파이프라인 최적화 | ETL 파이프라인, 실행 워크플로우, 자동화 스크립트 |

---

## 🚀 개발 환경 (Development)

| 라벨명 | 설명 | 사용 예시 |
|--------|------|-----------|
| `environment` | 개발 환경 관련 | 가상환경 설정, 의존성 설치 문제, 환경 변수 설정 |
| `config` | 설정 파일 관련 | YAML 설정 파일, Config 로더 개발, 환경별 설정 분리 |
| `deployment` | 배포 및 실행 환경 | Docker 컨테이너화, 서버 배포, CI/CD 파이프라인 |
| `integration` | 모듈 통합 작업 | main.py 통합, 모듈 간 연동, API 통합 |

---

## ⚡ 우선순위 (Priority)

| 라벨명 | 설명 | 사용 예시 |
|--------|------|-----------|
| `critical` | 중요도 최상 (당장 처리) | 시스템 다운, 치명적 버그, 발표 전 필수 수정 |
| `high` | 중요도 높은 이슈 | 주요 기능 오류, 필수 기능 미구현 |
| `medium` | 중요도 보통 이슈 | 일반적인 개선 사항, 부가 기능 |
| `low` | 중요도 낮은 이슈 | 사소한 개선, 향후 고려 사항 |

---

## 🔄 상태 관련 (Status)

| 라벨명 | 설명 | 사용 예시 |
|--------|------|-----------|
| `help wanted` | 추가적인 도움 필요 | 협업 요청, 기술 자문 필요, 리뷰 요청 |
| `question` | 질문 관련 | 기술 문의, 구현 방법 논의, 멘토님 질문 |
| `duplicate` | 중복된 이슈 | 이미 등록된 이슈와 동일한 내용 |
| `invalid` | 잘못된 이슈 또는 무효 처리 | 잘못 생성된 이슈, 프로젝트 범위 외 |
| `wontfix` | 해결하지 않을 이슈 (보류/폐기) | 우선순위 낮음, 프로젝트 방향과 맞지 않음 |

---

## 🧪 실험 관련 (Experiment)

| 라벨명 | 설명 | 사용 예시 |
|--------|------|-----------|
| `experiment` | 실험 관련 | 모델 성능 실험, 하이퍼파라미터 튜닝, A/B 테스트, 프롬프트 실험 |
| `optimization` | 성능 또는 속도 개선 | 응답 속도 향상, 메모리 사용량 최적화, 비용 절감 |

---

## 📝 사용 가이드

### 라벨 조합 예시

| 작업 내용 | 라벨 조합 |
|-----------|-----------|
| 새로운 RAG 기능 개발 | `feature` + `rag` + `high` |
| Agent 도구 버그 수정 | `bug` + `agent` + `tool` + `critical` |
| 프롬프트 개선 실험 | `experiment` + `prompt` + `medium` |
| UI 문서 작성 | `documentation` + `ui` + `low` |
| VectorDB 성능 최적화 | `optimization` + `vectordb` + `high` |
| Text2SQL 평가 코드 작성 | `testing` + `evaluation` + `text2sql` + `medium` |

### 라벨 선택 규칙

1. **최소 2개 이상** 라벨 사용 (유형 + 기술스택/우선순위)
2. **우선순위 라벨** 필수 포함
3. **기술 스택 라벨**은 관련된 모든 것 추가 가능
4. **명확한 제목**과 함께 사용

---

## 🎯 발표 준비 관련 라벨 조합

프로젝트 발표(11/06)를 위한 중요 이슈들:

| 작업 내용 | 라벨 조합 |
|-----------|-----------|
| 발표 자료 작성 | `critical` + `documentation` |
| 데모 UI 완성 | `critical` + `ui` |
| 사용자 시나리오 10개 테스트 | `high` + `experiment` |
| 성능 평가 결과 | `high` + `evaluation` |
