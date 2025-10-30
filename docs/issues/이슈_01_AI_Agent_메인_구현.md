## 제목 : AI Agent 메인 시스템 구현 (LangGraph + 도구 통합)

---

## 📋 작업 개요
**작업 주제:** LangGraph 기반 AI Agent 시스템 개발 및 도구 통합
**담당자:** @최현화
**마감일:** 11/03 24:00

## 📅 기간
- 시작일: 2025-10-28
- 종료일: 2025-11-03

---

## 📌 이슈 목적

LangGraph를 사용하여 논문 리뷰 챗봇의 핵심 AI Agent 시스템을 구현합니다. 사용자 질문을 분석하여 적절한 도구(RAG 검색, 용어집, 웹 검색, 요약, 파일 저장)를 자동으로 선택하고 실행하는 지능형 Agent를 개발합니다.

**핵심 목표:**
- LangGraph StateGraph 구조 설계 및 구현
- 질문 라우팅 로직 구현 (일반 답변 / RAG 검색 / 용어집 / 웹 검색 / 요약)
- 5가지 Langchain 도구 통합
- 멀티턴 대화 메모리 관리
- OpenAI + Solar(Upstage) 듀얼 LLM 전략

---

## ✅ 작업 항목 체크리스트

### Phase 1: LangGraph 기본 구조 (2일)
- [ ] AgentState 정의 (TypedDict: question, difficulty, messages, tool_result, final_answer, next_action)
- [ ] StateGraph 생성 및 노드 추가
- [ ] 5개 노드 구현
  - [ ] `router_node`: 질문 분석 및 라우팅 (일반/RAG/용어집/웹/요약 판단)
  - [ ] `general_answer_node`: 일반적인 질문에 직접 답변
  - [ ] `search_paper_node`: RAG 검색 도구 호출 → 답변 생성
  - [ ] `glossary_node`: 용어집 도구 호출 → 답변 생성
  - [ ] `web_search_node`: 웹 검색 도구 호출 → 답변 생성
- [ ] 조건부 엣지 구현 (`route_question` 함수)
- [ ] Agent 컴파일 및 실행 테스트

### Phase 2: 도구 통합 (2일)
- [ ] Langchain @tool 데코레이터 기반 도구 5개 통합
  - [ ] 도구 1: `search_paper_database` (RAG 검색)
  - [ ] 도구 2: `search_latest_papers` (웹 검색)
  - [ ] 도구 3: `search_glossary` (용어집)
  - [ ] 도구 4: `summarize_paper` (논문 요약)
  - [ ] 도구 5: `save_to_file` (파일 저장)
- [ ] ToolNode 생성 및 Agent에 통합
- [ ] 도구 실행 결과 State 업데이트 로직

### Phase 3: LLM 클라이언트 및 메모리 (2일)
- [ ] LLM 클라이언트 구현 (`src/llm/client.py`)
  - [ ] OpenAI API 연동 (gpt-4o-mini, gpt-4o)
  - [ ] Solar API 연동 (solar-pro, solar-mini)
  - [ ] fallback 로직 (OpenAI 실패 시 Solar 사용)
  - [ ] 에러 핸들링 및 재시도
- [ ] ChatMemoryManager 구현 (`src/memory/chat_history.py`)
  - [ ] Langchain ChatMessageHistory 사용
  - [ ] add_user_message(), add_ai_message()
  - [ ] get_history() (최근 10턴 반환)
  - [ ] clear() 메모리 초기화
- [ ] 난이도별 프롬프트 적용 (Easy/Hard 모드)

### Phase 4: 통합 및 테스트 (1일)
- [ ] Agent 그래프 통합 (`src/agent/graph.py`)
- [ ] create_agent_graph() 함수 완성
- [ ] 단위 테스트 작성 (`tests/test_agent.py`)
  - [ ] router_node 테스트
  - [ ] general_answer_node 테스트
  - [ ] 도구 통합 테스트
  - [ ] 메모리 저장/로드 테스트
- [ ] main.py 통합 실행 테스트

### Phase 5: 로깅 및 문서화 (1일)
- [ ] Logger 클래스를 사용한 로깅 적용
  - [ ] 실험 폴더 생성 (experiments/날짜/날짜_시간_실험명/)
  - [ ] experiment.log 기록
  - [ ] config.yaml 저장
  - [ ] results.json 저장
- [ ] 코드 주석 작성
- [ ] 사용 예시 문서 작성

---

## 📦 설치/실행 명령어 예시

```bash
# 가상환경 활성화
source .venv/bin/activate

# 필요한 패키지 설치
pip install langchain langchain-openai langchain-upstage langgraph

# 환경변수 설정
export OPENAI_API_KEY="your-openai-api-key"
export UPSTAGE_API_KEY="your-upstage-api-key"

# Agent 실행 테스트
python src/agent/graph.py

# 단위 테스트 실행
pytest tests/test_agent.py -v
```

---

### ⚡️ 참고

**중요 사항:**
1. **PRD 05, 06 필수 준수**: 모든 print()를 logger.write()로 변경, 실험 폴더 구조 준수
2. **LangGraph 패턴**: StateGraph → add_node → add_edge → add_conditional_edges → compile
3. **도구 통합**: Langchain @tool 데코레이터 사용, invoke() 메서드로 도구 호출
4. **듀얼 LLM**: OpenAI 우선, 실패 시 Solar로 fallback
5. **난이도 모드**: Easy(초심자용), Hard(전문가용) 프롬프트 분리

**주의:**
- Agent 노드에서 반드시 `return state` (State 업데이트)
- 조건부 엣지에서 명확한 라우팅 키 반환 ("general", "search_paper", "glossary", "web_search")
- 메모리는 최근 10턴만 유지 (토큰 절약)

---

### 유용한 링크

**필수 참고 PRD 문서:**
- `docs/PRD/01_프로젝트_개요.md` - 프로젝트 전체 개요 및 목표
- `docs/PRD/02_프로젝트_구조.md` - 폴더 구조 및 모듈 배치
- `docs/PRD/05_로깅_시스템.md` ⭐ - Logger 클래스 사용법 및 규칙
- `docs/PRD/06_실험_추적_관리.md` ⭐ - 실험 폴더 구조 및 명명 규칙
- `docs/PRD/10_기술_요구사항.md` - 기술 스택 및 라이브러리
- `docs/PRD/12_AI_Agent_설계.md` - LangGraph 구조 및 도구 정의
- `docs/PRD/14_LLM_설정.md` - LLM 선택 전략 및 에러 핸들링

**참고 PRD 문서:**
- `docs/PRD/03_브랜치_전략.md` - Feature 브랜치 전략
- `docs/PRD/04_일정_관리.md` - 개발 일정 및 마일스톤
- `docs/PRD/11_데이터베이스_설계.md` - DB 스키마
- `docs/PRD/13_RAG_시스템_설계.md` - RAG 파이프라인
- `docs/PRD/15_프롬프트_엔지니어링.md` - 프롬프트 템플릿

**외부 링크:**
- LangGraph 공식 문서: https://langchain-ai.github.io/langgraph/
- Langchain Tools: https://python.langchain.com/docs/modules/agents/tools/
- Langchain Memory: https://python.langchain.com/docs/modules/memory/

**자료조사 문서:**
- `docs/research/01_자료조사_LangGraph.md`

## 🔖 추천 라벨

`feature` `agent` `tool` `memory` `integration` `high` `critical`

---
