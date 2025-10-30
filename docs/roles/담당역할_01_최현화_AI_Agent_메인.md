# 담당역할: 최현화 - AI Agent 메인

## 담당자 정보
- **이름**: 최현화
- **역할**: 팀장
- **참여 기간**: 전체 기간
- **핵심 역할**: AI Agent 그래프 설계 및 구현, LLM 클라이언트, 메모리 시스템, 프로젝트 총괄

---

## 담당 모듈 및 도구

### 1. AI Agent 그래프 (`src/agent/`)
- LangGraph StateGraph 설계 및 구현
- 라우터 노드 (질문 분석 및 도구 선택)
- 조건부 엣지 (conditional_edges)
- Agent State 관리 (TypedDict)
- 도구 노드 연결 (6가지 도구)

### 2. LLM 클라이언트 (`src/llm/`)
- Langchain ChatOpenAI 및 Solar(Upstage) API 래퍼 구현
- 다중 LLM 선택 로직 (OpenAI + Solar)
- 에러 핸들링 및 재시도 로직
- 스트리밍 응답 처리 (astream)
- 토큰 사용량 추적 (get_openai_callback)
- Function calling 설정

### 3. 대화 메모리 시스템 (`src/memory/`)
- Langchain ConversationBufferMemory 구현
- 대화 히스토리 관리 (ChatMessageHistory)
- 컨텍스트 윈도우 최적화
- 세션 관리

### 4. 도구: 논문 요약 도구 (`src/tools/summarize.py`)
- Langchain @tool 데코레이터 활용
- load_summarize_chain 구현 (stuff, map_reduce, refine)
- 난이도별 요약 (Easy/Hard)
- 섹션별 요약 기능

### 5. 도구: 일반 답변 도구
- LLM 직접 호출 (ChatOpenAI)
- 간단한 인사, 일반 상식 질문 처리
- 난이도별 프롬프트 적용

### 6. 프로젝트 총괄
- 기능 통합 및 디버깅
- main.py 작성 (LangGraph 컴파일 및 실행)
- 코드 리뷰 및 PR 관리
- 발표 자료 총괄

---

## 도구 1: 일반 답변 도구

### 기능 설명
간단한 인사, 일반 상식 질문에 LLM의 자체 지식을 활용하여 직접 답변하는 도구

### 구현 방법

**파일 경로**: `src/agent/nodes.py`

1. **일반 답변 노드 함수 생성**
   - AgentState를 파라미터로 받는 `general_answer_node` 함수 정의
   - state에서 question과 difficulty 추출
   - 난이도에 따라 다른 SystemMessage 설정
     - Easy: 친절하고 이해하기 쉬운 언어로 답변하도록 지시
     - Hard: 전문적이고 기술적인 언어로 답변하도록 지시

2. **LLM 호출 구성**
   - langchain_openai.ChatOpenAI 사용
   - SystemMessage와 HumanMessage를 리스트로 구성
   - llm.invoke() 메서드로 메시지 전달
   - 응답 결과를 state["final_answer"]에 저장

3. **라우터 노드에서 일반 답변 판단 로직**
   - 사용자 질문을 LLM에 전달하여 적절한 도구 선택
   - 질문 유형 분류 프롬프트 작성 (일반 인사, 상식 질문 등)
   - 선택된 도구를 state["tool_choice"]에 저장

### 사용하는 DB
**DB 사용 없음** (LLM 자체 지식 활용)

---

## 도구 2: 논문 요약 도구

### 기능 설명
특정 논문의 전체 내용을 난이도별(Easy/Hard)로 요약하는 도구

### 구현 방법

**파일 경로**: `src/tools/summarize.py`, `src/llm/chains.py`

1. **논문 검색 및 전체 내용 조회** (`src/tools/summarize.py`)
   - @tool 데코레이터로 `summarize_paper` 함수 정의
   - 파라미터: paper_title (str), difficulty (str)
   - PostgreSQL 연결 (psycopg2 사용)
   - papers 테이블에서 ILIKE로 논문 제목 검색
   - paper_id 추출 후 Vector DB에서 해당 논문의 모든 청크 조회
   - filter 파라미터로 {"paper_id": paper_id} 전달
   - 난이도에 따라 적절한 요약 체인 선택 후 실행

2. **요약 체인 구현** (`src/llm/chains.py`)
   - Easy 모드 프롬프트: PromptTemplate로 초심자용 요약 규칙 정의
     - 전문 용어 쉽게 풀이, 핵심 아이디어 3가지 이내, 실생활 비유 포함
   - Hard 모드 프롬프트: 전문가용 요약 규칙 정의
     - 기술적 세부사항, 수식/알고리즘 설명, 관련 연구 비교
   - load_summarize_chain으로 체인 생성
     - chain_type: "stuff" (짧은 논문), "map_reduce" (중간 논문), "refine" (긴 논문)

3. **요약 방식 선택 로직**
   - 논문 청크 수에 따라 적절한 chain_type 선택
   - 5개 이하: stuff (모든 청크 한 번에 처리)
   - 5~15개: map_reduce (각 청크 요약 후 통합)
   - 15개 이상: refine (순차적 요약)

### 사용하는 DB

#### PostgreSQL + pgvector (Vector DB)
- **컬렉션**: `paper_chunks`
- **역할**: 논문 전체 내용을 청크로 나눠 저장 (pgvector extension 사용)
- **메타데이터 필터**: `paper_id`로 특정 논문의 모든 청크 조회
- **검색 방식**: 제목 유사도 검색 + 메타데이터 필터
- **벡터 검색**: Cosine Similarity, L2 Distance

#### PostgreSQL (관계형 데이터)
- **테이블**: `papers`
- **역할**: 논문 메타데이터 조회 (제목으로 paper_id 찾기)
- **쿼리**: `SELECT * FROM papers WHERE title ILIKE '%{paper_title}%'`

---

## LangGraph Agent 그래프 구현

### 구현 방법

**파일 경로**: `src/agent/state.py`, `src/agent/graph.py`

### 1. State 정의 (`src/agent/state.py`)
- TypedDict를 상속한 AgentState 클래스 정의
- 필수 필드:
  - question (str): 사용자 질문
  - difficulty (str): 난이도 (easy/hard)
  - tool_choice (str): 선택된 도구
  - tool_result (str): 도구 실행 결과
  - final_answer (str): 최종 답변
  - messages: Annotated[Sequence[BaseMessage], operator.add] - 대화 히스토리

### 2. 그래프 구성 (`src/agent/graph.py`)
- `create_agent_graph` 함수 생성
- StateGraph(AgentState) 인스턴스 생성
- 노드 추가:
  - workflow.add_node("router", router_node)
  - workflow.add_node("general", general_answer_node)
  - workflow.add_node("search_paper", search_paper_node)
  - workflow.add_node("web_search", web_search_node)
  - workflow.add_node("search_glossary", glossary_node)
  - workflow.add_node("summarize_paper", summarize_node)
  - workflow.add_node("save_file", save_file_node)
- 시작점 설정: workflow.set_entry_point("router")
- 조건부 엣지 설정: add_conditional_edges로 라우터에서 각 도구로 분기
- 모든 도구 노드에서 END로 연결
- workflow.compile()로 그래프 컴파일 후 반환

### 3. 라우터 노드 구현 (`src/agent/nodes.py`)
- `router_node` 함수 정의
- 사용자 질문을 분석하여 적절한 도구 선택
- 도구 목록과 각 도구의 사용 케이스를 포함한 프롬프트 작성
- LLM에게 프롬프트 전달하여 도구 이름 반환받기
- 반환된 도구 이름을 state["tool_choice"]에 저장
- 라우팅 결정 로그 출력

### 4. 라우팅 함수 (`src/agent/graph.py`)
- `route_to_tool` 함수: state["tool_choice"] 값을 반환
- add_conditional_edges에서 이 함수를 사용하여 다음 노드 결정

---

## LLM 클라이언트 구현

### 구현 방법

**파일 경로**: `src/llm/client.py`

### 1. 다중 LLM 클라이언트 클래스
- `LLMClient` 클래스 정의
- __init__ 메서드:
  - provider 파라미터로 "openai" 또는 "solar" 선택
  - provider에 따라 ChatOpenAI 또는 ChatUpstage 인스턴스 생성
  - 환경변수에서 API 키 로드 (OPENAI_API_KEY, UPSTAGE_API_KEY)
  - streaming=True 설정

### 2. 에러 핸들링 및 재시도
- tenacity 라이브러리의 @retry 데코레이터 사용
- `invoke_with_retry` 메서드:
  - stop_after_attempt(3): 최대 3회 재시도
  - wait_exponential: 지수 백오프 (2초 → 4초 → 8초)
  - LLM 호출 실패 시 자동 재시도

### 3. 토큰 사용량 추적
- `invoke_with_tracking` 메서드 구현
- OpenAI 사용 시: get_openai_callback으로 토큰 수와 비용 추적
- Solar 사용 시: 기본 로그만 출력
- 각 호출마다 토큰 정보 출력

### 4. 스트리밍 응답 처리
- `astream` 비동기 메서드 구현
- async for 루프로 LLM 응답을 청크 단위로 yield
- Streamlit UI에서 실시간 응답 표시에 사용

### 5. LLM 선택 전략
- `get_llm_for_task` 함수 구현
- 작업 유형별 최적 LLM 선택:
  - routing: Solar (빠른 응답)
  - generation: GPT-4 (높은 정확도)
  - summarization: GPT-4 (품질 중요)
  - 기본값: GPT-3.5-turbo (비용 효율)

---

## 대화 메모리 시스템

### 구현 방법

**파일 경로**: `src/memory/chat_history.py`

### 1. ChatMemoryManager 클래스
- ConversationBufferMemory 인스턴스 생성
  - return_messages=True: 메시지 객체 형태로 반환
  - memory_key="chat_history": 메모리 키 설정
- `add_user_message`: 사용자 메시지 추가
- `add_ai_message`: AI 메시지 추가
- `get_history`: 전체 대화 히스토리 반환
- `clear`: 대화 히스토리 초기화

### 2. 세션 기반 메모리 (선택사항)
- PostgresChatMessageHistory 사용
- `get_session_history` 함수:
  - session_id로 특정 세션의 대화 히스토리 조회
  - PostgreSQL에 대화 내용 영구 저장
  - 여러 사용자 세션 관리 가능

### 3. Agent와 메모리 통합
- Agent 실행 시 messages 필드에 메모리 히스토리 전달
- 응답 생성 후 사용자 메시지와 AI 메시지를 메모리에 추가
- 이후 질문에서 이전 대화 컨텍스트 활용

---

## 개발 일정

### Phase 1: LLM 클라이언트 및 공통 인프라
- ChatOpenAI 래퍼 구현
- 에러 핸들링 및 재시도 로직
- 토큰 사용량 추적
- 스트리밍 응답 처리

### Phase 2: LangGraph Agent 그래프
- State 정의
- 라우터 노드 구현
- 조건부 엣지 설정
- 일반 답변 노드 구현

### Phase 3: 메모리 시스템
- ConversationBufferMemory 구현
- 대화 히스토리 관리
- 세션 관리

### Phase 4: 논문 요약 도구
- load_summarize_chain 구현
- 난이도별 프롬프트 설계
- 요약 방식 선택 로직

### Phase 5: 통합 작업
- main.py 작성
- 모든 모듈 통합
- 디버깅 및 테스트

### Phase 6: 발표 준비
- 발표 자료 작성
- README.md 작성
- 최종 점검

---

## main.py 구현

### 구현 방법

**파일 경로**: `main.py` (프로젝트 루트)

1. **필요한 모듈 import**
   - src.agent.graph에서 create_agent_graph
   - src.llm.client에서 LLMClient
   - src.memory.chat_history에서 ChatMemoryManager

2. **초기화**
   - LLMClient 인스턴스 생성 (model="gpt-4", temperature=0.7)
   - create_agent_graph()로 Agent 생성
   - ChatMemoryManager 인스턴스 생성

3. **Agent 실행 루프**
   - 테스트 질문 리스트 준비 (질문, 난이도 튜플)
   - 각 질문에 대해:
     - agent.invoke()로 실행 (question, difficulty, messages 전달)
     - 결과에서 final_answer 추출
     - memory_manager에 사용자 메시지와 AI 메시지 추가
     - 결과 출력

4. **실행**
   - if __name__ == "__main__": main() 추가
   - 커맨드라인에서 python main.py로 실행

---

## Feature 브랜치

- `feature/agent-graph` - LangGraph 그래프 구현
- `feature/llm-client` - LLM 클라이언트 구현
- `feature/memory` - 대화 메모리 시스템
- `feature/tool-summarize` - 논문 요약 도구
- `feature/integration` - 통합 및 main.py

---

## 참고 자료

- LangGraph 공식 문서: https://langchain-ai.github.io/langgraph/
- Langchain ChatOpenAI: https://python.langchain.com/docs/integrations/chat/openai/
- Langchain Memory: https://python.langchain.com/docs/modules/memory/
- Langchain Summarization: https://python.langchain.com/docs/use_cases/summarization/
- Langchain Callbacks: https://python.langchain.com/docs/modules/callbacks/
