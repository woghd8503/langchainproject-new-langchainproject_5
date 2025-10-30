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

### 예제 코드

```python
# src/agent/nodes.py

import os
from datetime import datetime
from typing import TypedDict
from langchain_openai import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage
from src.utils.logger import Logger

# Logger 초기화
today = datetime.now().strftime("%Y%m%d")
time_now = datetime.now().strftime("%H%M%S")
experiment_name = "agent_general"
log_dir = f"experiments/{today}/{today}_{time_now}_{experiment_name}"
os.makedirs(log_dir, exist_ok=True)
logger = Logger(log_path=f"{log_dir}/experiment.log")

class AgentState(TypedDict):
    question: str
    difficulty: str
    tool_choice: str
    final_answer: str

def general_answer_node(state: AgentState):
    """
    일반 답변 노드: LLM의 자체 지식으로 직접 답변
    """
    question = state["question"]
    difficulty = state.get("difficulty", "easy")

    logger.write(f"일반 답변 노드 실행: {question}")
    logger.write(f"난이도: {difficulty}")

    # 난이도에 따른 SystemMessage 설정
    if difficulty == "easy":
        system_msg = SystemMessage(content="""
당신은 친절한 AI 어시스턴트입니다.
초심자도 이해할 수 있도록 쉽고 명확하게 답변해주세요.
전문 용어는 최소화하고 일상적인 언어를 사용하세요.
        """)
    else:  # hard
        system_msg = SystemMessage(content="""
당신은 전문적인 AI 어시스턴트입니다.
기술적인 세부사항을 포함하여 정확하고 전문적으로 답변해주세요.
        """)

    # LLM 초기화
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)

    # 메시지 구성 및 LLM 호출
    messages = [system_msg, HumanMessage(content=question)]
    response = llm.invoke(messages)

    logger.write(f"LLM 응답: {response.content}")

    # 최종 답변 저장
    state["final_answer"] = response.content

    return state
```

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

### 예제 코드

```python
# src/tools/summarize.py

import os
from datetime import datetime
from langchain.tools import tool
from langchain_postgres.vectorstores import PGVector
from langchain_openai import ChatOpenAI
from langchain.chains.summarize import load_summarize_chain
from langchain.prompts import PromptTemplate
import psycopg2
from src.utils.logger import Logger

# Logger 초기화
today = datetime.now().strftime("%Y%m%d")
time_now = datetime.now().strftime("%H%M%S")
experiment_name = "agent_summarize"
log_dir = f"experiments/{today}/{today}_{time_now}_{experiment_name}"
os.makedirs(log_dir, exist_ok=True)
logger = Logger(log_path=f"{log_dir}/experiment.log")

@tool
def summarize_paper(paper_title: str, difficulty: str = "easy") -> str:
    """
    특정 논문을 요약합니다. 난이도에 따라 초심자용/전문가용 요약을 제공합니다.

    Args:
        paper_title: 논문 제목
        difficulty: 'easy' (초심자) 또는 'hard' (전문가)

    Returns:
        논문 요약 내용
    """
    logger.write(f"논문 요약 시작: {paper_title}")
    logger.write(f"난이도: {difficulty}")

    # 1. PostgreSQL에서 논문 메타데이터 조회
    conn = psycopg2.connect("postgresql://user:password@localhost/papers")
    cursor = conn.cursor()

    cursor.execute(
        "SELECT * FROM papers WHERE title ILIKE %s",
        (f"%{paper_title}%",)
    )
    paper_meta = cursor.fetchone()

    if not paper_meta:
        logger.write(f"논문을 찾을 수 없음: {paper_title}")
        return f"'{paper_title}' 논문을 찾을 수 없습니다."

    paper_id = paper_meta[0]
    logger.write(f"논문 ID: {paper_id}")

    # 2. Vector DB에서 논문 전체 내용 조회
    vectorstore = PGVector(
        collection_name="paper_chunks",
        connection_string="postgresql://user:password@localhost:5432/papers"
    )

    paper_chunks = vectorstore.similarity_search(
        paper_title,
        k=10,
        filter={"paper_id": paper_id}
    )

    logger.write(f"검색된 청크 수: {len(paper_chunks)}")

    # 3. 난이도별 프롬프트
    if difficulty == "easy":
        prompt_template = """
다음 논문을 초심자도 이해할 수 있도록 쉽게 요약해주세요:
- 전문 용어는 풀어서 설명
- 핵심 아이디어 3가지
- 실생활 비유 포함

논문 내용: {text}

쉬운 요약:
        """
    else:  # hard
        prompt_template = """
다음 논문을 전문가 수준으로 요약해주세요:
- 기술적 세부사항 포함
- 수식 및 알고리즘 설명
- 관련 연구와의 비교

논문 내용: {text}

전문가용 요약:
        """

    PROMPT = PromptTemplate(template=prompt_template, input_variables=["text"])

    # 4. 요약 체인 실행
    llm = ChatOpenAI(model="gpt-4", temperature=0)
    chain = load_summarize_chain(llm, chain_type="stuff", prompt=PROMPT)

    logger.write("요약 체인 실행 중...")
    summary = chain.run(paper_chunks)

    logger.write(f"요약 완료: {len(summary)} 글자")

    return summary
```

---

## Agent 아키텍처 다이어그램

### 1. LangGraph Agent 구조

```mermaid
graph LR
    START([🔸 시작]) --> Router{라우터<br/>노드}

    Router -->|일반 질문| General[일반 답변]
    Router -->|논문 검색| RAG[RAG 검색]
    Router -->|웹 검색| Web[웹 검색]
    Router -->|용어 질문| Glossary[용어집]
    Router -->|요약 요청| Summarize[논문 요약]
    Router -->|저장 요청| Save[파일 저장]

    General --> END([✅ 종료])
    RAG --> END
    Web --> END
    Glossary --> END
    Summarize --> END
    Save --> END

    style START fill:#81c784,stroke:#388e3c,color:#000
    style END fill:#66bb6a,stroke:#2e7d32,color:#000
    style Router fill:#ba68c8,stroke:#7b1fa2,color:#000
    style General fill:#ce93d8,stroke:#7b1fa2,color:#000
    style RAG fill:#ce93d8,stroke:#7b1fa2,color:#000
    style Web fill:#ce93d8,stroke:#7b1fa2,color:#000
    style Glossary fill:#ce93d8,stroke:#7b1fa2,color:#000
    style Summarize fill:#ce93d8,stroke:#7b1fa2,color:#000
    style Save fill:#ce93d8,stroke:#7b1fa2,color:#000
```

### 2. LLM 선택 전략

```mermaid
graph TB
    A[작업 유형] --> B{작업 분류}

    B -->|라우팅| C[Solar<br/>빠른 응답]
    B -->|답변 생성| D[GPT-4<br/>높은 정확도]
    B -->|요약| E[GPT-4<br/>품질 중요]
    B -->|기타| F[GPT-3.5<br/>비용 효율]

    C --> G[LLM 호출]
    D --> G
    E --> G
    F --> G

    G --> H{에러?}
    H -->|Yes| I[재시도<br/>최대 3회]
    H -->|No| J[✅ 결과 반환]
    I --> G

    style A fill:#90caf9,stroke:#1976d2,color:#000
    style B fill:#ba68c8,stroke:#7b1fa2,color:#000
    style C fill:#ce93d8,stroke:#7b1fa2,color:#000
    style D fill:#ce93d8,stroke:#7b1fa2,color:#000
    style E fill:#ce93d8,stroke:#7b1fa2,color:#000
    style F fill:#ce93d8,stroke:#7b1fa2,color:#000
    style G fill:#a5d6a7,stroke:#388e3c,color:#000
    style H fill:#ba68c8,stroke:#7b1fa2
    style I fill:#ffcc80,stroke:#f57c00,color:#000
    style J fill:#66bb6a,stroke:#2e7d32,color:#000
```

### 3. 에러 핸들링 흐름

```mermaid
sequenceDiagram
    autonumber
    participant Agent
    participant LLM
    participant Retry

    Agent->>LLM: API 호출

    alt 성공
        LLM-->>Agent: ✅ 응답 반환
    else 실패 (1차)
        LLM-->>Retry: ❌ 에러
        Retry->>Retry: 대기 2초
        Retry->>LLM: 재시도 (1/3)

        alt 성공
            LLM-->>Agent: ✅ 응답 반환
        else 실패 (2차)
            LLM-->>Retry: ❌ 에러
            Retry->>Retry: 대기 4초
            Retry->>LLM: 재시도 (2/3)

            alt 성공
                LLM-->>Agent: ✅ 응답 반환
            else 실패 (3차)
                LLM-->>Retry: ❌ 에러
                Retry->>Retry: 대기 8초
                Retry->>LLM: 재시도 (3/3)

                alt 성공
                    LLM-->>Agent: ✅ 응답 반환
                else 최종 실패
                    LLM-->>Agent: ❌ 에러 반환
                end
            end
        end
    end
```

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

### 예제 코드

```python
# src/agent/graph.py

import os
from datetime import datetime
from langgraph.graph import StateGraph, END
from typing import TypedDict
from langchain_openai import ChatOpenAI
from src.utils.logger import Logger

# Logger 초기화
today = datetime.now().strftime("%Y%m%d")
time_now = datetime.now().strftime("%H%M%S")
experiment_name = "agent_router"
log_dir = f"experiments/{today}/{today}_{time_now}_{experiment_name}"
os.makedirs(log_dir, exist_ok=True)
logger = Logger(log_path=f"{log_dir}/experiment.log")

class AgentState(TypedDict):
    question: str
    difficulty: str
    tool_choice: str
    tool_result: str
    final_answer: str

def router_node(state: AgentState):
    """
    질문을 분석하여 어떤 도구를 사용할지 결정
    """
    question = state["question"]

    logger.write(f"라우터 노드 실행: {question}")

    # LLM에게 라우팅 결정 요청
    routing_prompt = f"""
사용자 질문을 분석하여 적절한 도구를 선택하세요:

도구 목록:
- search_paper: 논문 데이터베이스에서 검색
- web_search: 웹에서 최신 논문 검색
- glossary: 용어 정의 검색
- summarize: 논문 요약
- save_file: 파일 저장
- general: 일반 답변

질문: {question}

하나의 도구 이름만 반환하세요:
    """

    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    tool_choice = llm.invoke(routing_prompt).content.strip()

    logger.write(f"라우팅 결정: {tool_choice}")

    state["tool_choice"] = tool_choice
    return state

def route_to_tool(state: AgentState):
    """라우팅 결정에 따라 다음 노드 선택"""
    return state["tool_choice"]

def create_agent_graph():
    """LangGraph Agent 그래프 생성"""
    logger.write("Agent 그래프 생성 시작")

    workflow = StateGraph(AgentState)

    # 노드 추가
    workflow.add_node("router", router_node)
    workflow.add_node("search_paper", search_paper_node)
    workflow.add_node("web_search", web_search_node)
    workflow.add_node("glossary", glossary_node)
    workflow.add_node("summarize", summarize_node)
    workflow.add_node("save_file", save_file_node)
    workflow.add_node("general", general_answer_node)

    # 시작점 설정
    workflow.set_entry_point("router")

    # 조건부 엣지 설정
    workflow.add_conditional_edges(
        "router",
        route_to_tool,
        {
            "search_paper": "search_paper",
            "web_search": "web_search",
            "glossary": "glossary",
            "summarize": "summarize",
            "save_file": "save_file",
            "general": "general"
        }
    )

    # 모든 노드에서 종료
    for node in ["search_paper", "web_search", "glossary", "summarize", "save_file", "general"]:
        workflow.add_edge(node, END)

    # 그래프 컴파일
    agent_executor = workflow.compile()

    logger.write("Agent 그래프 컴파일 완료")

    return agent_executor
```

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

### 예제 코드

```python
# src/llm/client.py

import os
from datetime import datetime
from langchain_openai import ChatOpenAI
from langchain_upstage import ChatUpstage
from tenacity import retry, stop_after_attempt, wait_exponential
from langchain.callbacks import get_openai_callback
from src.utils.logger import Logger

# Logger 초기화
today = datetime.now().strftime("%Y%m%d")
time_now = datetime.now().strftime("%H%M%S")
experiment_name = "agent_llm"
log_dir = f"experiments/{today}/{today}_{time_now}_{experiment_name}"
os.makedirs(log_dir, exist_ok=True)
logger = Logger(log_path=f"{log_dir}/experiment.log")

class LLMClient:
    """다중 LLM 클라이언트 클래스"""

    def __init__(self, provider="openai", model="gpt-3.5-turbo", temperature=0.7):
        """
        Args:
            provider: "openai" 또는 "solar"
            model: 모델 이름
            temperature: 창의성 수준 (0-1)
        """
        self.provider = provider

        logger.write(f"LLM 초기화: provider={provider}, model={model}")

        if provider == "openai":
            self.llm = ChatOpenAI(
                model=model,
                temperature=temperature,
                openai_api_key=os.getenv("OPENAI_API_KEY"),
                streaming=True
            )
        elif provider == "solar":
            self.llm = ChatUpstage(
                model="solar-1-mini-chat",
                temperature=temperature,
                api_key=os.getenv("UPSTAGE_API_KEY"),
                streaming=True
            )

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=2, min=2, max=8))
    def invoke_with_retry(self, messages):
        """
        에러 핸들링 및 재시도
        최대 3회 재시도, 지수 백오프 (2초 → 4초 → 8초)
        """
        logger.write("LLM 호출 시작 (재시도 가능)")
        return self.llm.invoke(messages)

    def invoke_with_tracking(self, messages):
        """토큰 사용량 추적"""
        if self.provider == "openai":
            with get_openai_callback() as cb:
                response = self.llm.invoke(messages)
                logger.write(f"Tokens Used: {cb.total_tokens}")
                logger.write(f"Total Cost: ${cb.total_cost:.4f}")
                return response
        else:
            return self.llm.invoke(messages)

    async def astream(self, messages):
        """스트리밍 응답 처리"""
        logger.write("스트리밍 응답 시작")
        async for chunk in self.llm.astream(messages):
            yield chunk


def get_llm_for_task(task_type):
    """작업 유형별 최적 LLM 선택"""
    logger.write(f"작업 유형별 LLM 선택: {task_type}")

    if task_type == "routing":
        return LLMClient(provider="solar", model="solar-1-mini-chat", temperature=0)
    elif task_type == "generation":
        return LLMClient(provider="openai", model="gpt-4", temperature=0.7)
    elif task_type == "summarization":
        return LLMClient(provider="openai", model="gpt-4", temperature=0)
    else:
        return LLMClient(provider="openai", model="gpt-3.5-turbo", temperature=0.7)
```

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

### 예제 코드

```python
# src/memory/chat_history.py

from langchain.memory import ConversationBufferMemory
from langchain_postgres import PostgresChatMessageHistory
import os

class ChatMemoryManager:
    """대화 메모리 관리 클래스"""

    def __init__(self):
        """ConversationBufferMemory 초기화"""
        self.memory = ConversationBufferMemory(
            return_messages=True,
            memory_key="chat_history"
        )

    def add_user_message(self, message: str):
        """사용자 메시지 추가"""
        self.memory.chat_memory.add_user_message(message)

    def add_ai_message(self, message: str):
        """AI 메시지 추가"""
        self.memory.chat_memory.add_ai_message(message)

    def get_history(self):
        """전체 대화 히스토리 반환"""
        return self.memory.load_memory_variables({})

    def clear(self):
        """대화 히스토리 초기화"""
        self.memory.clear()


def get_session_history(session_id: str):
    """
    세션 기반 메모리 (PostgreSQL 저장)

    Args:
        session_id: 세션 ID

    Returns:
        PostgresChatMessageHistory 인스턴스
    """
    connection_string = os.getenv("DATABASE_URL", "postgresql://user:password@localhost:5432/papers")

    return PostgresChatMessageHistory(
        session_id=session_id,
        connection_string=connection_string,
        table_name="chat_history"
    )


# 사용 예시
if __name__ == "__main__":
    # 기본 메모리 사용
    memory_manager = ChatMemoryManager()

    memory_manager.add_user_message("Transformer 논문 설명해줘")
    memory_manager.add_ai_message("Transformer는 2017년 Google에서 발표한...")

    logger.write(f"메모리 히스토리: {memory_manager.get_history()}")

    # 세션 기반 메모리 사용
    session_history = get_session_history("user_123")
    session_history.add_user_message("BERT 논문은?")
    session_history.add_ai_message("BERT는 2018년에...")

    logger.write(f"세션 메시지: {session_history.messages}")
    logger.close()
```

---

## 로깅 및 실험 추적 관리

### 로깅 시스템 사용

**중요**: 모든 출력은 Logger 클래스를 사용해야 합니다.

**파일 경로**: `src/utils/logger.py`

**사용 방법**:
1. 실험 폴더 및 Logger 생성
   ```python
   today = datetime.now().strftime("%Y%m%d")
   time_now = datetime.now().strftime("%H%M%S")
   experiment_name = "agent_main"  # agent_xxx, rag_xxx, feature_xxx
   log_dir = f"experiments/{today}/{today}_{time_now}_{experiment_name}"
   os.makedirs(log_dir, exist_ok=True)
   logger = Logger(log_path=f"{log_dir}/experiment.log")
   ```

2. 로그 기록
   - `logger.write()` 사용 (print() 대신)
   - 예: `logger.write(f"라우팅 결정: {tool_choice}")`

3. 실험 종료
   - `logger.close()` 필수 호출

### 실험 폴더 구조

**규칙**: PRD 문서 06_실험_추적_관리.md 참조

```
experiments/
├── {날짜}/
│   ├── {날짜}_{시간}_{실험명}/
│   │   ├── experiment.log         # 실험 로그
│   │   ├── config.yaml            # 설정 파일
│   │   └── results.json           # 결과 파일
```

**필수 파일**:
- `experiment.log`: logger.write() 출력
- `config.yaml`: LLM 설정, 난이도, 모델 정보
- `results.json`: 최종 답변, 도구 선택, 응답 시간

### 예제 코드

```python
import os
from datetime import datetime
from src.utils.logger import Logger
import yaml
import json

# 실험 폴더 생성
today = datetime.now().strftime("%Y%m%d")
time_now = datetime.now().strftime("%H%M%S")
experiment_name = "agent_main"
log_dir = f"experiments/{today}/{today}_{time_now}_{experiment_name}"
os.makedirs(log_dir, exist_ok=True)

# Logger 초기화
logger = Logger(log_path=f"{log_dir}/experiment.log")

# Config 저장
config = {
    "llm_provider": "openai",
    "model": "gpt-4",
    "temperature": 0.7,
    "difficulty": "easy"
}

with open(f"{log_dir}/config.yaml", "w") as f:
    yaml.dump(config, f)

# 실행 로그
logger.write("Agent 실행 시작")
logger.write(f"질문: {question}")
logger.write(f"선택된 도구: {tool_choice}")

# Results 저장
results = {
    "question": question,
    "tool_choice": tool_choice,
    "final_answer": final_answer,
    "response_time_ms": 1250,
    "timestamp": datetime.now().isoformat()
}

with open(f"{log_dir}/results.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

# Logger 종료
logger.close()
```

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

**3단계: AI Agent 메인 구현 (최현화)**
- `3-1. feature/llm-client` - LLM 클라이언트 구현
- `3-2. feature/memory` - 대화 메모리 시스템
- `3-3. feature/agent-graph` - LangGraph 그래프 구현
- `3-4. feature/tool-summarize` - 논문 요약 도구
- `3-5. feature/integration` - 통합 및 main.py

---

## 참고 PRD 문서

개발 시 반드시 참고해야 할 PRD 문서 목록:

### 필수 참고 문서
1. [01_프로젝트_개요.md](../PRD/01_프로젝트_개요.md) - 프로젝트 전체 개요 및 목표
2. [02_프로젝트_구조.md](../PRD/02_프로젝트_구조.md) - 폴더 구조 및 모듈 배치
3. [05_로깅_시스템.md](../PRD/05_로깅_시스템.md) ⭐ - Logger 클래스 사용법 및 규칙
4. [06_실험_추적_관리.md](../PRD/06_실험_추적_관리.md) ⭐ - 실험 폴더 구조 및 명명 규칙
5. [10_기술_요구사항.md](../PRD/10_기술_요구사항.md) - 기술 스택 및 라이브러리
6. [12_AI_Agent_설계.md](../PRD/12_AI_Agent_설계.md) - LangGraph 구조 및 도구 정의
7. [14_LLM_설정.md](../PRD/14_LLM_설정.md) - LLM 선택 전략 및 에러 핸들링

### 참고 문서
- [03_브랜치_전략.md](../PRD/03_브랜치_전략.md) - Feature 브랜치 전략
- [04_일정_관리.md](../PRD/04_일정_관리.md) - 개발 일정 및 마일스톤
- [11_데이터베이스_설계.md](../PRD/11_데이터베이스_설계.md) - DB 스키마 (요약 도구에서 사용)

---

## 참고 자료

- LangGraph 공식 문서: https://langchain-ai.github.io/langgraph/
- Langchain ChatOpenAI: https://python.langchain.com/docs/integrations/chat/openai/
- Langchain Memory: https://python.langchain.com/docs/modules/memory/
- Langchain Summarization: https://python.langchain.com/docs/use_cases/summarization/
- Langchain Callbacks: https://python.langchain.com/docs/modules/callbacks/
