# 담당역할: 최현화 - AI Agent 메인

## 담당자 정보
- **이름**: 최현화
- **역할**: 팀장
- **참여 기간**: 10/28 ~ 11/6 (전체 기간)
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

### Langchain 구현

#### 1. LLM 직접 호출
```python
# src/agent/nodes.py

from langchain_openai import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage

def general_answer_node(state: AgentState):
    """
    일반 질문에 LLM이 직접 답변
    """
    question = state["question"]
    difficulty = state.get("difficulty", "easy")

    # 난이도별 시스템 메시지
    if difficulty == "easy":
        system_msg = SystemMessage(content="""
        당신은 친절한 AI 어시스턴트입니다.
        쉽고 이해하기 쉬운 언어로 답변해주세요.
        """)
    else:
        system_msg = SystemMessage(content="""
        당신은 전문적인 AI 어시스턴트입니다.
        기술적이고 정확한 언어로 답변해주세요.
        """)

    # LLM 호출
    response = llm.invoke([
        system_msg,
        HumanMessage(content=question)
    ])

    state["final_answer"] = response.content
    return state
```

#### 2. 라우터 노드에서 일반 답변 판단
```python
def router_node(state: AgentState):
    """
    질문을 분석하여 어떤 도구를 사용할지 결정
    """
    question = state["question"]

    # LLM에게 라우팅 결정 요청
    routing_prompt = f"""
    사용자 질문을 분석하여 적절한 도구를 선택하세요:

    질문: {question}

    도구 목록:
    - general: 일반 인사, 상식 질문 (도구 불필요)
    - search_paper: 논문 데이터베이스 검색
    - web_search: 웹 검색
    - search_glossary: 용어 정의 검색
    - summarize_paper: 논문 요약
    - save_file: 파일 저장

    선택할 도구 (하나만):
    """

    tool_choice = llm.invoke(routing_prompt).content.strip()
    state["tool_choice"] = tool_choice

    return state
```

### 사용하는 DB
**DB 사용 없음** (LLM 자체 지식 활용)

---

## 도구 2: 논문 요약 도구

### 기능 설명
특정 논문의 전체 내용을 난이도별(Easy/Hard)로 요약하는 도구

### Langchain 구현

#### 1. 논문 검색 및 전체 내용 조회
```python
# src/tools/summarize.py

from langchain.tools import tool
from langchain.chains.summarize import load_summarize_chain
from langchain.prompts import PromptTemplate
import psycopg2

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
    # 1. PostgreSQL에서 논문 메타데이터 조회
    conn = psycopg2.connect("postgresql://user:password@localhost/papers")
    cursor = conn.cursor()

    cursor.execute(
        "SELECT * FROM papers WHERE title ILIKE %s",
        (f"%{paper_title}%",)
    )
    paper_meta = cursor.fetchone()

    if not paper_meta:
        return f"논문 '{paper_title}'을 찾을 수 없습니다."

    paper_id = paper_meta[0]

    # 2. Vector DB에서 논문 전체 내용 조회 (여러 청크)
    paper_chunks = vectorstore.similarity_search(
        paper_title,
        k=10,  # 여러 청크 가져오기
        filter={"paper_id": paper_id}
    )

    # 3. 난이도별 요약 체인 실행
    if difficulty == "easy":
        summary = easy_summarize_chain.run(paper_chunks)
    else:
        summary = hard_summarize_chain.run(paper_chunks)

    return summary
```

#### 2. Langchain Summarization Chain 구현
```python
# src/llm/chains.py

from langchain.chains.summarize import load_summarize_chain
from langchain.prompts import PromptTemplate

# Easy 모드 요약 프롬프트
EASY_SUMMARY_PROMPT = PromptTemplate(
    template="""
    다음 논문 내용을 초심자도 이해할 수 있도록 쉽게 요약해주세요:

    요약 규칙:
    1. 전문 용어는 쉬운 말로 풀어서 설명
    2. 핵심 아이디어 3가지 이내로 요약
    3. 실생활 비유 포함
    4. 수식은 최소화

    논문 내용:
    {text}

    요약:
    """,
    input_variables=["text"]
)

# Hard 모드 요약 프롬프트
HARD_SUMMARY_PROMPT = PromptTemplate(
    template="""
    다음 논문 내용을 전문가 수준으로 요약해주세요:

    요약 규칙:
    1. 기술적 세부사항 포함
    2. 수식 및 알고리즘 설명
    3. 관련 연구와의 비교
    4. 주요 기여도 명확히
    5. 한계점 및 향후 연구 방향

    논문 내용:
    {text}

    요약:
    """,
    input_variables=["text"]
)

# Easy 모드 요약 체인 (짧은 논문용)
easy_summarize_chain = load_summarize_chain(
    llm=llm,
    chain_type="stuff",  # 모든 청크를 한 번에 LLM에 전달
    prompt=EASY_SUMMARY_PROMPT
)

# Hard 모드 요약 체인 (긴 논문용)
hard_summarize_chain = load_summarize_chain(
    llm=llm,
    chain_type="map_reduce",  # 각 청크를 요약 후 최종 통합
    map_prompt=HARD_SUMMARY_PROMPT,
    combine_prompt=PromptTemplate(
        template="""
        다음은 논문의 각 부분을 요약한 내용입니다. 이를 종합하여 전체 논문 요약을 작성해주세요:

        {text}

        종합 요약:
        """,
        input_variables=["text"]
    )
)
```

#### 3. 요약 방식 선택
```python
def get_summarize_chain(difficulty: str, paper_length: int):
    """
    논문 길이와 난이도에 따라 적절한 요약 체인 선택

    Args:
        difficulty: 'easy' 또는 'hard'
        paper_length: 논문 청크 수

    Returns:
        적절한 요약 체인
    """
    if paper_length <= 5:
        # 짧은 논문: stuff 방식
        chain_type = "stuff"
    elif paper_length <= 15:
        # 중간 논문: map_reduce 방식
        chain_type = "map_reduce"
    else:
        # 긴 논문: refine 방식 (순차적 요약)
        chain_type = "refine"

    prompt = EASY_SUMMARY_PROMPT if difficulty == "easy" else HARD_SUMMARY_PROMPT

    return load_summarize_chain(
        llm=llm,
        chain_type=chain_type,
        prompt=prompt
    )
```

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

### 1. State 정의
```python
# src/agent/state.py

from typing import TypedDict, Annotated, Sequence
from langchain.schema import BaseMessage
import operator

class AgentState(TypedDict):
    """Agent 상태 정의"""
    question: str  # 사용자 질문
    difficulty: str  # 난이도 (easy/hard)
    tool_choice: str  # 선택된 도구
    tool_result: str  # 도구 실행 결과
    final_answer: str  # 최종 답변
    messages: Annotated[Sequence[BaseMessage], operator.add]  # 대화 히스토리
```

### 2. 그래프 구성
```python
# src/agent/graph.py

from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI

def create_agent_graph():
    """AI Agent 그래프 생성"""
    workflow = StateGraph(AgentState)

    # 노드 추가
    workflow.add_node("router", router_node)
    workflow.add_node("general", general_answer_node)
    workflow.add_node("search_paper", search_paper_node)
    workflow.add_node("web_search", web_search_node)
    workflow.add_node("search_glossary", glossary_node)
    workflow.add_node("summarize_paper", summarize_node)
    workflow.add_node("save_file", save_file_node)

    # 시작점 설정
    workflow.set_entry_point("router")

    # 조건부 엣지 (라우터에서 도구 선택)
    workflow.add_conditional_edges(
        "router",
        route_to_tool,
        {
            "general": "general",
            "search_paper": "search_paper",
            "web_search": "web_search",
            "search_glossary": "search_glossary",
            "summarize_paper": "summarize_paper",
            "save_file": "save_file"
        }
    )

    # 모든 도구 노드에서 종료
    workflow.add_edge("general", END)
    workflow.add_edge("search_paper", END)
    workflow.add_edge("web_search", END)
    workflow.add_edge("search_glossary", END)
    workflow.add_edge("summarize_paper", END)
    workflow.add_edge("save_file", END)

    # 그래프 컴파일
    return workflow.compile()

def route_to_tool(state: AgentState) -> str:
    """라우팅 결정에 따라 다음 노드 선택"""
    return state["tool_choice"]
```

### 3. 라우터 노드 (핵심)
```python
def router_node(state: AgentState):
    """
    질문을 분석하여 어떤 도구를 사용할지 결정
    LLM의 Function Calling 기능 활용
    """
    question = state["question"]

    # LLM에게 라우팅 결정 요청
    routing_prompt = f"""
    사용자 질문을 분석하여 가장 적절한 도구를 선택하세요.

    질문: {question}

    도구 목록:
    1. general - 일반 인사, 상식 질문 (예: "안녕하세요", "고마워")
    2. search_paper - 논문 데이터베이스 검색 (예: "Transformer 논문 설명해줘")
    3. web_search - 최신 정보 웹 검색 (예: "2025년 최신 LLM 논문은?")
    4. search_glossary - 용어 정의 검색 (예: "Attention이 뭐야?")
    5. summarize_paper - 논문 요약 (예: "BERT 논문 요약해줘")
    6. save_file - 파일 저장 (예: "이 내용 저장해줘")

    선택할 도구 (하나만, 도구 이름만 출력):
    """

    response = llm.invoke([
        SystemMessage(content="당신은 질문 분석 전문가입니다."),
        HumanMessage(content=routing_prompt)
    ])

    tool_choice = response.content.strip()
    state["tool_choice"] = tool_choice

    print(f"[Router] 질문: {question}")
    print(f"[Router] 선택된 도구: {tool_choice}")

    return state
```

---

## LLM 클라이언트 구현

### 1. 다중 LLM 클라이언트 (OpenAI + Solar)
```python
# src/llm/client.py

from langchain_openai import ChatOpenAI
from langchain_upstage import ChatUpstage
from langchain.callbacks import get_openai_callback
from tenacity import retry, stop_after_attempt, wait_exponential
import os

class LLMClient:
    """다중 LLM 클라이언트 래퍼 (OpenAI + Solar)"""

    def __init__(self, provider: str = "openai", model: str = "gpt-4", temperature: float = 0.7):
        """
        Args:
            provider: "openai" 또는 "solar"
            model: 모델 이름
            temperature: 온도 (0~1)
        """
        self.provider = provider

        if provider == "openai":
            self.llm = ChatOpenAI(
                model=model,
                temperature=temperature,
                openai_api_key=os.getenv("OPENAI_API_KEY"),
                streaming=True
            )
        elif provider == "solar":
            self.llm = ChatUpstage(
                model="solar-1-mini-chat",  # 또는 "solar-pro"
                temperature=temperature,
                upstage_api_key=os.getenv("UPSTAGE_API_KEY"),
                streaming=True
            )
        else:
            raise ValueError(f"Unsupported provider: {provider}")

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def invoke_with_retry(self, messages):
        """재시도 로직을 포함한 LLM 호출"""
        return self.llm.invoke(messages)

    def invoke_with_tracking(self, messages):
        """토큰 사용량 추적을 포함한 LLM 호출"""
        if self.provider == "openai":
            with get_openai_callback() as cb:
                response = self.llm.invoke(messages)
                print(f"[OpenAI] Tokens: {cb.total_tokens}, Cost: ${cb.total_cost:.4f}")
        else:
            # Solar는 별도 추적
            response = self.llm.invoke(messages)
            print(f"[Solar] Response generated")

        return response

    async def astream(self, messages):
        """스트리밍 응답"""
        async for chunk in self.llm.astream(messages):
            yield chunk


# LLM 선택 전략
def get_llm_for_task(task_type: str):
    """
    작업 유형에 따라 적절한 LLM 선택

    Args:
        task_type: "routing", "generation", "summarization" 등

    Returns:
        LLMClient 인스턴스
    """
    if task_type == "routing":
        # 라우팅은 빠른 Solar 사용
        return LLMClient(provider="solar", model="solar-1-mini-chat", temperature=0)
    elif task_type == "generation":
        # 답변 생성은 정확도 높은 GPT-4 사용
        return LLMClient(provider="openai", model="gpt-4", temperature=0.7)
    elif task_type == "summarization":
        # 요약은 GPT-4 사용
        return LLMClient(provider="openai", model="gpt-4", temperature=0.5)
    else:
        # 기본값: OpenAI GPT-3.5
        return LLMClient(provider="openai", model="gpt-3.5-turbo", temperature=0.7)
```

### 2. 스트리밍 응답 처리
```python
async def stream_response(agent, question, difficulty):
    """Agent 실행 결과를 스트리밍으로 반환"""
    async for event in agent.astream_events(
        {"question": question, "difficulty": difficulty}
    ):
        if event["event"] == "on_chat_model_stream":
            chunk = event["data"]["chunk"]
            yield chunk.content
```

---

## 대화 메모리 시스템

### 1. ConversationBufferMemory 구현
```python
# src/memory/chat_history.py

from langchain.memory import ConversationBufferMemory
from langchain.schema import ChatMessageHistory

class ChatMemoryManager:
    """대화 히스토리 관리"""

    def __init__(self):
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
        """대화 히스토리 조회"""
        return self.memory.chat_memory.messages

    def clear(self):
        """대화 히스토리 초기화"""
        self.memory.clear()
```

### 2. 세션 관리 (PostgreSQL)
```python
from langchain.memory.chat_message_histories import PostgresChatMessageHistory

def get_session_history(session_id: str):
    """세션 ID로 대화 히스토리 조회"""
    return PostgresChatMessageHistory(
        connection_string="postgresql://user:password@localhost/papers",
        session_id=session_id
    )
```

---

## 개발 일정

### Phase 1: LLM 클라이언트 및 공통 인프라 (10/28~10/29)
- ChatOpenAI 래퍼 구현
- 에러 핸들링 및 재시도 로직
- 토큰 사용량 추적
- 스트리밍 응답 처리

### Phase 2: LangGraph Agent 그래프 (10/30~11/01)
- State 정의
- 라우터 노드 구현
- 조건부 엣지 설정
- 일반 답변 노드 구현

### Phase 3: 메모리 시스템 (11/01~11/02)
- ConversationBufferMemory 구현
- 대화 히스토리 관리
- 세션 관리

### Phase 4: 논문 요약 도구 (11/02~11/03)
- load_summarize_chain 구현
- 난이도별 프롬프트 설계
- 요약 방식 선택 로직

### Phase 5: 통합 작업 (11/04~11/05)
- main.py 작성
- 모든 모듈 통합
- 디버깅 및 테스트

### Phase 6: 발표 준비 (11/05~11/06)
- 발표 자료 작성
- README.md 작성
- 최종 점검

---

## main.py 구현

```python
# main.py

from src.agent.graph import create_agent_graph
from src.llm.client import LLMClient
from src.memory.chat_history import ChatMemoryManager

def main():
    """메인 실행 함수"""

    # LLM 클라이언트 초기화
    llm_client = LLMClient(model="gpt-4", temperature=0.7)

    # Agent 그래프 생성
    agent = create_agent_graph()

    # 메모리 관리자 초기화
    memory_manager = ChatMemoryManager()

    # 테스트 질문
    test_questions = [
        ("안녕하세요", "easy"),
        ("Transformer 논문 설명해줘", "easy"),
        ("Attention Mechanism이 뭐야?", "easy"),
        ("BERT 논문 요약해줘", "hard")
    ]

    for question, difficulty in test_questions:
        print(f"\n[질문] {question} (난이도: {difficulty})")

        # Agent 실행
        result = agent.invoke({
            "question": question,
            "difficulty": difficulty,
            "messages": memory_manager.get_history()
        })

        # 메모리에 추가
        memory_manager.add_user_message(question)
        memory_manager.add_ai_message(result["final_answer"])

        print(f"[답변] {result['final_answer']}")

if __name__ == "__main__":
    main()
```

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
