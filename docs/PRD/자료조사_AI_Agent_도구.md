# 자료조사: AI Agent 도구 (Tools)

## 문서 정보
- **작성일**: 2025-10-29
- **프로젝트**: 논문 리뷰 챗봇 (AI Agent + RAG)
- **팀명**: 연결의 민족

---

## 1. AI Agent 도구(Tool)의 개념

### 1.1 도구란?

**도구(Tool)**는 AI Agent가 특정 작업을 수행하기 위해 호출할 수 있는 **함수**입니다.

- LLM은 사용자 질문을 분석하여 어떤 도구를 사용할지 판단
- 필요한 도구의 **arguments(매개변수)**를 추출
- 도구를 실행하고 결과를 받아 최종 답변 생성

### 1.2 질문: "일반 답변, RAG 검색, 웹 검색 분기"가 도구인가?

**답변: 네, 맞습니다!**

- "일반 답변으로 바로 답변할지"
- "RAG의 지식 베이스에서 찾아 답변할지"
- "웹 검색을 할지"

이러한 **분기 결정 자체가 AI Agent의 핵심 기능**이며, 각 분기는 **도구(Tool)**로 구현됩니다.

#### 예시: 라우팅 도구

```python
# 도구 1: 일반 답변 (LLM의 자체 지식 활용)
def general_answer(question: str) -> str:
    """
    간단한 인사, 일반 상식 질문에 LLM이 직접 답변
    """
    return llm.invoke(question)

# 도구 2: RAG 검색 (논문 데이터베이스 검색)
def search_papers(query: str) -> str:
    """
    논문 데이터베이스에서 관련 논문을 검색하여 답변
    """
    docs = vectorstore.similarity_search(query, k=3)
    context = "\n".join([doc.page_content for doc in docs])
    return llm.invoke(f"Context: {context}\n\nQuestion: {query}")

# 도구 3: 웹 검색 (최신 논문 검색)
def web_search(query: str) -> str:
    """
    웹에서 최신 논문을 검색
    """
    results = tavily_search.run(query)
    return results
```

**LLM이 자동으로 판단:**
- "안녕하세요" → `general_answer` 도구 사용
- "Transformer 논문 설명해줘" → `search_papers` 도구 사용
- "2025년 최신 LLM 논문은?" → `web_search` 도구 사용

---

## 2. OT 자료 요구사항 분석

### 2.1 필수 도구 (OT 자료 기준)

1. **웹 검색 기능** (필수)
2. **파일 저장 기능** (필수)
3. **3개 이상의 도구 사용** (필수)
   - 직접 구현한 도구 포함 필요
   - 특정 페르소나에 맞는 도구 (논문 리뷰 챗봇 → 논문 관련 도구)

### 2.2 선택 기능 (가산점)

1. **RAG 기능** (Vector DB, RDB 기반)
2. **성능 평가 기능**
3. **노드/엣지 추가** (LangGraph 활용)

---

## 3. 논문 리뷰 챗봇을 위한 5가지 도구 추천

### 도구 1: RAG 논문 검색 도구 (Paper Search Tool) ★ 직접 구현 필수

**기능:** 로컬 데이터베이스에 저장된 논문을 검색

**사용 시점:**
- "Transformer 논문 설명해줘"
- "BERT와 GPT의 차이점은?"
- "Attention 메커니즘이 뭐야?"

**구현 예시:**
```python
from langchain.tools import tool
from langchain.vectorstores import Chroma

@tool
def search_paper_database(query: str) -> str:
    """
    논문 데이터베이스에서 관련 논문을 검색합니다.

    Args:
        query: 검색할 질문 또는 키워드

    Returns:
        관련 논문 내용 및 메타데이터
    """
    # Vector DB에서 유사도 검색
    docs = vectorstore.similarity_search(query, k=3)

    results = []
    for doc in docs:
        results.append({
            "content": doc.page_content,
            "title": doc.metadata.get("title"),
            "authors": doc.metadata.get("authors"),
            "year": doc.metadata.get("year")
        })

    return format_paper_results(results)
```

**DB 연동:**
- Vector DB (ChromaDB/pgvector): 논문 본문 임베딩 검색
- PostgreSQL: 논문 메타데이터 (제목, 저자, 년도 등) 조회

---

### 도구 2: 웹 검색 도구 (Web Search Tool) ★ 필수

**기능:** 최신 논문 정보를 웹에서 검색

**사용 시점:**
- "2025년 최신 LLM 논문은?"
- "GPT-5가 나왔어?"
- "오늘 arXiv에 올라온 논문 찾아줘"

**구현 예시:**
```python
from langchain.tools import TavilySearchResults

# Tavily Search API 사용 (Langchain 공식 추천)
web_search_tool = TavilySearchResults(
    max_results=5,
    search_depth="advanced",
    include_answer=True,
    include_raw_content=True
)
```

**또는 DuckDuckGo (무료):**
```python
from langchain.tools import DuckDuckGoSearchRun

web_search_tool = DuckDuckGoSearchRun()
```

**특화 검색:**
- arXiv API 활용: 논문 전문 검색 사이트에 특화된 검색
- Google Scholar API: 학술 논문 전용 검색

---

### 도구 3: 논문 용어집 검색 도구 (Glossary Search Tool) ★ 직접 구현

**기능:** 논문에 자주 등장하는 전문 용어 설명

**사용 시점:**
- "Attention이 뭐야?"
- "Fine-tuning이란?"
- "BLEU 스코어 설명해줘"

**구현 예시:**
```python
@tool
def search_glossary(term: str) -> str:
    """
    논문 용어집에서 전문 용어를 검색하여 설명합니다.

    Args:
        term: 검색할 용어

    Returns:
        용어 정의 및 설명
    """
    # PostgreSQL glossary 테이블에서 검색
    result = db.execute(
        "SELECT term, definition, category FROM glossary WHERE term ILIKE %s",
        (f"%{term}%",)
    ).fetchone()

    if result:
        return f"**{result['term']}**: {result['definition']} (카테고리: {result['category']})"
    else:
        # 용어집에 없으면 RAG로 논문에서 검색
        return search_paper_database(f"{term} 정의")
```

**용어집 RAG 활용:**
- 용어집 데이터를 Vector DB에도 임베딩 저장
- 사용자 질문에 용어가 포함되어 있으면 자동으로 컨텍스트에 추가

---

### 도구 4: 논문 요약 도구 (Paper Summarization Tool) ★ 직접 구현

**기능:** 특정 논문의 전체 내용을 요약

**사용 시점:**
- "Attention is All You Need 논문 요약해줘"
- "BERT 논문의 핵심 내용은?"
- "이 논문의 주요 기여도는 뭐야?"

**구현 예시:**
```python
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
    paper_meta = db.execute(
        "SELECT * FROM papers WHERE title ILIKE %s",
        (f"%{paper_title}%",)
    ).fetchone()

    # 2. Vector DB에서 논문 전체 내용 조회
    paper_chunks = vectorstore.similarity_search(
        paper_title,
        k=10,  # 여러 청크 가져오기
        filter={"paper_id": paper_meta["paper_id"]}
    )

    full_content = "\n".join([chunk.page_content for chunk in paper_chunks])

    # 3. 난이도별 프롬프트
    if difficulty == "easy":
        prompt = f"""
        다음 논문을 초심자도 이해할 수 있도록 쉽게 요약해주세요:
        - 전문 용어는 풀어서 설명
        - 핵심 아이디어 3가지
        - 실생활 비유 포함

        논문 내용: {full_content}
        """
    else:  # hard
        prompt = f"""
        다음 논문을 전문가 수준으로 요약해주세요:
        - 기술적 세부사항 포함
        - 수식 및 알고리즘 설명
        - 관련 연구와의 비교

        논문 내용: {full_content}
        """

    return llm.invoke(prompt)
```

---

### 도구 5: 파일 저장 도구 (Save to File Tool) ★ 필수

**기능:** 대화 내용, 논문 요약, 참고 자료를 파일로 저장

**사용 시점:**
- "이 요약 내용 파일로 저장해줘"
- "오늘 대화 내용 저장하고 싶어"
- "찾은 논문 리스트 파일로 만들어줘"

**구현 예시:**
```python
import os
from datetime import datetime

@tool
def save_to_file(content: str, filename: str = None) -> str:
    """
    내용을 텍스트 파일로 저장합니다.

    Args:
        content: 저장할 내용
        filename: 파일명 (선택, 없으면 자동 생성)

    Returns:
        저장된 파일 경로
    """
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"paper_review_{timestamp}.txt"

    # data/outputs 폴더에 저장
    output_dir = "data/outputs"
    os.makedirs(output_dir, exist_ok=True)

    filepath = os.path.join(output_dir, filename)

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(content)

    return f"파일이 저장되었습니다: {filepath}"
```

**추가 기능:**
- PDF 형식으로 저장 (reportlab 라이브러리 사용)
- Markdown 형식으로 저장
- 논문 인용 형식 포함 (APA, MLA 등)

---

## 4. 추가 도구 (선택 사항)

### 도구 6: Text-to-SQL 도구 (DB Query Tool)

**기능:** 자연어를 SQL 쿼리로 변환하여 논문 통계 조회

**사용 시점:**
- "2024년에 발표된 논문 개수는?"
- "가장 많이 인용된 논문 Top 5는?"
- "저자별 논문 수 알려줘"

**구현 예시:**
```python
@tool
def query_paper_statistics(question: str) -> str:
    """
    논문 데이터베이스에서 통계 정보를 조회합니다.

    Args:
        question: 자연어 질문

    Returns:
        쿼리 결과
    """
    # LLM을 사용해 SQL 쿼리 생성
    sql_prompt = f"""
    다음 질문을 SQL 쿼리로 변환하세요.

    테이블 스키마:
    papers (paper_id, title, authors, publish_date, citation_count, category)

    질문: {question}

    SQL:
    """

    sql_query = llm.invoke(sql_prompt)

    # 쿼리 실행
    results = db.execute(sql_query).fetchall()

    return format_query_results(results)
```

---

### 도구 7: 논문 비교 도구 (Paper Comparison Tool)

**기능:** 여러 논문의 차이점 비교

**사용 시점:**
- "BERT와 GPT 비교해줘"
- "Transformer와 RNN의 차이는?"

---

### 도구 8: 인용 추출 도구 (Citation Extraction Tool)

**기능:** 논문의 주요 인용문 추출

---

## 5. 도구 라우팅: LLM이 자동으로 도구 선택

### 5.1 라우팅 메커니즘

LLM은 사용자 질문을 분석하여 적절한 도구를 선택합니다.

**예시:**

| 사용자 질문 | LLM이 선택하는 도구 | 이유 |
|-------------|---------------------|------|
| "안녕하세요" | 일반 답변 (도구 사용 안 함) | 간단한 인사 |
| "Transformer 논문 설명해줘" | `search_paper_database` | 로컬 DB에 있는 논문 검색 |
| "2025년 최신 LLM 논문은?" | `web_search_tool` | 최신 정보는 웹 검색 필요 |
| "Attention이 뭐야?" | `search_glossary` | 용어 정의 질문 |
| "BERT 논문 요약해줘" | `summarize_paper` | 특정 논문 요약 요청 |
| "이 내용 저장해줘" | `save_to_file` | 파일 저장 요청 |

### 5.2 LangGraph를 활용한 라우팅 구현

```python
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated, Sequence
import operator

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    next_action: str

def router_node(state: AgentState):
    """
    사용자 질문을 분석하여 어떤 도구를 사용할지 결정
    """
    last_message = state["messages"][-1]

    # LLM에게 라우팅 결정 요청
    routing_prompt = f"""
    사용자 질문을 분석하여 적절한 도구를 선택하세요:

    도구 목록:
    - search_paper_database: 로컬 논문 DB 검색
    - web_search: 웹에서 최신 논문 검색
    - search_glossary: 용어 정의 검색
    - summarize_paper: 논문 요약
    - save_to_file: 파일 저장
    - general_answer: 도구 불필요 (직접 답변)

    질문: {last_message.content}

    선택된 도구:
    """

    tool_choice = llm.invoke(routing_prompt)
    return {"next_action": tool_choice}

def conditional_edge(state: AgentState):
    """
    라우팅 결정에 따라 다음 노드 선택
    """
    return state["next_action"]

# 그래프 구성
workflow = StateGraph(AgentState)

workflow.add_node("router", router_node)
workflow.add_node("search_paper", search_paper_database_node)
workflow.add_node("web_search", web_search_node)
workflow.add_node("glossary", search_glossary_node)
workflow.add_node("summarize", summarize_paper_node)
workflow.add_node("save_file", save_to_file_node)
workflow.add_node("general", general_answer_node)

workflow.set_entry_point("router")

workflow.add_conditional_edges(
    "router",
    conditional_edge,
    {
        "search_paper_database": "search_paper",
        "web_search": "web_search",
        "search_glossary": "glossary",
        "summarize_paper": "summarize",
        "save_to_file": "save_file",
        "general_answer": "general"
    }
)

# 모든 노드에서 종료
workflow.add_edge("search_paper", END)
workflow.add_edge("web_search", END)
workflow.add_edge("glossary", END)
workflow.add_edge("summarize", END)
workflow.add_edge("save_file", END)
workflow.add_edge("general", END)

agent = workflow.compile()
```

---

## 6. 난이도별 답변 모드 구현

### 6.1 Easy 모드 (초심자)

**특징:**
- 전문 용어 최소화
- 비유와 예시 많이 사용
- 단계별 설명
- 수식 최소화

**프롬프트 예시:**
```python
EASY_MODE_PROMPT = """
당신은 AI/ML 초심자를 위한 논문 리뷰 어시스턴트입니다.

답변 규칙:
1. 전문 용어가 나오면 반드시 쉬운 말로 풀어서 설명
2. 실생활 비유 사용 (예: "Attention은 사람이 책을 읽을 때 중요한 부분에 집중하는 것과 같습니다")
3. 수식은 최소화하고, 나오면 직관적으로 설명
4. 핵심 아이디어 3가지 이내로 요약

사용자 질문: {question}
"""
```

### 6.2 Hard 모드 (전문가)

**특징:**
- 기술적 세부사항 포함
- 수식 및 알고리즘 설명
- 관련 논문 비교
- 구현 세부사항

**프롬프트 예시:**
```python
HARD_MODE_PROMPT = """
당신은 AI/ML 전문가를 위한 논문 리뷰 어시스턴트입니다.

답변 규칙:
1. 기술적 세부사항 및 수식 포함
2. 알고리즘의 시간/공간 복잡도 분석
3. 관련 논문과의 비교 (장단점)
4. 구현 시 고려사항
5. 최신 연구 동향과의 연결

사용자 질문: {question}
"""
```

### 6.3 UI에서 모드 선택

```python
import streamlit as st

# Streamlit UI
difficulty_mode = st.selectbox(
    "답변 난이도 선택",
    ["Easy 모드 (초심자용)", "Hard 모드 (전문가용)"]
)

difficulty = "easy" if "Easy" in difficulty_mode else "hard"

# Agent에 난이도 전달
response = agent.run(user_query, difficulty=difficulty)
```

---

## 7. 최종 도구 목록 요약

### 필수 도구 (5개)

| 번호 | 도구 이름 | 필수 여부 (OT 기준) | 직접 구현 여부 | 설명 |
|------|-----------|---------------------|----------------|------|
| 1 | RAG 논문 검색 도구 | ✅ (선택 기능이지만 프로젝트 핵심) | ✅ 직접 구현 | 로컬 DB에서 논문 검색 |
| 2 | 웹 검색 도구 | ✅ 필수 | ❌ Langchain 제공 | 최신 논문 웹 검색 |
| 3 | 논문 용어집 검색 도구 | 추가 제안 | ✅ 직접 구현 | 전문 용어 설명 |
| 4 | 논문 요약 도구 | 추가 제안 | ✅ 직접 구현 | 난이도별 논문 요약 |
| 5 | 파일 저장 도구 | ✅ 필수 | ✅ 직접 구현 | 대화/요약 내용 저장 |

### 선택 도구 (추가 가능)

| 번호 | 도구 이름 | 설명 |
|------|-----------|------|
| 6 | Text-to-SQL 도구 | 논문 통계 정보 조회 (가산점) |
| 7 | 논문 비교 도구 | 여러 논문 비교 분석 |
| 8 | 인용 추출 도구 | 논문 주요 인용문 추출 |

---

## 8. 참고 자료

- Langchain Tools 문서: https://docs.langchain.com/oss/python/langchain/tools#tools
- Langchain Agent 문서: https://docs.langchain.com/oss/python/langchain/agents
- LangGraph 문서: https://langchain-ai.github.io/langgraph/
- Tavily Search API: https://tavily.com/
- OpenAI Function Calling: https://platform.openai.com/docs/guides/function-calling
