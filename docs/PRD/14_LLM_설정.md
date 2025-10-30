# 14. LLM 설정

## 문서 정보
- **작성일**: 2025-10-30
- **프로젝트명**: 논문 리뷰 챗봇 (AI Agent + RAG)
- **팀명**: 연결의 민족

---

## 1. LLM 모델 선택

### 1.1 개발 환경

```python
from langchain_openai import ChatOpenAI
from langchain_upstage import ChatUpstage

# 개발용 Option 1: GPT-3.5-turbo (비용 절감)
llm_openai_dev = ChatOpenAI(
    model="gpt-3.5-turbo",
    temperature=0.0,
    max_tokens=2000,
    streaming=True
)

# 개발용 Option 2: Solar-pro (한국어 특화, 비용 효율적)
llm_solar_dev = ChatUpstage(
    model="solar-pro",
    temperature=0.0,
    max_tokens=2000,
    streaming=True
)
```

### 1.2 프로덕션 환경

```python
# 프로덕션 Option 1: GPT-4 (높은 품질)
llm_openai_prod = ChatOpenAI(
    model="gpt-4",
    temperature=0.7,
    max_tokens=3000,
    streaming=True
)

# 프로덕션 Option 2: Solar-pro (한국어 답변, 비용 효율)
llm_solar_prod = ChatUpstage(
    model="solar-pro",
    temperature=0.7,
    max_tokens=3000,
    streaming=True
)
```

### 1.3 하이브리드 전략 (권장)

```python
# 난이도별 모델 선택
def get_llm(difficulty="easy", language="ko"):
    """
    난이도와 언어에 따라 적절한 LLM 선택

    - Easy 모드 + 한국어: Solar (한국어 특화, 저비용)
    - Hard 모드 + 영어: GPT-4 (기술적 정확도)
    """
    if difficulty == "easy" and language == "ko":
        return ChatUpstage(model="solar-pro", temperature=0.7)
    elif difficulty == "hard":
        return ChatOpenAI(model="gpt-4", temperature=0.7)
    else:
        return ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)
```

---

## 2. API 키 관리

### 2.1 환경 변수 설정

```bash
# .env 파일
OPENAI_API_KEY=sk-...
SOLAR_API_KEY=up-...
TAVILY_API_KEY=tvly-...
DATABASE_URL=postgresql://user:password@localhost:5432/papers
```

### 2.2 코드에서 로드

```python
import os
from dotenv import load_dotenv

load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")
solar_api_key = os.getenv("SOLAR_API_KEY")

if not openai_api_key:
    raise ValueError("OPENAI_API_KEY가 설정되지 않았습니다")
if not solar_api_key:
    raise ValueError("SOLAR_API_KEY가 설정되지 않았습니다")
```

---

## 3. 에러 핸들링

### 3.1 LLM API 호출 흐름

```mermaid
sequenceDiagram
    autonumber
    participant Agent as AI Agent
    participant Client as LLM Client
    participant API1 as OpenAI API
    participant API2 as Solar API

    Agent->>Client: generate_answer(question, difficulty)
    Client->>Client: 난이도별 모델 선택

    alt Easy 모드 + 한국어
        Client->>API2: Solar API 호출<br/>solar-pro
        API2-->>Client: 한국어 답변
    else Hard 모드 or 영어
        Client->>API1: OpenAI API 호출<br/>GPT-4
        API1-->>Client: 상세 답변
    end

    Client-->>Agent: 최종 답변 반환
```

### 3.2 에러 처리 흐름

```mermaid
graph TB
    subgraph Request["🔸 API 요청"]
        direction LR
        A[LLM 호출 시도] --> B{에러<br/>발생?}
        B -->|No| C[✅ 정상 응답<br/>반환]
    end

    subgraph Retry["🔹 재시도 로직"]
        direction TB
        B -->|Yes| D{재시도<br/>횟수<br/>< 3?}
        D -->|Yes| E[대기<br/>2^n초<br/>Exponential Backoff]
        E --> A
        D -->|No| F[최종 실패<br/>에러 로그]
    end

    subgraph Fallback["🔺 대체 전략"]
        direction LR
        F --> G{대체<br/>API<br/>사용?}
        G -->|Yes| H[다른 모델로<br/>재시도]
        G -->|No| I[❌ 사용자에게<br/>에러 메시지]
        H --> A
    end

    Request --> Retry
    Retry --> Fallback

    %% Subgraph 스타일
    style Request fill:#e1f5ff,stroke:#01579b,stroke-width:3px,color:#000
    style Retry fill:#f3e5f5,stroke:#4a148c,stroke-width:3px,color:#000
    style Fallback fill:#e8f5e9,stroke:#1b5e20,stroke-width:3px,color:#000

    %% 노드 스타일
    style A fill:#90caf9,stroke:#1976d2,color:#000
    style B fill:#ba68c8,stroke:#7b1fa2,color:#fff
    style C fill:#66bb6a,stroke:#2e7d32,color:#fff
    style D fill:#ba68c8,stroke:#7b1fa2,color:#fff
    style E fill:#ce93d8,stroke:#7b1fa2,color:#000
    style F fill:#ef9a9a,stroke:#c62828,color:#000
    style G fill:#ba68c8,stroke:#7b1fa2,color:#fff
    style H fill:#ffcc80,stroke:#f57c00,color:#000
    style I fill:#ef9a9a,stroke:#c62828,color:#000
```

### 3.3 재시도 로직 구현

```python
from tenacity import retry, stop_after_attempt, wait_exponential

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10)
)
def llm_invoke_with_retry(prompt):
    """LLM 호출 시 자동 재시도"""
    try:
        response = llm.invoke(prompt)
        return response
    except Exception as e:
        logger.write(f"LLM 호출 실패: {e}")
        raise
```

### 3.2 타임아웃 설정

```python
llm = ChatOpenAI(
    model="gpt-4",
    temperature=0.7,
    request_timeout=30,  # 30초 타임아웃
    max_retries=2
)
```

---

## 4. 토큰 사용량 추적

### 4.1 토큰 카운팅

```python
from langchain.callbacks import get_openai_callback

with get_openai_callback() as cb:
    response = llm.invoke("논문 요약해줘")

    logger.write(f"총 토큰: {cb.total_tokens}")
    logger.write(f"프롬프트 토큰: {cb.prompt_tokens}")
    logger.write(f"완성 토큰: {cb.completion_tokens}")
    logger.write(f"총 비용: ${cb.total_cost}")
```

### 4.2 비용 모니터링

```python
def monitor_cost(func):
    """비용 모니터링 데코레이터"""
    def wrapper(*args, **kwargs):
        with get_openai_callback() as cb:
            result = func(*args, **kwargs)
            logger.write(f"[비용] {func.__name__}: ${cb.total_cost:.4f}")
            return result
    return wrapper

@monitor_cost
def generate_answer(question):
    return llm.invoke(question)
```

---

## 5. 스트리밍 응답

### 5.1 기본 스트리밍

```python
llm = ChatOpenAI(
    model="gpt-4",
    streaming=True,
    callbacks=[StreamingStdOutCallbackHandler()]
)

for chunk in llm.stream("논문 요약해줘"):
    print(chunk.content, end="", flush=True)
```

### 5.2 Streamlit 통합

```python
from langchain.callbacks import StreamlitCallbackHandler

import streamlit as st

st_callback = StreamlitCallbackHandler(st.container())

llm = ChatOpenAI(
    model="gpt-4",
    streaming=True,
    callbacks=[st_callback]
)
```

---

## 6. 모델 파라미터

### 6.1 Temperature 설정

| 값 | 용도 | 설명 |
|----|------|------|
| 0.0 | 검색, 라우팅 | 결정론적, 일관된 출력 |
| 0.3-0.5 | 요약, 분류 | 약간의 창의성 |
| 0.7-0.9 | 답변 생성 | 자연스러운 답변 |

```python
# 라우팅용: 낮은 temperature
router_llm = ChatOpenAI(model="gpt-4", temperature=0.0)

# 답변 생성용: 높은 temperature
answer_llm = ChatOpenAI(model="gpt-4", temperature=0.7)
```

### 6.2 Max Tokens 설정

```python
llm = ChatOpenAI(
    model="gpt-4",
    max_tokens=3000,  # 최대 토큰 수 제한
    stop=["\n\n---\n\n"]  # 중단 시퀀스
)
```

---

## 7. 참고 자료

- OpenAI API: https://platform.openai.com/docs/api-reference
- Langchain ChatOpenAI: https://python.langchain.com/docs/integrations/chat/openai/
