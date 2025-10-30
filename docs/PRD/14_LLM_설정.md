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

# 개발용: GPT-3.5-turbo (비용 절감)
llm_dev = ChatOpenAI(
    model="gpt-3.5-turbo",
    temperature=0.0,
    max_tokens=2000,
    streaming=True
)
```

### 1.2 프로덕션 환경

```python
# 프로덕션: GPT-4 (높은 품질)
llm_prod = ChatOpenAI(
    model="gpt-4",
    temperature=0.7,
    max_tokens=3000,
    streaming=True
)
```

---

## 2. API 키 관리

### 2.1 환경 변수 설정

```bash
# .env 파일
OPENAI_API_KEY=sk-...
TAVILY_API_KEY=tvly-...
DATABASE_URL=postgresql://user:password@localhost:5432/papers
```

### 2.2 코드에서 로드

```python
import os
from dotenv import load_dotenv

load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("OPENAI_API_KEY가 설정되지 않았습니다")
```

---

## 3. 에러 핸들링

### 3.1 재시도 로직

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
