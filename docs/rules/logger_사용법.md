# Logger 사용법

## 📋 개요

`src/utils/logger.py`의 `Logger` 클래스는 프로젝트에서 표준 출력(`print`)을 대체하여 로그를 파일에 저장하고 관리하는 기능을 제공합니다.

**주요 기능:**
- 타임스탬프가 포함된 로그 기록
- 파일과 콘솔 동시 출력
- 표준 출력/에러 리디렉션
- tqdm 진행률 표시 지원
- 에러 메시지 색상 구분 (빨간색)

---

## 🚀 기본 사용법

### 1. Logger 초기화

```python
import os
from datetime import datetime
from src.utils.logger import Logger

# 로그 파일 경로 생성 (experiments/날짜/날짜_시간_실험명/ 구조)
today = datetime.now().strftime("%Y%m%d")        # 예: "20251028"
time_now = datetime.now().strftime("%H%M%S")     # 예: "143052"
experiment_name = "vectordb_build"                # 실험명 또는 기능명

# 디렉토리 자동 생성
log_dir = f"experiments/{today}/{today}_{time_now}_{experiment_name}"
os.makedirs(log_dir, exist_ok=True)

# Logger 인스턴스 생성
logger = Logger(
    log_path=f"{log_dir}/experiment.log",  # 로그 파일 경로
    print_also=True                          # 콘솔에도 출력할지 여부 (기본값: True)
)
```

**매개변수:**
- `log_path` (str): 로그 파일 저장 경로 (`experiments/날짜/날짜_시간_실험명/로그파일.log`)
- `print_also` (bool): `True`면 파일과 콘솔에 동시 출력, `False`면 파일에만 저장

**로그 파일 저장 구조:**
```
experiments/
└── 20251028/                              # 날짜 폴더 (YYYYMMDD)
    └── 20251028_143052_vectordb_build/    # 날짜_시간_실험명 폴더
        └── experiment.log                 # 로그 파일
```

---

### 2. 기본 로그 기록 (`print` 대체)

#### ❌ 기존 방식 (print 사용)
```python
print("모델 학습을 시작합니다.")
print(f"Epoch: {epoch}, Loss: {loss}")
```

#### ✅ Logger 사용 방식
```python
logger.write("모델 학습을 시작합니다.")
logger.write(f"Epoch: {epoch}, Loss: {loss}")
```

**출력 형식:**
```
2025-10-28 19:15:30 | 모델 학습을 시작합니다.
2025-10-28 19:15:32 | Epoch: 1, Loss: 0.523
```

---

### 3. 로그 메시지 옵션

#### 파일에만 저장 (콘솔 출력 안함)
```python
logger.write("내부 디버그 정보", print_also=False)
```

#### 에러 메시지 출력 (빨간색)
```python
logger.write("오류: 파일을 찾을 수 없습니다.", print_error=True)
```

**에러 메시지는 콘솔에서 빨간색으로 표시됩니다.**

---

## 🔄 표준 출력 리디렉션

### 자동으로 print문을 로그 파일에 기록하기

`start_redirect()`를 사용하면 모든 `print()` 호출이 자동으로 로그 파일에 기록됩니다.

```python
import os
from datetime import datetime
from src.utils.logger import Logger

# 로그 디렉토리 생성
today = datetime.now().strftime("%Y%m%d")
time_now = datetime.now().strftime("%H%M%S")
experiment_name = "model_training"
log_dir = f"experiments/{today}/{today}_{time_now}_{experiment_name}"
os.makedirs(log_dir, exist_ok=True)

logger = Logger(f"{log_dir}/training.log")

# 리디렉션 시작
logger.start_redirect()

# 이제 print도 자동으로 로그에 기록됨
print("이 메시지는 로그 파일에 저장됩니다.")
print("에러 발생!", file=sys.stderr)  # stderr도 로그에 기록됨

# 리디렉션 중지 (원상 복구)
logger.stop_redirect()

# 이제 print는 일반적으로 동작
print("이 메시지는 콘솔에만 출력됩니다.")
```

---

## 📊 실전 사용 예시

### 예시 1: RAG 파이프라인 로그 기록

```python
import os
from datetime import datetime
from src.utils.logger import Logger

def build_vectordb():
    # 로그 디렉토리 생성
    today = datetime.now().strftime("%Y%m%d")
    time_now = datetime.now().strftime("%H%M%S")
    experiment_name = "rag_vectordb_build"
    log_dir = f"experiments/{today}/{today}_{time_now}_{experiment_name}"
    os.makedirs(log_dir, exist_ok=True)

    # Logger 초기화
    logger = Logger(f"{log_dir}/vectordb_build.log")

    logger.write("=" * 50)
    logger.write("VectorDB 구축 시작")
    logger.write("=" * 50)

    # 문서 로드
    logger.write("문서 로드 중...")
    documents = load_documents("data/raw/")
    logger.write(f"총 {len(documents)}개 문서 로드 완료")

    # 텍스트 분할
    logger.write("텍스트 분할 중...")
    chunks = text_splitter.split_documents(documents)
    logger.write(f"{len(chunks)}개 청크 생성 완료")

    # 임베딩 생성
    logger.write("임베딩 생성 및 VectorDB 저장 중...")
    vectordb = Chroma.from_documents(chunks, embeddings)
    logger.write("VectorDB 구축 완료")

    # 로거 종료
    logger.close()

    return vectordb
```

**로그 파일 출력 예시:**
```
2025-10-28 19:20:15 | ==================================================
2025-10-28 19:20:15 | VectorDB 구축 시작
2025-10-28 19:20:15 | ==================================================
2025-10-28 19:20:15 | 문서 로드 중...
2025-10-28 19:20:18 | 총 45개 문서 로드 완료
2025-10-28 19:20:18 | 텍스트 분할 중...
2025-10-28 19:20:20 | 237개 청크 생성 완료
2025-10-28 19:20:20 | 임베딩 생성 및 VectorDB 저장 중...
2025-10-28 19:22:45 | VectorDB 구축 완료
```

---

### 예시 2: AI Agent 실행 로그

```python
import os
from datetime import datetime
from src.utils.logger import Logger

def run_agent(user_query: str):
    # 로그 디렉토리 생성
    today = datetime.now().strftime("%Y%m%d")
    time_now = datetime.now().strftime("%H%M%S")
    experiment_name = "agent_execution"
    log_dir = f"experiments/{today}/{today}_{time_now}_{experiment_name}"
    os.makedirs(log_dir, exist_ok=True)

    # Logger 초기화
    logger = Logger(f"{log_dir}/agent_execution.log")

    logger.write(f"사용자 질문: {user_query}")

    # 질문 분류
    query_type = router.classify(user_query)
    logger.write(f"질문 유형 분류: {query_type}")

    # 도구 호출
    if query_type == "web_search":
        logger.write("웹 검색 도구 호출")
        result = web_search_tool.run(user_query)
    elif query_type == "rag":
        logger.write("RAG 검색 도구 호출")
        result = rag_tool.run(user_query)
    else:
        logger.write("일반 답변 생성")
        result = llm.generate(user_query)

    logger.write(f"최종 답변 생성 완료 (길이: {len(result)}자)")
    logger.close()

    return result
```

---

### 예시 3: 평가 시스템 로그

```python
import os
from datetime import datetime
from src.utils.logger import Logger

def evaluate_chatbot(test_questions: list):
    # 로그 디렉토리 생성
    today = datetime.now().strftime("%Y%m%d")
    time_now = datetime.now().strftime("%H%M%S")
    experiment_name = "chatbot_evaluation"
    log_dir = f"experiments/{today}/{today}_{time_now}_{experiment_name}"
    os.makedirs(log_dir, exist_ok=True)

    # Logger 초기화
    logger = Logger(f"{log_dir}/evaluation.log")

    logger.write("=" * 60)
    logger.write("챗봇 성능 평가 시작")
    logger.write(f"총 {len(test_questions)}개 질문 테스트")
    logger.write("=" * 60)

    scores = []

    for i, question in enumerate(test_questions, 1):
        logger.write(f"\n[{i}/{len(test_questions)}] 질문: {question}")

        try:
            # 챗봇 답변 생성
            answer = chatbot.generate(question)
            logger.write(f"답변: {answer[:100]}...")  # 처음 100자만 로그

            # 평가 점수 계산
            score = evaluator.score(question, answer)
            scores.append(score)
            logger.write(f"평가 점수: {score}/5")

        except Exception as e:
            logger.write(f"오류 발생: {str(e)}", print_error=True)
            scores.append(0)

    # 최종 결과
    avg_score = sum(scores) / len(scores)
    logger.write("\n" + "=" * 60)
    logger.write(f"평가 완료 - 평균 점수: {avg_score:.2f}/5")
    logger.write("=" * 60)

    logger.close()

    return avg_score
```

---

### 예시 4: tqdm 진행률 표시와 함께 사용

```python
import os
from datetime import datetime
from src.utils.logger import Logger
from tqdm import tqdm

def process_documents(doc_list: list):
    # 로그 디렉토리 생성
    today = datetime.now().strftime("%Y%m%d")
    time_now = datetime.now().strftime("%H%M%S")
    experiment_name = "document_processing"
    log_dir = f"experiments/{today}/{today}_{time_now}_{experiment_name}"
    os.makedirs(log_dir, exist_ok=True)

    # Logger 초기화
    logger = Logger(f"{log_dir}/document_processing.log")

    # tqdm 리디렉션 설정
    logger.tqdm_redirect()

    logger.write("문서 처리 시작")

    # tqdm 진행률 표시
    for doc in tqdm(doc_list, desc="문서 처리 중"):
        # 문서 처리 로직
        processed = preprocess(doc)

        # 중요한 이벤트만 로그 기록
        if has_error(processed):
            logger.write(f"경고: {doc.name} 처리 중 오류", print_error=True)

    logger.write("문서 처리 완료")
    logger.close()
```

---

## 🎯 권장 사용 패턴

### 1. 함수/클래스 시작 시 Logger 생성
```python
import os
from datetime import datetime
from src.utils.logger import Logger

def train_model():
    # 로그 디렉토리 생성
    today = datetime.now().strftime("%Y%m%d")
    time_now = datetime.now().strftime("%H%M%S")
    experiment_name = "model_training"
    log_dir = f"experiments/{today}/{today}_{time_now}_{experiment_name}"
    os.makedirs(log_dir, exist_ok=True)

    # Logger 생성
    logger = Logger(f"{log_dir}/training.log")
    logger.write("학습 시작")

    # ... 학습 코드 ...

    logger.write("학습 완료")
    logger.close()
```

### 2. Context Manager 패턴 (선택 사항)
Logger를 context manager로 사용하려면 `__enter__`, `__exit__` 메서드를 추가해야 합니다.

현재는 명시적으로 `close()`를 호출하는 것을 권장합니다:
```python
import os
from datetime import datetime
from src.utils.logger import Logger

# 로그 디렉토리 생성
today = datetime.now().strftime("%Y%m%d")
time_now = datetime.now().strftime("%H%M%S")
experiment_name = "my_experiment"
log_dir = f"experiments/{today}/{today}_{time_now}_{experiment_name}"
os.makedirs(log_dir, exist_ok=True)

logger = Logger(f"{log_dir}/experiment.log")
try:
    logger.write("작업 시작")
    # ... 작업 코드 ...
finally:
    logger.close()  # 항상 로그 파일을 닫음
```

---

## 📝 주요 메서드 정리

| 메서드 | 설명 | 사용 예시 |
|--------|------|-----------|
| `write(message, print_also=True, print_error=False)` | 로그 메시지 기록 | `logger.write("메시지")` |
| `start_redirect()` | stdout/stderr를 로그로 리디렉션 | `logger.start_redirect()` |
| `stop_redirect()` | 리디렉션 중지 | `logger.stop_redirect()` |
| `tqdm_redirect()` | tqdm 출력을 로그로 리디렉션 | `logger.tqdm_redirect()` |
| `flush()` | 버퍼 플러시 | `logger.flush()` |
| `close()` | 로그 파일 닫기 | `logger.close()` |

---

## ⚠️ 주의사항

### 1. 로그 파일 경로 확인
로그 파일을 저장할 디렉토리를 반드시 생성해야 합니다:
```python
import os
from datetime import datetime

# experiments/날짜/날짜_시간_실험명/ 구조로 생성
today = datetime.now().strftime("%Y%m%d")
time_now = datetime.now().strftime("%H%M%S")
experiment_name = "my_experiment"  # 실험명이나 기능명
log_dir = f"experiments/{today}/{today}_{time_now}_{experiment_name}"
os.makedirs(log_dir, exist_ok=True)  # 디렉토리 자동 생성

logger = Logger(f"{log_dir}/experiment.log")
```

### 2. Logger 종료 필수
작업 완료 후 반드시 `close()`를 호출하여 파일을 닫아야 합니다:
```python
logger.close()
```

### 3. 리디렉션 사용 시 주의
`start_redirect()` 사용 시 모든 `print()`가 로그 파일에 기록되므로, 필요 없는 출력은 제거하거나 `stop_redirect()`로 중지하세요.

### 4. 빈 메시지 자동 무시
빈 문자열이나 공백만 있는 메시지는 자동으로 무시됩니다:
```python
logger.write("")        # 기록되지 않음
logger.write("   ")     # 기록되지 않음
```

---

## 🔍 로그 파일 위치 규칙

### 필수 디렉토리 구조
모든 로그 파일은 **experiments/날짜/날짜_시간_실험명/** 구조로 저장해야 합니다:

```
experiments/
├── 20251028/                                      # 날짜 폴더 (YYYYMMDD)
│   ├── 20251028_143052_rag_vectordb_build/       # 날짜_시간_실험명
│   │   ├── vectordb_build.log
│   │   └── (기타 실험 결과 파일)
│   │
│   ├── 20251028_150823_agent_execution/          # 날짜_시간_실험명
│   │   └── agent_execution.log
│   │
│   ├── 20251028_163045_chatbot_evaluation/       # 날짜_시간_실험명
│   │   ├── evaluation.log
│   │   └── evaluation_results.json
│   │
│   └── 20251028_180912_model_training/           # 날짜_시간_실험명
│       ├── training.log
│       └── checkpoints/
│
└── 20251029/                                      # 다음 날 실험
    └── 20251029_091520_feature_development/
        └── feature.log
```

### 로그 경로 생성 템플릿
```python
import os
from datetime import datetime

# 1. 날짜와 시간 생성
today = datetime.now().strftime("%Y%m%d")        # "20251028"
time_now = datetime.now().strftime("%H%M%S")     # "143052"

# 2. 실험명/기능명 정의
experiment_name = "vectordb_build"  # 작업 내용에 맞게 변경

# 3. 전체 경로 생성 (날짜/날짜_시간_실험명)
log_dir = f"experiments/{today}/{today}_{time_now}_{experiment_name}"
os.makedirs(log_dir, exist_ok=True)

# 4. Logger 생성
from src.utils.logger import Logger
logger = Logger(f"{log_dir}/experiment.log")
```

### 실험명 명명 규칙
- **기능 개발**: `feature_<기능명>` (예: `feature_web_search_tool`)
- **RAG 실험**: `rag_<내용>` (예: `rag_vectordb_build`, `rag_retriever_test`)
- **Agent 실험**: `agent_<내용>` (예: `agent_tool_calling`, `agent_routing`)
- **평가**: `eval_<대상>` (예: `eval_chatbot_accuracy`, `eval_response_quality`)
- **디버깅**: `debug_<문제>` (예: `debug_memory_leak`, `debug_api_error`)

### `.gitignore`에 로그 파일 추가
experiments 폴더의 로그 파일은 Git에 커밋하지 않도록 설정:
```
# .gitignore
experiments/**/*.log
experiments/**/checkpoints/
experiments/**/temp/
```

---

## 💡 print 대신 logger 사용하는 이유

| 항목 | print | logger.write |
|------|-------|--------------|
| 파일 저장 | ❌ 콘솔에만 출력 | ✅ 파일에 자동 저장 |
| 타임스탬프 | ❌ 수동 추가 필요 | ✅ 자동 추가 |
| 에러 구분 | ❌ 구분 없음 | ✅ 빨간색 표시 |
| 로그 관리 | ❌ 어려움 | ✅ 파일로 체계적 관리 |
| 디버깅 | ❌ 재실행 필요 | ✅ 로그 파일 확인 |
| 실험 추적 | ❌ 불가능 | ✅ 날짜별 추적 가능 |

---

## 🎓 실습 예제

실제로 logger를 사용해보는 간단한 예제:

```python
import os
import time
from datetime import datetime
from src.utils.logger import Logger

# 로그 디렉토리 생성
today = datetime.now().strftime("%Y%m%d")
time_now = datetime.now().strftime("%H%M%S")
experiment_name = "logger_practice"
log_dir = f"experiments/{today}/{today}_{time_now}_{experiment_name}"
os.makedirs(log_dir, exist_ok=True)

# Logger 생성
logger = Logger(f"{log_dir}/practice.log")

# 1. 기본 로그
logger.write("연습 시작")

# 2. 반복문에서 사용
for i in range(1, 6):
    logger.write(f"반복 {i}번째")
    time.sleep(0.5)

# 3. 에러 로그
logger.write("에러 발생 시뮬레이션", print_error=True)

# 4. 파일에만 기록
logger.write("이 메시지는 파일에만 저장됨", print_also=False)

# 5. 종료
logger.write("연습 종료")
logger.close()

print(f"로그 파일을 확인하세요: {log_dir}/practice.log")
```

---

## 📚 참고

- 로그 파일 위치: `src/utils/logger.py`
- 프로젝트 루트에서 실행 시 경로 주의
- 한글 인코딩: UTF-8 자동 적용
