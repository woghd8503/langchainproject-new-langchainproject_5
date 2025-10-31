# **프로젝트 제목**  
프로젝트의 간단한 소개와 목적을 작성합니다.  
- **프로젝트 기간:** YYYY.MM.DD ~ YYYY.MM.DD  
- **배포 링크:** [서비스 바로가기](링크 입력) *(필요 시 추가)*  

---

## **1. 서비스 구성 요소**  
### **1.1 주요 기능**  
- 기능 1: *(주요 기능 간단 설명)*  
- 기능 2: *(주요 기능 간단 설명)*  
- 기능 3: *(주요 기능 간단 설명)*  

### **1.2 사용자 흐름**  
- 사용자 시나리오 예시:  
  1. *(유저 행동 1 설명)*  
  2. *(유저 행동 2 설명)*  

---

## **2. 활용 장비 및 협업 툴**  

### **2.1 활용 장비**  
- **서버 장비:** *(예: AWS EC2 t2.medium)*  
- **개발 환경:** *(예: Ubuntu 20.04, Windows 11)*  
- **테스트 장비:** *(예: MacBook Pro, GPU RTX 3090)*  

### **2.2 협업 툴**  
- **소스 관리:** GitHub  
- **프로젝트 관리:** Jira, Notion  
- **커뮤니케이션:** Slack  
- **버전 관리:** Git  

---

## **3. 최종 선정 AI 모델 구조**  
- **모델 이름:** *(예: BERT, GPT-4, YOLOv8)*  
- **구조 및 설명:** *(모델의 세부 구조 및 특징 설명)*  
- **학습 데이터:** *(데이터 출처 및 전처리 방법 설명)*  
- **평가 지표:** *(정확도, F1-Score, RMSE 등 평가 기준 설명)*  

---

## **4. 서비스 아키텍처**  
### **4.1 시스템 구조도**  
서비스 아키텍처 다이어그램을 첨부합니다. *(예: 이미지, 다이어그램)*  

![서비스 아키텍처 예시](링크 입력)  

### **4.2 데이터 흐름도**  
- 데이터 처리 및 서비스 간 연결 흐름 설명  
- 예시:  
  1. 사용자 입력 → AI 분석 → 결과 반환  
  2. 데이터 저장 → 전처리 → 모델 적용  

---

## **5. 사용 기술 스택**  
### **5.1 백엔드**  
- Flask / FastAPI / Django *(필요한 항목 작성)*  
- 데이터베이스: SQLite / PostgreSQL / MySQL  

### **5.2 프론트엔드**  
- React.js / Next.js / Vue.js *(필요한 항목 작성)*  

### **5.3 머신러닝 및 데이터 분석**  
- TensorFlow / PyTorch  
- scikit-learn / Pandas / NumPy  

### **5.4 배포 및 운영**  
- AWS EC2 / S3 / Lambda  
- Docker / Kubernetes / GitHub Actions  

---

## **6. 팀원 소개**  

| 이름      | 역할              | GitHub                               | 담당 기능                                 |
|----------|------------------|-------------------------------------|-----------------------------------------|
| **홍길동** | 팀장/백엔드 개발자 | [GitHub 링크](링크 입력)             | 서버 구축, API 개발, 배포 관리            |
| **김철수** | 프론트엔드 개발자  | [GitHub 링크](링크 입력)             | UI/UX 디자인, 프론트엔드 개발             |
| **이영희** | AI 모델 개발자    | [GitHub 링크](링크 입력)             | AI 모델 선정 및 학습, 데이터 분석         |
| **박수진** | 데이터 엔지니어    | [GitHub 링크](링크 입력)             | 데이터 수집, 전처리, 성능 평가 및 테스트   |

---

## **7. Appendix**  
### **7.1 참고 자료**  
- 논문 및 문서: *(참고 논문 또는 기술 문서 링크 추가)*  
- 데이터 출처: *(데이터셋 링크 또는 설명)*  
- 코드 참고 자료: *(레퍼런스 코드 또는 문서 링크)*  

### **7.2 설치 및 실행 방법**  
1. **필수 라이브러리 설치:**  
    ```bash
    pip install -r requirements.txt
    ```

2. **서버 실행:**  
    ```bash
    python app.py
    ```

3. **웹페이지 접속:**  
    ```
    http://localhost:5000
    ```

### **7.3 ExperimentManager 사용법**

실험 관리 시스템 (ExperimentManager)은 챗봇 실행마다 체계적인 실험 추적 및 평가를 제공합니다.

#### **7.3.1 기본 사용법**

```python
from src.utils.experiment_manager import ExperimentManager

# with 문을 사용한 실험 관리 (권장)
with ExperimentManager() as exp:
    # 실험 메타데이터 업데이트
    exp.update_metadata(
        difficulty="easy",
        tool_used="rag_paper",
        user_query="RAG에 대해 알려줘"
    )

    # 도구별 Logger 사용
    tool_logger = exp.get_tool_logger("rag_paper")
    tool_logger.write("RAG 검색 시작")
    tool_logger.close()

    # 프롬프트 저장
    exp.save_system_prompt("You are a helpful AI assistant.")
    exp.save_user_prompt("RAG에 대해 알려줘")

    # 평가 지표 저장
    exp.save_rag_metrics({
        'recall_at_5': 0.88,
        'precision_at_5': 0.92
    })

    # 실험 종료 시 자동으로 close() 호출
```

#### **7.3.2 주요 기능**

1. **Session ID 자동 부여**
   - 날짜별로 session_001부터 자동 증가
   - 폴더명 형식: `experiments/YYYYMMDD/YYYYMMDD_HHMMSS_session_XXX/`

2. **7개 서브 폴더 자동 생성**
   - `tools/` - 도구별 실행 로그
   - `database/` - DB 쿼리 및 검색 결과
   - `prompts/` - 프롬프트 기록
   - `ui/` - UI 인터랙션 로그
   - `outputs/` - 최종 답변 결과
   - `evaluation/` - 평가 지표 (RAG, Agent, Latency, Cost)
   - `debug/` - 디버그 정보 (선택)

3. **metadata.json 자동 관리**
   - 실험 시작 시 자동 생성
   - 실험 종료 시 end_time 자동 기록

#### **7.3.3 사용 예시**

**예시 1: DB 쿼리 기록**
```python
with ExperimentManager() as exp:
    exp.log_sql_query(
        query="SELECT * FROM papers WHERE title LIKE '%RAG%'",
        description="RAG 관련 논문 검색",
        tool="rag_paper",
        execution_time_ms=180
    )
```

**예시 2: pgvector 검색 기록**
```python
with ExperimentManager() as exp:
    exp.log_pgvector_search({
        'tool': 'rag_paper',
        'collection': 'papers',
        'query_text': 'RAG에 대해 알려줘',
        'top_k': 5,
        'execution_time_ms': 180
    })
```

**예시 3: Agent 정확도 저장**
```python
with ExperimentManager() as exp:
    exp.save_agent_accuracy({
        'routing_accuracy': 0.95,
        'correct_decisions': 18,
        'incorrect_decisions': 2,
        'average_confidence': 0.88
    })
```

**예시 4: 응답 시간 분석**
```python
with ExperimentManager() as exp:
    exp.save_latency_report({
        'total_time_ms': 3250,
        'routing_time_ms': 150,
        'retrieval_time_ms': 1200,
        'generation_time_ms': 1900
    })
```

**예시 5: 비용 분석**
```python
with ExperimentManager() as exp:
    exp.save_cost_analysis({
        'total_tokens': 2140,
        'prompt_tokens': 1250,
        'completion_tokens': 890,
        'cost_usd': 0.0214,
        'cost_krw': 28.62
    })
```

**예시 6: UI 인터랙션 기록**
```python
with ExperimentManager() as exp:
    exp.log_ui_interaction("사용자가 '쉬움' 난이도 선택")
    exp.log_ui_event({
        'event_type': 'question_submitted',
        'difficulty': 'easy',
        'query_length': 25
    })
```

**예시 7: 전체 워크플로우**
```python
with ExperimentManager() as exp:
    # 1. 메타데이터 설정
    exp.update_metadata(difficulty="easy", user_query="RAG란?")

    # 2. 프롬프트 저장
    exp.save_system_prompt("You are an AI expert.")
    exp.save_user_prompt("RAG란?")

    # 3. 도구 실행 및 로그
    tool_logger = exp.get_tool_logger("rag_paper")
    tool_logger.write("검색 시작")
    tool_logger.close()

    # 4. DB 쿼리 기록
    exp.log_sql_query("SELECT * FROM papers...", "논문 검색")

    # 5. 답변 저장
    exp.save_output("response.txt", "RAG는...")

    # 6. 평가 지표 저장
    exp.save_rag_metrics({'recall_at_5': 0.88})
    exp.save_latency_report({'total_time_ms': 3250})
    exp.save_cost_analysis({'total_tokens': 2140})

    # 7. 종료 (자동)
```

#### **7.3.4 실험 검색 및 집계**

**실험 검색:**
```bash
# 난이도별 검색
python scripts/find_experiments.py --difficulty easy

# 도구별 검색
python scripts/find_experiments.py --tool rag_paper

# 날짜별 검색
python scripts/find_experiments.py --date 20251031

# 복합 조건 검색
python scripts/find_experiments.py --difficulty easy --tool rag_paper --min-time 2000 --max-time 5000

# 상세 정보 출력
python scripts/find_experiments.py --difficulty easy --verbose
```

**평가 지표 집계:**
```bash
# JSON 형식으로 집계
python scripts/aggregate_metrics.py --date 20251031 --output results.json

# CSV 형식으로 집계
python scripts/aggregate_metrics.py --date 20251031 --output results.csv
```

집계 결과에는 다음이 포함됩니다:
- **RAG 지표**: Recall@5, Precision@5, Faithfulness, Answer Relevancy
- **Agent 정확도**: 라우팅 정확도, 정확한/잘못된 결정 수, 평균 신뢰도
- **응답 시간**: 평균, 중앙값, p95, 단계별 분석
- **비용**: 토큰 사용량, USD/KRW 비용

### **7.4 주요 커밋 기록 및 업데이트 내역**  

| 날짜         | 업데이트 내용                              | 담당자      |
|-------------|------------------------------------------|------------|
| YYYY.MM.DD  | 초기 프로젝트 세팅 및 환경 설정 추가          | 홍길동      |
| YYYY.MM.DD  | AI 모델 최적화 및 성능 개선                   | 이영희      |
| YYYY.MM.DD  | UI 디자인 및 페이지 구조 업데이트              | 김철수      |
| YYYY.MM.DD  | 데이터 전처리 및 분석 코드 추가                | 박수진      |
| YYYY.MM.DD  | 배포 환경 설정 및 Docker 이미지 구성           | 홍길동      |

