#!/usr/bin/env python3
# ---------------------- 통합 테스트 ---------------------- #
"""
실험 관리 시스템 통합 테스트

테스트 항목:
- 여러 실험 생성 및 검색
- find_experiments.py 스크립트 기능 테스트
- aggregate_metrics.py 스크립트 기능 테스트
- 실제 사용 시나리오 테스트
"""

# ------------------------- 표준 라이브러리 ------------------------- #
import json                                    # JSON 파일 처리
import shutil                                  # 폴더 삭제
import subprocess                              # 스크립트 실행
from pathlib import Path                       # 파일 경로 처리
from datetime import datetime                  # 날짜/시간 처리

# ------------------------- 서드파티 라이브러리 ------------------------- #
import pytest                                  # 테스트 프레임워크

# ------------------------- 프로젝트 모듈 ------------------------- #
from src.utils.experiment_manager import ExperimentManager


# ==================== 테스트 픽스처 ==================== #
# ---------------------- 실험 폴더 정리 픽스처 ---------------------- #
@pytest.fixture(autouse=True)
def cleanup_experiments():
    """
    각 테스트 후 실험 폴더 정리

    테스트 격리를 위해 테스트 실행 후 생성된 폴더 삭제
    """
    yield                                      # 테스트 실행

    # 테스트 후 정리
    if Path("experiments").exists():
        shutil.rmtree("experiments")


# ==================== 다중 실험 시나리오 테스트 ==================== #
# ---------------------- 여러 실험 생성 및 검색 테스트 ---------------------- #
def test_multiple_experiments_creation():
    """
    여러 실험 생성 및 검색 테스트

    검증 항목:
    - 3개 실험 생성
    - Session ID 자동 증가
    - 각 실험 독립적 폴더 생성
    - metadata.json 개별 관리
    """
    experiments = []

    # 3개 실험 생성
    for i in range(3):
        exp = ExperimentManager()

        # 메타데이터 업데이트
        exp.update_metadata(
            difficulty=["easy", "hard", "easy"][i],
            tool_used=["rag_paper", "web_search", "rag_glossary"][i],
            user_query=f"질문 {i+1}",
            success=True
        )

        # 평가 지표 저장
        exp.save_rag_metrics({
            'recall_at_5': 0.80 + i * 0.05,
            'precision_at_5': 0.85 + i * 0.03
        })

        exp.close()
        experiments.append(exp)

    # Session ID 확인
    assert experiments[0].metadata['session_id'] == "001"
    assert experiments[1].metadata['session_id'] == "002"
    assert experiments[2].metadata['session_id'] == "003"

    # 폴더 독립성 확인
    today = datetime.now().strftime("%Y%m%d")
    date_dir = Path(f"experiments/{today}")
    assert date_dir.exists()

    # 3개 세션 폴더 존재 확인
    session_folders = list(date_dir.glob("*_session_*"))
    assert len(session_folders) == 3


# ==================== 검색 기능 통합 테스트 ==================== #
# ---------------------- find_experiments 기능 테스트 ---------------------- #
def test_find_experiments_integration():
    """
    find_experiments.py 스크립트 통합 테스트

    검증 항목:
    - 난이도별 검색
    - 도구별 검색
    - 날짜별 검색
    - 검색 결과 정확성
    """
    # 테스트 실험 생성
    exp1 = ExperimentManager()
    exp1.update_metadata(difficulty="easy", tool_used="rag_paper", success=True)
    exp1.close()

    exp2 = ExperimentManager()
    exp2.update_metadata(difficulty="hard", tool_used="web_search", success=True)
    exp2.close()

    exp3 = ExperimentManager()
    exp3.update_metadata(difficulty="easy", tool_used="rag_glossary", success=False)
    exp3.close()

    # find_experiments 모듈 임포트
    import sys
    sys.path.insert(0, 'scripts')
    from find_experiments import find_experiments

    # 난이도별 검색 테스트
    easy_results = find_experiments(difficulty="easy")
    assert len(easy_results) == 2

    hard_results = find_experiments(difficulty="hard")
    assert len(hard_results) == 1

    # 도구별 검색 테스트
    rag_paper_results = find_experiments(tool="rag_paper")
    assert len(rag_paper_results) == 1

    # 성공 여부 검색 테스트
    success_results = find_experiments(min_success=True)
    assert len(success_results) == 2

    # 복합 조건 검색 테스트
    easy_success_results = find_experiments(difficulty="easy", min_success=True)
    assert len(easy_success_results) == 1


# ==================== 집계 기능 통합 테스트 ==================== #
# ---------------------- aggregate_metrics 기능 테스트 ---------------------- #
def test_aggregate_metrics_integration():
    """
    aggregate_metrics.py 스크립트 통합 테스트

    검증 항목:
    - RAG 지표 집계
    - Agent 정확도 집계
    - 응답 시간 집계
    - 비용 분석 집계
    """
    # 테스트 실험 3개 생성 (평가 지표 포함)
    for i in range(3):
        exp = ExperimentManager()

        exp.update_metadata(
            difficulty="easy",
            tool_used="rag_paper",
            response_time_ms=3000 + i * 200
        )

        # RAG 평가 지표
        exp.save_rag_metrics({
            'recall_at_5': 0.80 + i * 0.05,
            'precision_at_5': 0.85 + i * 0.03,
            'faithfulness': 0.88 + i * 0.02,
            'answer_relevancy': 0.90 + i * 0.02
        })

        # Agent 정확도
        exp.save_agent_accuracy({
            'routing_accuracy': 0.90 + i * 0.03,
            'correct_decisions': 15 + i,
            'incorrect_decisions': 2,
            'average_confidence': 0.85 + i * 0.03
        })

        # 응답 시간
        exp.save_latency_report({
            'total_time_ms': 3000 + i * 200,
            'routing_time_ms': 150,
            'retrieval_time_ms': 1200 + i * 50,
            'generation_time_ms': 1650 + i * 150
        })

        # 비용 분석
        exp.save_cost_analysis({
            'total_tokens': 2000 + i * 100,
            'prompt_tokens': 1200 + i * 50,
            'completion_tokens': 800 + i * 50,
            'cost_usd': 0.020 + i * 0.001,
            'cost_krw': 26.0 + i * 1.3
        })

        exp.close()

    # aggregate_metrics 모듈 임포트
    import sys
    if 'scripts' not in sys.path:
        sys.path.insert(0, 'scripts')
    from aggregate_metrics import aggregate_rag_metrics, aggregate_agent_accuracy, aggregate_latency, aggregate_cost

    # 날짜 가져오기
    today = datetime.now().strftime("%Y%m%d")

    # RAG 지표 집계 테스트
    rag_agg = aggregate_rag_metrics(today)
    assert 'recall_at_5' in rag_agg
    assert 'mean' in rag_agg['recall_at_5']
    assert 'min' in rag_agg['recall_at_5']
    assert 'max' in rag_agg['recall_at_5']

    # 평균값 검증 (0.80, 0.85, 0.90의 평균 = 0.85)
    assert abs(rag_agg['recall_at_5']['mean'] - 0.85) < 0.01

    # Agent 정확도 집계 테스트
    agent_agg = aggregate_agent_accuracy(today)
    assert 'routing_accuracy' in agent_agg
    assert agent_agg['total_decisions'] == 54  # (15+2) + (16+2) + (17+2) = 54

    # 응답 시간 집계 테스트
    latency_agg = aggregate_latency(today)
    assert 'total_time_ms' in latency_agg
    assert 'mean' in latency_agg['total_time_ms']
    # 평균값 검증 (3000, 3200, 3400의 평균 = 3200)
    assert abs(latency_agg['total_time_ms']['mean'] - 3200) < 1

    # 비용 분석 집계 테스트
    cost_agg = aggregate_cost(today)
    assert 'total_tokens' in cost_agg
    assert cost_agg['total_tokens']['sum'] == 6300  # 2000 + 2100 + 2200


# ==================== 실제 사용 시나리오 테스트 ==================== #
# ---------------------- 전체 워크플로우 시나리오 ---------------------- #
def test_real_world_scenario():
    """
    실제 사용 시나리오 통합 테스트

    시나리오:
    1. 챗봇 시작 (실험 초기화)
    2. 사용자 질문 입력
    3. Agent 도구 선택
    4. RAG 검색 실행
    5. DB 쿼리 기록
    6. 프롬프트 저장
    7. 답변 생성
    8. 평가 지표 계산
    9. 실험 종료
    10. 검색 및 집계
    """
    # 1. 챗봇 시작
    with ExperimentManager() as exp:
        # 2. 사용자 질문 입력
        user_query = "RAG에서 Retrieval의 역할은 무엇인가요?"
        difficulty = "easy"

        exp.update_metadata(
            difficulty=difficulty,
            user_query=user_query
        )

        # 3. Agent 도구 선택 시뮬레이션
        selected_tool = "rag_paper"
        exp.update_metadata(tool_used=selected_tool)

        # 도구별 Logger 생성
        tool_logger = exp.get_tool_logger(selected_tool)
        tool_logger.write("RAG 논문 검색 시작")
        tool_logger.close()

        # 4. RAG 검색 실행 시뮬레이션
        # 4-1. DB 쿼리 기록
        exp.log_sql_query(
            query="SELECT title, abstract, content FROM papers WHERE vector <=> '[0.1, 0.2, ...]' LIMIT 5",
            description="RAG 관련 논문 벡터 검색",
            tool=selected_tool,
            execution_time_ms=180
        )

        # 4-2. pgvector 검색 기록
        exp.log_pgvector_search({
            'tool': selected_tool,
            'collection': 'papers',
            'query_text': user_query,
            'top_k': 5,
            'execution_time_ms': 180
        })

        # 4-3. 검색 결과 저장
        exp.save_search_results(selected_tool, {
            'query': user_query,
            'results': [
                {'title': 'RAG 논문 1', 'score': 0.95},
                {'title': 'RAG 논문 2', 'score': 0.90},
                {'title': 'RAG 논문 3', 'score': 0.85},
            ],
            'count': 3
        })

        # 5. 프롬프트 저장
        system_prompt = "당신은 AI 논문을 설명하는 전문가입니다."
        user_prompt = user_query
        final_prompt = f"{system_prompt}\n\n{user_prompt}\n\n[검색된 논문 내용...]"

        exp.save_system_prompt(system_prompt)
        exp.save_user_prompt(user_prompt)
        exp.save_final_prompt(final_prompt)

        # 6. 답변 생성 시뮬레이션
        response = "Retrieval은 RAG에서 외부 지식을 검색하여 가져오는 역할을 합니다..."
        exp.save_output("response.txt", response)

        # 7. UI 인터랙션 기록
        exp.log_ui_interaction(f"사용자 질문: {user_query}")
        exp.log_ui_interaction(f"답변 생성 완료")

        exp.log_ui_event({
            'event_type': 'question_submitted',
            'difficulty': difficulty,
            'query_length': len(user_query)
        })

        # 8. 평가 지표 계산
        exp.save_rag_metrics({
            'recall_at_5': 0.88,
            'precision_at_5': 0.92,
            'faithfulness': 0.90,
            'answer_relevancy': 0.94
        })

        exp.save_agent_accuracy({
            'routing_accuracy': 1.0,
            'correct_decisions': 1,
            'incorrect_decisions': 0,
            'average_confidence': 0.95
        })

        exp.save_latency_report({
            'total_time_ms': 3500,
            'routing_time_ms': 120,
            'retrieval_time_ms': 1800,
            'generation_time_ms': 1580
        })

        exp.save_cost_analysis({
            'total_tokens': 2500,
            'prompt_tokens': 1500,
            'completion_tokens': 1000,
            'cost_usd': 0.025,
            'cost_krw': 33.5
        })

        exp.update_metadata(
            success=True,
            response_time_ms=3500
        )

        experiment_dir = exp.experiment_dir
        metadata_file = exp.metadata_file

    # 9. 실험 종료 후 검증
    # 모든 데이터가 정상적으로 저장되었는지 확인
    assert (experiment_dir / "chatbot.log").exists()
    assert (experiment_dir / "tools" / f"{selected_tool}.log").exists()
    assert (experiment_dir / "database" / "queries.sql").exists()
    assert (experiment_dir / "database" / "pgvector_searches.json").exists()
    assert (experiment_dir / "database" / "search_results.json").exists()
    assert (experiment_dir / "prompts" / "system_prompt.txt").exists()
    assert (experiment_dir / "prompts" / "user_prompt.txt").exists()
    assert (experiment_dir / "prompts" / "final_prompt.txt").exists()
    assert (experiment_dir / "ui" / "user_interactions.log").exists()
    assert (experiment_dir / "ui" / "ui_events.json").exists()
    assert (experiment_dir / "outputs" / "response.txt").exists()
    assert (experiment_dir / "evaluation" / "rag_metrics.json").exists()
    assert (experiment_dir / "evaluation" / "agent_accuracy.json").exists()
    assert (experiment_dir / "evaluation" / "latency_report.json").exists()
    assert (experiment_dir / "evaluation" / "cost_analysis.json").exists()

    # metadata.json 최종 검증
    with open(metadata_file, 'r', encoding='utf-8') as f:
        final_meta = json.load(f)

    assert final_meta['session_id'] == "001"
    assert final_meta['difficulty'] == "easy"
    assert final_meta['tool_used'] == "rag_paper"
    assert final_meta['user_query'] == user_query
    assert final_meta['success'] is True
    assert final_meta['response_time_ms'] == 3500
    assert final_meta['end_time'] is not None

    # 10. 검색 테스트
    import sys
    if 'scripts' not in sys.path:
        sys.path.insert(0, 'scripts')
    from find_experiments import find_experiments

    results = find_experiments(difficulty="easy", tool="rag_paper")
    assert len(results) == 1
    assert results[0]['metadata']['user_query'] == user_query


# ==================== 스크립트 실행 테스트 ==================== #
# ---------------------- find_experiments.py 스크립트 실행 테스트 ---------------------- #
def test_find_experiments_script_execution():
    """
    find_experiments.py 스크립트 실행 테스트

    검증 항목:
    - 명령줄에서 스크립트 실행
    - 정상 출력 확인
    """
    # 테스트 실험 생성
    exp = ExperimentManager()
    exp.update_metadata(difficulty="easy", tool_used="rag_paper")
    exp.close()

    # 스크립트 실행
    import sys
    result = subprocess.run(
        [sys.executable, 'scripts/find_experiments.py', '--difficulty', 'easy'],
        capture_output=True,
        text=True,
        env={'PYTHONPATH': '/home/ieyeppo/AI_Lab/langchain-project'}
    )

    # 실행 성공 확인
    assert result.returncode == 0
    assert "개의 실험을 찾았습니다" in result.stdout


# ---------------------- aggregate_metrics.py 스크립트 실행 테스트 ---------------------- #
def test_aggregate_metrics_script_execution():
    """
    aggregate_metrics.py 스크립트 실행 테스트

    검증 항목:
    - 명령줄에서 스크립트 실행
    - JSON 출력 확인
    """
    # 테스트 실험 생성
    exp = ExperimentManager()
    exp.save_rag_metrics({
        'recall_at_5': 0.85,
        'precision_at_5': 0.90
    })
    exp.close()

    # 스크립트 실행
    import sys
    import tempfile
    today = datetime.now().strftime("%Y%m%d")

    # 임시 출력 파일 생성
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp:
        output_file = tmp.name

    result = subprocess.run(
        [sys.executable, 'scripts/aggregate_metrics.py', '--date', today, '--output', output_file],
        capture_output=True,
        text=True,
        env={'PYTHONPATH': '/home/ieyeppo/AI_Lab/langchain-project'}
    )

    # 실행 성공 확인
    assert result.returncode == 0

    # 임시 파일 삭제
    Path(output_file).unlink()
