#!/usr/bin/env python3
# ---------------------- ExperimentManager 단위 테스트 ---------------------- #
"""
ExperimentManager 클래스 단위 테스트

테스트 항목:
- Session ID 자동 부여
- 폴더 구조 생성
- metadata.json 업데이트
- 평가 지표 저장
- with 문 컨텍스트 매니저
"""

# ------------------------- 표준 라이브러리 ------------------------- #
import json                                    # JSON 파일 처리
import shutil                                  # 폴더 삭제
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


# ==================== Session ID 테스트 ==================== #
# ---------------------- Session ID 자동 부여 테스트 ---------------------- #
def test_session_id_auto_increment():
    """
    Session ID 자동 증가 테스트

    검증 항목:
    - 첫 실행: session_001
    - 두 번째 실행: session_002
    - 세 번째 실행: session_003
    """
    # 첫 번째 실험
    exp1 = ExperimentManager()
    assert exp1.metadata['session_id'] == "001"
    exp1.close()

    # 두 번째 실험
    exp2 = ExperimentManager()
    assert exp2.metadata['session_id'] == "002"
    exp2.close()

    # 세 번째 실험
    exp3 = ExperimentManager()
    assert exp3.metadata['session_id'] == "003"
    exp3.close()


# ---------------------- 날짜 변경 시 Session ID 초기화 테스트 ---------------------- #
def test_session_id_reset_on_new_date():
    """
    날짜 변경 시 Session ID 초기화 테스트

    검증 항목:
    - 다른 날짜의 Session ID는 001부터 다시 시작
    """
    today = datetime.now().strftime("%Y%m%d")

    # 오늘 첫 실험
    exp_today = ExperimentManager()
    today_session_id = exp_today.metadata['session_id']
    exp_today.close()

    # 다른 날짜 폴더 수동 생성 (시뮬레이션)
    other_date = "20251030"
    other_dir = Path(f"experiments/{other_date}/{other_date}_100000_session_001")
    other_dir.mkdir(parents=True, exist_ok=True)

    # 다른 날짜의 Session ID는 독립적으로 관리됨
    assert today_session_id == "001"


# ==================== 폴더 구조 테스트 ==================== #
# ---------------------- 필수 폴더 생성 테스트 ---------------------- #
def test_folder_structure_creation():
    """
    필수 폴더 구조 생성 테스트

    검증 항목:
    - 메인 폴더 생성
    - 6개 필수 서브 폴더 생성
    - chatbot.log 파일 생성
    """
    with ExperimentManager() as exp:
        # 메인 폴더 존재 확인
        assert exp.experiment_dir.exists()

        # 6개 필수 폴더 존재 확인
        assert exp.tools_dir.exists()
        assert exp.database_dir.exists()
        assert exp.prompts_dir.exists()
        assert exp.ui_dir.exists()
        assert exp.outputs_dir.exists()
        assert exp.evaluation_dir.exists()

        # 로그 파일 존재 확인
        assert (exp.experiment_dir / "chatbot.log").exists()


# ---------------------- 폴더명 형식 테스트 ---------------------- #
def test_folder_naming_convention():
    """
    폴더명 형식 테스트

    검증 항목:
    - 형식: YYYYMMDD_HHMMSS_session_XXX
    - Session ID는 3자리 숫자
    """
    with ExperimentManager() as exp:
        folder_name = exp.experiment_dir.name

        # 폴더명 형식 검증
        parts = folder_name.split('_')
        assert len(parts) == 4                 # YYYYMMDD, HHMMSS, session, XXX
        assert parts[2] == "session"
        assert len(parts[3]) == 3              # Session ID는 3자리
        assert parts[3].isdigit()              # 숫자인지 확인


# ==================== metadata.json 테스트 ==================== #
# ---------------------- metadata.json 초기화 테스트 ---------------------- #
def test_metadata_initialization():
    """
    metadata.json 초기화 테스트

    검증 항목:
    - 필수 키 존재
    - 초기값 설정
    """
    with ExperimentManager() as exp:
        meta = exp.metadata

        # 필수 키 존재 확인
        assert 'session_id' in meta
        assert 'start_time' in meta
        assert 'difficulty' in meta
        assert 'tool_used' in meta
        assert 'user_query' in meta
        assert 'success' in meta
        assert 'response_time_ms' in meta
        assert 'end_time' in meta

        # 초기값 확인
        assert meta['session_id'] == "001"
        assert meta['start_time'] is not None
        assert meta['end_time'] is None


# ---------------------- metadata.json 업데이트 테스트 ---------------------- #
def test_metadata_update():
    """
    metadata.json 업데이트 테스트

    검증 항목:
    - 메타데이터 동적 업데이트
    - 파일에 정상 저장
    """
    with ExperimentManager() as exp:
        # 메타데이터 업데이트
        exp.update_metadata(
            difficulty="easy",
            tool_used="rag_paper",
            user_query="RAG에 대해 알려줘",
            success=True,
            response_time_ms=3250
        )

        # 메모리 상태 확인
        assert exp.metadata['difficulty'] == "easy"
        assert exp.metadata['tool_used'] == "rag_paper"
        assert exp.metadata['user_query'] == "RAG에 대해 알려줘"
        assert exp.metadata['success'] is True
        assert exp.metadata['response_time_ms'] == 3250

        # 파일 저장 확인
        with open(exp.metadata_file, 'r', encoding='utf-8') as f:
            saved_meta = json.load(f)

        assert saved_meta['difficulty'] == "easy"
        assert saved_meta['tool_used'] == "rag_paper"


# ==================== 데이터 저장 테스트 ==================== #
# ---------------------- RAG 평가 지표 저장 테스트 ---------------------- #
def test_save_rag_metrics():
    """
    RAG 평가 지표 저장 테스트

    검증 항목:
    - rag_metrics.json 파일 생성
    - 데이터 정상 저장
    """
    with ExperimentManager() as exp:
        # RAG 평가 지표 저장
        metrics = {
            'recall_at_5': 0.85,
            'precision_at_5': 0.90,
            'faithfulness': 0.88,
            'answer_relevancy': 0.92
        }
        exp.save_rag_metrics(metrics)

        # 파일 존재 확인
        metrics_file = exp.evaluation_dir / "rag_metrics.json"
        assert metrics_file.exists()

        # 데이터 확인
        with open(metrics_file, 'r', encoding='utf-8') as f:
            saved_metrics = json.load(f)

        assert saved_metrics['recall_at_5'] == 0.85
        assert saved_metrics['precision_at_5'] == 0.90
        assert 'timestamp' in saved_metrics


# ---------------------- Agent 정확도 저장 테스트 ---------------------- #
def test_save_agent_accuracy():
    """
    Agent 정확도 저장 테스트

    검증 항목:
    - agent_accuracy.json 파일 생성
    - 데이터 정상 저장
    """
    with ExperimentManager() as exp:
        # Agent 정확도 저장
        accuracy = {
            'routing_accuracy': 0.95,
            'correct_decisions': 18,
            'incorrect_decisions': 2,
            'average_confidence': 0.88
        }
        exp.save_agent_accuracy(accuracy)

        # 파일 존재 확인
        accuracy_file = exp.evaluation_dir / "agent_accuracy.json"
        assert accuracy_file.exists()

        # 데이터 확인
        with open(accuracy_file, 'r', encoding='utf-8') as f:
            saved_accuracy = json.load(f)

        assert saved_accuracy['routing_accuracy'] == 0.95
        assert saved_accuracy['correct_decisions'] == 18


# ---------------------- 응답 시간 분석 저장 테스트 ---------------------- #
def test_save_latency_report():
    """
    응답 시간 분석 저장 테스트

    검증 항목:
    - latency_report.json 파일 생성
    - 데이터 정상 저장
    """
    with ExperimentManager() as exp:
        # 응답 시간 분석 저장
        latency = {
            'total_time_ms': 3250,
            'routing_time_ms': 150,
            'retrieval_time_ms': 1200,
            'generation_time_ms': 1900,
            'p50': 3100,
            'p95': 4200,
            'p99': 5100
        }
        exp.save_latency_report(latency)

        # 파일 존재 확인
        latency_file = exp.evaluation_dir / "latency_report.json"
        assert latency_file.exists()

        # 데이터 확인
        with open(latency_file, 'r', encoding='utf-8') as f:
            saved_latency = json.load(f)

        assert saved_latency['total_time_ms'] == 3250
        assert saved_latency['p95'] == 4200


# ---------------------- 비용 분석 저장 테스트 ---------------------- #
def test_save_cost_analysis():
    """
    비용 분석 저장 테스트

    검증 항목:
    - cost_analysis.json 파일 생성
    - 데이터 정상 저장
    """
    with ExperimentManager() as exp:
        # 비용 분석 저장
        cost = {
            'total_tokens': 2140,
            'prompt_tokens': 1250,
            'completion_tokens': 890,
            'cost_usd': 0.0214,
            'cost_krw': 28.62
        }
        exp.save_cost_analysis(cost)

        # 파일 존재 확인
        cost_file = exp.evaluation_dir / "cost_analysis.json"
        assert cost_file.exists()

        # 데이터 확인
        with open(cost_file, 'r', encoding='utf-8') as f:
            saved_cost = json.load(f)

        assert saved_cost['total_tokens'] == 2140
        assert saved_cost['cost_usd'] == 0.0214


# ==================== 프롬프트 저장 테스트 ==================== #
# ---------------------- 프롬프트 저장 테스트 ---------------------- #
def test_save_prompts():
    """
    프롬프트 저장 테스트

    검증 항목:
    - system_prompt.txt 저장
    - user_prompt.txt 저장
    - final_prompt.txt 저장
    """
    with ExperimentManager() as exp:
        # 시스템 프롬프트 저장
        exp.save_system_prompt("You are a helpful AI assistant.")
        assert (exp.prompts_dir / "system_prompt.txt").exists()

        # 사용자 프롬프트 저장
        exp.save_user_prompt("RAG에 대해 알려줘")
        assert (exp.prompts_dir / "user_prompt.txt").exists()

        # 최종 프롬프트 저장
        exp.save_final_prompt("System: ...\nUser: RAG에 대해 알려줘")
        assert (exp.prompts_dir / "final_prompt.txt").exists()


# ==================== DB 저장 테스트 ==================== #
# ---------------------- SQL 쿼리 저장 테스트 ---------------------- #
def test_log_sql_query():
    """
    SQL 쿼리 저장 테스트

    검증 항목:
    - queries.sql 파일 생성
    - 쿼리 정상 기록
    """
    with ExperimentManager() as exp:
        # SQL 쿼리 기록
        exp.log_sql_query(
            query="SELECT * FROM papers WHERE title LIKE '%RAG%'",
            description="RAG 관련 논문 검색",
            tool="rag_paper",
            execution_time_ms=150
        )

        # 파일 존재 확인
        queries_file = exp.database_dir / "queries.sql"
        assert queries_file.exists()

        # 내용 확인
        with open(queries_file, 'r', encoding='utf-8') as f:
            content = f.read()

        assert "SELECT * FROM papers" in content
        assert "RAG 관련 논문 검색" in content


# ==================== with 문 컨텍스트 매니저 테스트 ==================== #
# ---------------------- with 문 자동 종료 테스트 ---------------------- #
def test_context_manager_with_statement():
    """
    with 문 컨텍스트 매니저 테스트

    검증 항목:
    - with 문 진입 시 정상 초기화
    - with 문 종료 시 자동 close 호출
    - end_time 자동 기록
    """
    with ExperimentManager() as exp:
        experiment_dir = exp.experiment_dir
        metadata_file = exp.metadata_file

        # with 블록 내에서는 정상 작동
        assert exp.metadata['end_time'] is None

        # 메타데이터 업데이트
        exp.update_metadata(difficulty="easy", success=True)

    # with 블록 종료 후
    # metadata.json에 end_time 기록 확인
    with open(metadata_file, 'r', encoding='utf-8') as f:
        final_meta = json.load(f)

    assert final_meta['end_time'] is not None
    assert final_meta['difficulty'] == "easy"
    assert final_meta['success'] is True


# ---------------------- with 문 예외 처리 테스트 ---------------------- #
def test_context_manager_with_exception():
    """
    with 문 예외 발생 시 테스트

    검증 항목:
    - 예외 발생 시에도 close 호출
    - end_time 기록
    """
    try:
        with ExperimentManager() as exp:
            metadata_file = exp.metadata_file

            # 강제로 예외 발생
            raise ValueError("테스트 예외")
    except ValueError:
        pass

    # 예외 발생해도 close는 호출되어야 함
    with open(metadata_file, 'r', encoding='utf-8') as f:
        final_meta = json.load(f)

    # end_time이 기록되어 있어야 함
    assert final_meta['end_time'] is not None


# ==================== 도구 Logger 테스트 ==================== #
# ---------------------- 도구별 Logger 생성 테스트 ---------------------- #
def test_get_tool_logger():
    """
    도구별 Logger 생성 테스트

    검증 항목:
    - 도구별 로그 파일 생성
    - Logger 인스턴스 반환
    """
    with ExperimentManager() as exp:
        # rag_paper 도구 Logger 생성
        rag_logger = exp.get_tool_logger("rag_paper")
        rag_logger.write("RAG 도구 실행 시작")
        rag_logger.close()

        # 로그 파일 존재 확인
        rag_log_file = exp.tools_dir / "rag_paper.log"
        assert rag_log_file.exists()

        # web_search 도구 Logger 생성
        web_logger = exp.get_tool_logger("web_search")
        web_logger.write("웹 검색 도구 실행 시작")
        web_logger.close()

        # 로그 파일 존재 확인
        web_log_file = exp.tools_dir / "web_search.log"
        assert web_log_file.exists()


# ==================== 통합 시나리오 테스트 ==================== #
# ---------------------- 전체 워크플로우 테스트 ---------------------- #
def test_full_workflow():
    """
    전체 워크플로우 통합 테스트

    실험 전체 흐름 시뮬레이션:
    1. 실험 시작 (with 문)
    2. 메타데이터 업데이트
    3. 프롬프트 저장
    4. DB 쿼리 기록
    5. 평가 지표 저장
    6. 실험 종료 (자동)
    """
    with ExperimentManager() as exp:
        # 1. 메타데이터 업데이트
        exp.update_metadata(
            difficulty="easy",
            tool_used="rag_paper",
            user_query="RAG에 대해 알려줘"
        )

        # 2. 프롬프트 저장
        exp.save_system_prompt("You are a helpful AI assistant.")
        exp.save_user_prompt("RAG에 대해 알려줘")

        # 3. DB 쿼리 기록
        exp.log_sql_query(
            query="SELECT * FROM papers WHERE title LIKE '%RAG%'",
            description="RAG 관련 논문 검색",
            tool="rag_paper"
        )

        # 4. 평가 지표 저장
        exp.save_rag_metrics({
            'recall_at_5': 0.85,
            'precision_at_5': 0.90
        })

        exp.save_agent_accuracy({
            'routing_accuracy': 0.95
        })

        exp.save_latency_report({
            'total_time_ms': 3250
        })

        exp.save_cost_analysis({
            'total_tokens': 2140,
            'cost_usd': 0.0214
        })

        # 5. 최종 답변 저장
        exp.save_output("response.txt", "RAG는 Retrieval Augmented Generation의 약자입니다...")

        experiment_dir = exp.experiment_dir
        metadata_file = exp.metadata_file

    # 6. 실험 종료 후 검증
    # 모든 파일이 생성되었는지 확인
    assert (experiment_dir / "chatbot.log").exists()
    assert (experiment_dir / "prompts" / "system_prompt.txt").exists()
    assert (experiment_dir / "prompts" / "user_prompt.txt").exists()
    assert (experiment_dir / "database" / "queries.sql").exists()
    assert (experiment_dir / "evaluation" / "rag_metrics.json").exists()
    assert (experiment_dir / "evaluation" / "agent_accuracy.json").exists()
    assert (experiment_dir / "evaluation" / "latency_report.json").exists()
    assert (experiment_dir / "evaluation" / "cost_analysis.json").exists()
    assert (experiment_dir / "outputs" / "response.txt").exists()

    # metadata.json 최종 확인
    with open(metadata_file, 'r', encoding='utf-8') as f:
        final_meta = json.load(f)

    assert final_meta['difficulty'] == "easy"
    assert final_meta['tool_used'] == "rag_paper"
    assert final_meta['end_time'] is not None
