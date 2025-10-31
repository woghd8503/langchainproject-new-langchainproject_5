# ---------------------- 실험 관리 시스템 모듈 ---------------------- #
"""
실험 폴더 생성 및 관리 시스템

주요 기능:
- Session ID 자동 부여
- 실험 폴더 자동 생성 (tools, database, prompts, ui, outputs, evaluation)
- metadata.json 중앙 관리
- 도구별 Logger 제공
- DB 쿼리 및 검색 결과 저장
- 프롬프트 기록
- UI 인터랙션 로그
- 평가 지표 저장
"""

# ------------------------- 표준 라이브러리 ------------------------- #
import json                                    # JSON 파일 처리
from pathlib import Path                       # 파일 경로 처리
from datetime import datetime                  # 날짜 및 시간 처리
from typing import Dict, Optional              # 타입 힌팅

# ------------------------- 서드파티 라이브러리 ------------------------- #
import yaml                                    # YAML 파일 처리

# ------------------------- 프로젝트 모듈 ------------------------- #
from src.utils.logger import Logger            # Logger 클래스


# ==================== ExperimentManager 클래스 정의 ==================== #
class ExperimentManager:
    """
    실험 폴더 생성 및 관리 클래스

    DB, UI, 프롬프트 등 모든 정보를 체계적으로 기록
    """

    # ---------------------- 초기화 메서드 ---------------------- #
    def __init__(self):
        """실험 매니저 초기화"""
        # 현재 날짜 및 시간 생성
        today = datetime.now().strftime("%Y%m%d")          # 날짜 (YYYYMMDD)
        time_now = datetime.now().strftime("%H%M%S")       # 시간 (HHMMSS)

        # 당일 Session ID 생성
        session_id = self._get_next_session_id(today)

        # 메인 실험 폴더 경로 생성
        self.experiment_dir = Path(
            f"experiments/{today}/{today}_{time_now}_session_{session_id:03d}"
        )
        self.experiment_dir.mkdir(parents=True, exist_ok=True)

        # 서브 폴더 경로 설정
        self.tools_dir = self.experiment_dir / "tools"              # 도구 실행 로그
        self.database_dir = self.experiment_dir / "database"        # DB 관련 기록
        self.prompts_dir = self.experiment_dir / "prompts"          # 프롬프트 기록
        self.ui_dir = self.experiment_dir / "ui"                    # UI 관련 기록
        self.outputs_dir = self.experiment_dir / "outputs"          # 생성된 결과물
        self.evaluation_dir = self.experiment_dir / "evaluation"    # 평가 지표
        self.debug_dir = self.experiment_dir / "debug"              # 디버그 정보 (선택)

        # 필수 폴더 6개 생성
        for folder in [self.tools_dir, self.database_dir, self.prompts_dir,
                       self.ui_dir, self.outputs_dir, self.evaluation_dir]:
            folder.mkdir(exist_ok=True)

        # 메타데이터 초기화
        self.metadata = {
            'session_id': f"{session_id:03d}",               # Session ID
            'start_time': datetime.now().isoformat(),        # 시작 시간
            'difficulty': None,                              # 난이도
            'tool_used': None,                               # 사용된 도구
            'user_query': None,                              # 사용자 질문
            'success': None,                                 # 성공 여부
            'response_time_ms': None,                        # 응답 시간 (밀리초)
            'end_time': None                                 # 종료 시간
        }

        self.metadata_file = self.experiment_dir / "metadata.json"

        # Logger 초기화
        self.logger = Logger(str(self.experiment_dir / "chatbot.log"))
        self.logger.write(f"세션 시작: session_{session_id:03d}")
        self.logger.write(f"폴더 경로: {self.experiment_dir}")

        # DB 관련 변수 초기화
        self.db_queries = []                                 # SQL 쿼리 리스트
        self.pgvector_searches = []                          # pgvector 검색 기록
        self.search_results = {}                             # DB 검색 결과
        self.db_performance = {                              # DB 성능 정보
            'summary': {},
            'queries': []
        }


    # ---------------------- Session ID 자동 부여 메서드 ---------------------- #
    def _get_next_session_id(self, date: str) -> int:
        """
        당일 다음 Session ID 반환

        Args:
            date: 날짜 (YYYYMMDD 형식)

        Returns:
            다음 Session ID (1부터 시작)

        예시:
            - 첫 실행: session_001
            - 두 번째 실행: session_002
            - 다음 날: 다시 session_001부터 시작
        """
        date_dir = Path(f"experiments/{date}")

        # 날짜 폴더가 없으면 첫 실행
        if not date_dir.exists():
            return 1

        # 기존 Session 폴더 찾기
        existing_sessions = list(date_dir.glob(f"{date}_*_session_*"))
        if not existing_sessions:
            return 1

        # 가장 큰 Session ID 찾기
        max_id = 0
        for session_dir in existing_sessions:
            try:
                session_id_str = session_dir.name.split('_')[-1]
                session_id = int(session_id_str)
                max_id = max(max_id, session_id)
            except (IndexError, ValueError):
                continue

        return max_id + 1


    # ==================== 도구 로그 메서드 ==================== #
    # ---------------------- 도구별 Logger 생성 ---------------------- #
    def get_tool_logger(self, tool_name: str) -> Logger:
        """
        도구별 Logger 생성

        Args:
            tool_name: 도구명 (rag_paper, rag_glossary, web_search 등)

        Returns:
            Logger 인스턴스
        """
        log_path = self.tools_dir / f"{tool_name}.log"
        return Logger(str(log_path))


    # ==================== DB 관련 메서드 ==================== #
    # ---------------------- SQL 쿼리 기록 ---------------------- #
    def log_sql_query(
        self,
        query: str,
        description: str = "",
        tool: str = "",
        execution_time_ms: Optional[int] = None
    ):
        """
        SQL 쿼리 기록

        Args:
            query: SQL 쿼리문
            description: 쿼리 설명
            tool: 사용한 도구명
            execution_time_ms: 실행 시간 (밀리초)
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # 쿼리 레코드 생성
        query_record = f"""-- 실행 시간: {timestamp}
                            -- 도구: {tool}
                            -- 설명: {description}
                        """
        if execution_time_ms:
            query_record += f"-- 실행 소요: {execution_time_ms}ms\n"

        query_record += f"\n{query};\n\n"

        self.db_queries.append(query_record)

        # queries.sql 파일에 즉시 추가
        with open(self.database_dir / "queries.sql", 'a', encoding='utf-8') as f:
            f.write(query_record)

        self.logger.write(f"SQL 쿼리 기록: {description}")


    # ---------------------- pgvector 검색 기록 ---------------------- #
    def log_pgvector_search(self, search_info: Dict):
        """
        pgvector 검색 기록

        Args:
            search_info: 검색 정보 딕셔너리
                - tool: 도구명
                - collection: 컬렉션명
                - query_text: 검색 쿼리
                - top_k: 검색 결과 수
                - execution_time_ms: 실행 시간
                등
        """
        search_info['timestamp'] = datetime.now().isoformat()
        self.pgvector_searches.append(search_info)

        # pgvector_searches.json 업데이트
        with open(self.database_dir / "pgvector_searches.json", 'w', encoding='utf-8') as f:
            json.dump(self.pgvector_searches, f, ensure_ascii=False, indent=2)

        self.logger.write(f"pgvector 검색 기록: {search_info.get('tool', 'unknown')}")


    # ---------------------- DB 검색 결과 저장 ---------------------- #
    def save_search_results(self, tool_name: str, results: Dict):
        """
        DB 검색 결과 저장

        Args:
            tool_name: 도구명
            results: 검색 결과 딕셔너리
        """
        results['timestamp'] = datetime.now().isoformat()
        self.search_results[tool_name] = results

        # search_results.json 업데이트
        with open(self.database_dir / "search_results.json", 'w', encoding='utf-8') as f:
            json.dump(self.search_results, f, ensure_ascii=False, indent=2)

        self.logger.write(f"검색 결과 저장: {tool_name}")


    # ---------------------- DB 성능 정보 저장 ---------------------- #
    def save_db_performance(self, performance_data: Dict):
        """
        DB 성능 정보 저장

        Args:
            performance_data: 성능 데이터 딕셔너리
        """
        with open(self.database_dir / "db_performance.json", 'w', encoding='utf-8') as f:
            json.dump(performance_data, f, ensure_ascii=False, indent=2)

        self.logger.write("DB 성능 정보 저장 완료")


    # ==================== 프롬프트 관련 메서드 ==================== #
    # ---------------------- 시스템 프롬프트 저장 ---------------------- #
    def save_system_prompt(self, system_prompt: str, metadata: Optional[Dict] = None):
        """
        시스템 프롬프트 저장

        Args:
            system_prompt: 시스템 프롬프트 텍스트
            metadata: 메타데이터 딕셔너리
        """
        content = system_prompt

        if metadata:
            content += "\n\n===== 메타데이터 =====\n"
            for key, value in metadata.items():
                content += f"{key}: {value}\n"

        with open(self.prompts_dir / "system_prompt.txt", 'w', encoding='utf-8') as f:
            f.write(content)

        self.logger.write("시스템 프롬프트 저장 완료")


    # ---------------------- 사용자 프롬프트 저장 ---------------------- #
    def save_user_prompt(self, user_prompt: str, metadata: Optional[Dict] = None):
        """
        사용자 프롬프트 저장

        Args:
            user_prompt: 사용자 프롬프트 텍스트
            metadata: 메타데이터 딕셔너리
        """
        content = user_prompt

        if metadata:
            content += "\n\n===== 메타데이터 =====\n"
            for key, value in metadata.items():
                content += f"{key}: {value}\n"

        with open(self.prompts_dir / "user_prompt.txt", 'w', encoding='utf-8') as f:
            f.write(content)

        self.logger.write("사용자 프롬프트 저장 완료")


    # ---------------------- 최종 프롬프트 저장 ---------------------- #
    def save_final_prompt(self, final_prompt: str, metadata: Optional[Dict] = None):
        """
        최종 프롬프트 저장

        Args:
            final_prompt: 최종 프롬프트 텍스트
            metadata: 메타데이터 딕셔너리
        """
        content = final_prompt

        if metadata:
            content += "\n\n===== 메타데이터 =====\n"
            for key, value in metadata.items():
                content += f"{key}: {value}\n"

        with open(self.prompts_dir / "final_prompt.txt", 'w', encoding='utf-8') as f:
            f.write(content)

        self.logger.write("최종 프롬프트 저장 완료")


    # ---------------------- 프롬프트 템플릿 정보 저장 ---------------------- #
    def save_prompt_template(self, template_info: Dict):
        """
        프롬프트 템플릿 정보 저장

        Args:
            template_info: 템플릿 정보 딕셔너리
        """
        with open(self.prompts_dir / "prompt_template.yaml", 'w', encoding='utf-8') as f:
            yaml.dump(template_info, f, allow_unicode=True, sort_keys=False)

        self.logger.write("프롬프트 템플릿 저장 완료")


    # ==================== UI 관련 메서드 ==================== #
    # ---------------------- Streamlit 세션 상태 저장 ---------------------- #
    def save_streamlit_session(self, session_data: Dict):
        """
        Streamlit 세션 상태 저장

        Args:
            session_data: 세션 데이터 딕셔너리
        """
        with open(self.ui_dir / "streamlit_session.json", 'w', encoding='utf-8') as f:
            json.dump(session_data, f, ensure_ascii=False, indent=2)

        self.logger.write("Streamlit 세션 저장 완료")


    # ---------------------- UI 인터랙션 로그 ---------------------- #
    def log_ui_interaction(self, interaction: str):
        """
        UI 인터랙션 로그

        Args:
            interaction: 인터랙션 설명
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_line = f"{timestamp} | {interaction}\n"

        with open(self.ui_dir / "user_interactions.log", 'a', encoding='utf-8') as f:
            f.write(log_line)


    # ---------------------- UI 이벤트 기록 ---------------------- #
    def log_ui_event(self, event: Dict):
        """
        UI 이벤트 기록

        Args:
            event: 이벤트 딕셔너리
        """
        event['timestamp'] = datetime.now().isoformat()

        # 기존 이벤트 읽기
        events_file = self.ui_dir / "ui_events.json"
        if events_file.exists():
            with open(events_file, 'r', encoding='utf-8') as f:
                events = json.load(f)
        else:
            events = []

        events.append(event)

        # 업데이트
        with open(events_file, 'w', encoding='utf-8') as f:
            json.dump(events, f, ensure_ascii=False, indent=2)


    # ==================== 출력 관련 메서드 ==================== #
    # ---------------------- 결과물 저장 ---------------------- #
    def save_output(self, filename: str, content: str):
        """
        결과물 저장

        Args:
            filename: 파일명
            content: 내용
        """
        output_path = self.outputs_dir / filename
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(content)

        self.logger.write(f"결과물 저장: {filename}")


    # ==================== 평가 지표 관련 메서드 ==================== #
    # ---------------------- RAG 평가 지표 저장 ---------------------- #
    def save_rag_metrics(self, metrics: Dict):
        """
        RAG 평가 지표 저장

        Args:
            metrics: RAG 평가 지표 딕셔너리
        """
        metrics['timestamp'] = datetime.now().isoformat()

        with open(self.evaluation_dir / "rag_metrics.json", 'w', encoding='utf-8') as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)

        self.logger.write("RAG 평가 지표 저장 완료")


    # ---------------------- Agent 정확도 저장 ---------------------- #
    def save_agent_accuracy(self, accuracy_data: Dict):
        """
        Agent 정확도 저장

        Args:
            accuracy_data: Agent 정확도 데이터 딕셔너리
        """
        accuracy_data['timestamp'] = datetime.now().isoformat()

        with open(self.evaluation_dir / "agent_accuracy.json", 'w', encoding='utf-8') as f:
            json.dump(accuracy_data, f, ensure_ascii=False, indent=2)

        self.logger.write("Agent 정확도 저장 완료")


    # ---------------------- 응답 시간 분석 저장 ---------------------- #
    def save_latency_report(self, latency_data: Dict):
        """
        응답 시간 분석 저장

        Args:
            latency_data: 응답 시간 데이터 딕셔너리
        """
        latency_data['timestamp'] = datetime.now().isoformat()

        with open(self.evaluation_dir / "latency_report.json", 'w', encoding='utf-8') as f:
            json.dump(latency_data, f, ensure_ascii=False, indent=2)

        self.logger.write("응답 시간 분석 저장 완료")


    # ---------------------- 비용 분석 저장 ---------------------- #
    def save_cost_analysis(self, cost_data: Dict):
        """
        비용 분석 저장

        Args:
            cost_data: 비용 분석 데이터 딕셔너리
        """
        cost_data['timestamp'] = datetime.now().isoformat()

        with open(self.evaluation_dir / "cost_analysis.json", 'w', encoding='utf-8') as f:
            json.dump(cost_data, f, ensure_ascii=False, indent=2)

        self.logger.write("비용 분석 저장 완료")


    # ---------------------- 테스트 결과 저장 ---------------------- #
    def save_test_results(self, test_data: Dict):
        """
        테스트 결과 저장

        Args:
            test_data: 테스트 결과 데이터 딕셔너리
        """
        test_data['timestamp'] = datetime.now().isoformat()

        with open(self.evaluation_dir / "test_results.json", 'w', encoding='utf-8') as f:
            json.dump(test_data, f, ensure_ascii=False, indent=2)

        self.logger.write("테스트 결과 저장 완료")


    # ==================== 디버그 관련 메서드 ==================== #
    # ---------------------- 디버그 정보 저장 ---------------------- #
    def save_debug_info(self, filename: str, data: Dict):
        """
        디버그 정보 저장

        Args:
            filename: 파일명
            data: 디버그 데이터
        """
        # debug 폴더가 없으면 생성
        self.debug_dir.mkdir(exist_ok=True)

        debug_path = self.debug_dir / filename
        with open(debug_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)


    # ==================== 메타데이터 관련 메서드 ==================== #
    # ---------------------- 메타데이터 업데이트 ---------------------- #
    def update_metadata(self, **kwargs):
        """
        메타데이터 업데이트

        Args:
            **kwargs: 업데이트할 키-값 쌍
        """
        self.metadata.update(kwargs)

        with open(self.metadata_file, 'w', encoding='utf-8') as f:
            json.dump(self.metadata, f, ensure_ascii=False, indent=2)

        self.logger.write(f"메타데이터 업데이트: {list(kwargs.keys())}")


    # ---------------------- 전체 설정 저장 ---------------------- #
    def save_config(self, config: Dict):
        """
        전체 설정 저장

        Args:
            config: 설정 딕셔너리
        """
        config_path = self.experiment_dir / "config.yaml"
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, allow_unicode=True, sort_keys=False)

        self.logger.write("설정 파일 저장 완료")


    # ---------------------- 실험 종료 ---------------------- #
    def close(self):
        """실험 종료"""
        # 종료 시간 기록
        self.metadata['end_time'] = datetime.now().isoformat()

        # 최종 메타데이터 저장
        with open(self.metadata_file, 'w', encoding='utf-8') as f:
            json.dump(self.metadata, f, ensure_ascii=False, indent=2)

        self.logger.write("=" * 50)
        self.logger.write("실험 종료")
        self.logger.write("=" * 50)
        self.logger.close()


    # ==================== Context Manager 지원 ==================== #
    # ---------------------- with 문 진입 ---------------------- #
    def __enter__(self):
        """with 문 지원"""
        return self


    # ---------------------- with 문 종료 ---------------------- #
    def __exit__(self, exc_type, exc_val, exc_tb):
        """with 문 종료 시 자동 close"""
        self.close()
