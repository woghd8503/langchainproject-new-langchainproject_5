#!/usr/bin/env python3
# ---------------------- 평가 지표 집계 스크립트 ---------------------- #
"""
실험 평가 지표 집계 도구

주요 기능:
- 여러 실험의 evaluation/ 폴더 데이터 수집
- RAG 평가 지표 집계 (Recall, Precision, Faithfulness)
- Agent 정확도 집계 (도구 선택 정확도)
- 응답 시간 집계 (p50, p95, p99)
- 비용 분석 집계 (토큰 사용량, USD/KRW)
- CSV/JSON 형식으로 결과 저장
"""

# ------------------------- 표준 라이브러리 ------------------------- #
import json                                    # JSON 파일 처리
import csv                                     # CSV 파일 처리
import argparse                                # 명령줄 인자 처리
from pathlib import Path                       # 파일 경로 처리
from datetime import datetime                  # 날짜 및 시간 처리
from typing import Dict, List, Optional        # 타입 힌팅
from statistics import mean, median            # 통계 함수


# ==================== 집계 함수들 ==================== #
# ---------------------- RAG 평가 지표 집계 ---------------------- #
def aggregate_rag_metrics(date: str) -> Dict:
    """
    RAG 평가 지표 집계

    Args:
        date: 날짜 (YYYYMMDD)

    Returns:
        집계된 RAG 평가 지표
    """
    metrics_list = []                          # RAG 평가 지표 리스트

    # 날짜별 실험 폴더 검색
    date_dir = Path(f"experiments/{date}")
    if not date_dir.exists():
        print(f"날짜 폴더가 존재하지 않습니다: {date}")
        return {}

    # 모든 session 폴더 순회
    for session_dir in date_dir.glob("*_session_*"):
        rag_metrics_file = session_dir / "evaluation" / "rag_metrics.json"

        # rag_metrics.json 파일이 있으면 로드
        if rag_metrics_file.exists():
            try:
                with open(rag_metrics_file, encoding='utf-8') as f:
                    metrics = json.load(f)
                    metrics_list.append(metrics)
            except Exception as e:
                print(f"RAG 메트릭 읽기 실패: {rag_metrics_file} - {e}")

    # 데이터가 없으면 빈 딕셔너리 반환
    if not metrics_list:
        return {}

    # -------------- 집계 계산 -------------- #
    # Retrieval 지표 집계 (flat 및 nested 구조 모두 지원)
    recall_values = []
    precision_values = []
    faithfulness_values = []
    relevancy_values = []

    for m in metrics_list:
        # flat 구조 우선 확인, 없으면 nested 구조 확인
        recall_values.append(
            m.get('recall_at_5') or m.get('retrieval_metrics', {}).get('recall_at_5', 0)
        )
        precision_values.append(
            m.get('precision_at_5') or m.get('retrieval_metrics', {}).get('precision_at_5', 0)
        )
        faithfulness_values.append(
            m.get('faithfulness') or m.get('generation_metrics', {}).get('faithfulness', 0)
        )
        relevancy_values.append(
            m.get('answer_relevancy') or m.get('generation_metrics', {}).get('answer_relevancy', 0)
        )

    return {
        'total_sessions': len(metrics_list),           # 총 세션 수
        'recall_at_5': {
            'mean': mean(recall_values),               # 평균
            'min': min(recall_values),                 # 최소값
            'max': max(recall_values),                 # 최대값
            'median': median(recall_values)            # 중앙값
        },
        'precision_at_5': {
            'mean': mean(precision_values),
            'min': min(precision_values),
            'max': max(precision_values),
            'median': median(precision_values)
        },
        'faithfulness': {
            'mean': mean(faithfulness_values),
            'min': min(faithfulness_values),
            'max': max(faithfulness_values),
            'median': median(faithfulness_values)
        },
        'answer_relevancy': {
            'mean': mean(relevancy_values),
            'min': min(relevancy_values),
            'max': max(relevancy_values),
            'median': median(relevancy_values)
        }
    }


# ---------------------- Agent 정확도 집계 ---------------------- #
def aggregate_agent_accuracy(date: str) -> Dict:
    """
    Agent 정확도 집계

    Args:
        date: 날짜 (YYYYMMDD)

    Returns:
        집계된 Agent 정확도
    """
    accuracy_list = []                         # Agent 정확도 리스트

    # 날짜별 실험 폴더 검색
    date_dir = Path(f"experiments/{date}")
    if not date_dir.exists():
        return {}

    # 모든 session 폴더 순회
    for session_dir in date_dir.glob("*_session_*"):
        accuracy_file = session_dir / "evaluation" / "agent_accuracy.json"

        # agent_accuracy.json 파일이 있으면 로드
        if accuracy_file.exists():
            try:
                with open(accuracy_file, encoding='utf-8') as f:
                    accuracy = json.load(f)
                    accuracy_list.append(accuracy)
            except Exception as e:
                print(f"Agent 정확도 읽기 실패: {accuracy_file} - {e}")

    # 데이터가 없으면 빈 딕셔너리 반환
    if not accuracy_list:
        return {}

    # -------------- 집계 계산 -------------- #
    # 도구 선택 정확도 집계 (flat 및 nested 구조 모두 지원)
    routing_accuracy_values = []
    correct_decisions = 0
    incorrect_decisions = 0
    confidence_values = []

    for a in accuracy_list:
        # flat 구조 우선 확인
        if 'routing_accuracy' in a:
            routing_accuracy_values.append(a.get('routing_accuracy', 0))
        if 'correct_decisions' in a:
            correct_decisions += a.get('correct_decisions', 0)
        if 'incorrect_decisions' in a:
            incorrect_decisions += a.get('incorrect_decisions', 0)
        if 'average_confidence' in a:
            confidence_values.append(a.get('average_confidence', 0))

        # nested 구조 확인
        if 'routing_decision' in a:
            rd = a.get('routing_decision', {})
            if rd.get('correct', False):
                correct_decisions += 1
            else:
                incorrect_decisions += 1
            confidence_values.append(rd.get('confidence', 0))

    total_decisions = correct_decisions + incorrect_decisions
    total_count = len(accuracy_list)

    return {
        'total_sessions': total_count,                 # 총 세션 수
        'total_decisions': total_decisions,            # 총 결정 수
        'routing_accuracy': mean(routing_accuracy_values) if routing_accuracy_values else (correct_decisions / total_decisions if total_decisions > 0 else 0),
        'correct_decisions': correct_decisions,        # 정확한 선택 수
        'incorrect_decisions': incorrect_decisions,    # 잘못된 선택 수
        'average_confidence': mean(confidence_values) if confidence_values else 0     # 평균 신뢰도
    }


# ---------------------- 응답 시간 집계 ---------------------- #
def aggregate_latency(date: str) -> Dict:
    """
    응답 시간 집계

    Args:
        date: 날짜 (YYYYMMDD)

    Returns:
        집계된 응답 시간
    """
    latency_list = []                          # 응답 시간 리스트

    # 날짜별 실험 폴더 검색
    date_dir = Path(f"experiments/{date}")
    if not date_dir.exists():
        return {}

    # 모든 session 폴더 순회
    for session_dir in date_dir.glob("*_session_*"):
        latency_file = session_dir / "evaluation" / "latency_report.json"

        # latency_report.json 파일이 있으면 로드
        if latency_file.exists():
            try:
                with open(latency_file, encoding='utf-8') as f:
                    latency = json.load(f)
                    latency_list.append(latency)
            except Exception as e:
                print(f"응답 시간 읽기 실패: {latency_file} - {e}")

    # 데이터가 없으면 빈 딕셔너리 반환
    if not latency_list:
        return {}

    # -------------- 집계 계산 -------------- #
    # 전체 응답 시간 집계 (flat 및 nested 구조 모두 지원)
    total_times = []
    routing_times = []
    retrieval_times = []
    generation_times = []

    for l in latency_list:
        # flat 구조 우선 확인, 없으면 nested 구조 확인
        total_times.append(
            l.get('total_time_ms') or l.get('total_latency', {}).get('total_time_ms', 0)
        )
        routing_times.append(
            l.get('routing_time_ms') or l.get('breakdown', {}).get('routing_time_ms', 0)
        )
        retrieval_times.append(
            l.get('retrieval_time_ms') or l.get('breakdown', {}).get('retrieval_time_ms', 0)
        )
        generation_times.append(
            l.get('generation_time_ms') or l.get('breakdown', {}).get('generation_time_ms', 0)
        )

    return {
        'total_sessions': len(latency_list),           # 총 세션 수
        'total_time_ms': {
            'mean': mean(total_times),                 # 평균
            'min': min(total_times),                   # 최소값
            'max': max(total_times),                   # 최대값
            'median': median(total_times),             # 중앙값
            'p95': sorted(total_times)[int(len(total_times) * 0.95)] if total_times else 0    # p95
        },
        'routing_time_ms': {
            'mean': mean(routing_times) if routing_times else 0,
            'median': median(routing_times) if routing_times else 0
        },
        'retrieval_time_ms': {
            'mean': mean(retrieval_times) if retrieval_times else 0,
            'median': median(retrieval_times) if retrieval_times else 0
        },
        'generation_time_ms': {
            'mean': mean(generation_times) if generation_times else 0,
            'median': median(generation_times) if generation_times else 0
        }
    }


# ---------------------- 비용 분석 집계 ---------------------- #
def aggregate_cost(date: str) -> Dict:
    """
    비용 분석 집계

    Args:
        date: 날짜 (YYYYMMDD)

    Returns:
        집계된 비용 분석
    """
    cost_list = []                             # 비용 분석 리스트

    # 날짜별 실험 폴더 검색
    date_dir = Path(f"experiments/{date}")
    if not date_dir.exists():
        return {}

    # 모든 session 폴더 순회
    for session_dir in date_dir.glob("*_session_*"):
        cost_file = session_dir / "evaluation" / "cost_analysis.json"

        # cost_analysis.json 파일이 있으면 로드
        if cost_file.exists():
            try:
                with open(cost_file, encoding='utf-8') as f:
                    cost = json.load(f)
                    cost_list.append(cost)
            except Exception as e:
                print(f"비용 분석 읽기 실패: {cost_file} - {e}")

    # 데이터가 없으면 빈 딕셔너리 반환
    if not cost_list:
        return {}

    # -------------- 집계 계산 -------------- #
    # 토큰 사용량 집계 (flat 및 nested 구조 모두 지원)
    total_tokens = []
    cost_usd = []
    cost_krw = []

    for c in cost_list:
        # flat 구조 우선 확인, 없으면 nested 구조 확인
        total_tokens.append(
            c.get('total_tokens') or c.get('llm_usage', {}).get('total_tokens', 0)
        )
        cost_usd.append(
            c.get('cost_usd') or c.get('cost_breakdown_usd', {}).get('total_cost', 0)
        )
        cost_krw.append(
            c.get('cost_krw') or c.get('cost_breakdown_krw', {}).get('total_cost', 0)
        )

    return {
        'total_sessions': len(cost_list),              # 총 세션 수
        'total_tokens': {
            'sum': sum(total_tokens),                  # 총합
            'mean': mean(total_tokens),                # 평균
            'min': min(total_tokens),                  # 최소값
            'max': max(total_tokens)                   # 최대값
        },
        'total_cost_usd': {
            'sum': sum(cost_usd),                      # 총 비용 (USD)
            'mean': mean(cost_usd),                    # 평균 비용
            'min': min(cost_usd),                      # 최소 비용
            'max': max(cost_usd)                       # 최대 비용
        },
        'total_cost_krw': {
            'sum': sum(cost_krw),                      # 총 비용 (KRW)
            'mean': mean(cost_krw),                    # 평균 비용
            'min': min(cost_krw),                      # 최소 비용
            'max': max(cost_krw)                       # 최대 비용
        }
    }


# ==================== 저장 함수들 ==================== #
# ---------------------- JSON 형식으로 저장 ---------------------- #
def save_as_json(aggregated_data: Dict, output_path: str):
    """
    집계 결과를 JSON 형식으로 저장

    Args:
        aggregated_data: 집계된 데이터
        output_path: 출력 파일 경로
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(aggregated_data, f, ensure_ascii=False, indent=2)

    print(f"\nJSON 파일 저장 완료: {output_path}")


# ---------------------- CSV 형식으로 저장 ---------------------- #
def save_as_csv(aggregated_data: Dict, output_path: str):
    """
    집계 결과를 CSV 형식으로 저장

    Args:
        aggregated_data: 집계된 데이터
        output_path: 출력 파일 경로
    """
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)

        # 헤더 작성
        writer.writerow(['Category', 'Metric', 'Value'])

        # RAG 메트릭 작성
        if 'rag_metrics' in aggregated_data:
            rag = aggregated_data['rag_metrics']
            if rag:
                writer.writerow(['RAG', 'Total Sessions', rag.get('total_sessions', 0)])
                writer.writerow(['RAG', 'Recall@5 (Mean)', f"{rag.get('recall_at_5', {}).get('mean', 0):.3f}"])
                writer.writerow(['RAG', 'Precision@5 (Mean)', f"{rag.get('precision_at_5', {}).get('mean', 0):.3f}"])
                writer.writerow(['RAG', 'Faithfulness (Mean)', f"{rag.get('faithfulness', {}).get('mean', 0):.3f}"])

        # Agent 정확도 작성
        if 'agent_accuracy' in aggregated_data:
            agent = aggregated_data['agent_accuracy']
            if agent:
                writer.writerow(['Agent', 'Total Sessions', agent.get('total_sessions', 0)])
                writer.writerow(['Agent', 'Routing Accuracy', f"{agent.get('routing_accuracy', 0):.3f}"])
                writer.writerow(['Agent', 'Correct Count', agent.get('correct_count', 0)])

        # 응답 시간 작성
        if 'latency' in aggregated_data:
            latency = aggregated_data['latency']
            if latency:
                writer.writerow(['Latency', 'Total Sessions', latency.get('total_sessions', 0)])
                writer.writerow(['Latency', 'Total Time (Mean) ms', f"{latency.get('total_time_ms', {}).get('mean', 0):.1f}"])
                writer.writerow(['Latency', 'Total Time (p95) ms', f"{latency.get('total_time_ms', {}).get('p95', 0):.1f}"])

        # 비용 작성
        if 'cost' in aggregated_data:
            cost = aggregated_data['cost']
            if cost:
                writer.writerow(['Cost', 'Total Sessions', cost.get('total_sessions', 0)])
                writer.writerow(['Cost', 'Total Tokens (Sum)', cost.get('total_tokens', {}).get('sum', 0)])
                writer.writerow(['Cost', 'Total Cost USD (Sum)', f"${cost.get('total_cost_usd', {}).get('sum', 0):.4f}"])
                writer.writerow(['Cost', 'Total Cost KRW (Sum)', f"₩{cost.get('total_cost_krw', {}).get('sum', 0):.2f}"])

    print(f"\nCSV 파일 저장 완료: {output_path}")


# ==================== 메인 실행부 ==================== #
# ---------------------- 메인 함수 ---------------------- #
def main():
    """메인 실행 함수"""
    # 명령줄 인자 파서 생성
    parser = argparse.ArgumentParser(
        description='실험 평가 지표 집계',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
                사용 예시:
                # 특정 날짜의 평가 지표 집계 (JSON 출력)
                python scripts/aggregate_metrics.py --date 20251031 --output results.json

                # 특정 날짜의 평가 지표 집계 (CSV 출력)
                python scripts/aggregate_metrics.py --date 20251031 --output results.csv

                # 모든 메트릭 집계
                python scripts/aggregate_metrics.py --date 20251031 --output results.json
                """
    )

    # 필수 인자 추가
    parser.add_argument(
        '--date',
        required=True,
        help='집계할 날짜 (YYYYMMDD 형식, 예: 20251031)'
    )

    parser.add_argument(
        '--output', '-o',
        required=True,
        help='출력 파일 경로 (.json 또는 .csv)'
    )

    # 인자 파싱
    args = parser.parse_args()

    print(f"\n{'='*80}")
    print(f"평가 지표 집계 시작: {args.date}")
    print(f"{'='*80}\n")

    # -------------- 평가 지표 집계 -------------- #
    aggregated_data = {}

    # RAG 평가 지표 집계
    print("RAG 평가 지표 집계 중...")
    aggregated_data['rag_metrics'] = aggregate_rag_metrics(args.date)

    # Agent 정확도 집계
    print("Agent 정확도 집계 중...")
    aggregated_data['agent_accuracy'] = aggregate_agent_accuracy(args.date)

    # 응답 시간 집계
    print("응답 시간 집계 중...")
    aggregated_data['latency'] = aggregate_latency(args.date)

    # 비용 분석 집계
    print("비용 분석 집계 중...")
    aggregated_data['cost'] = aggregate_cost(args.date)

    # 메타 정보 추가
    aggregated_data['meta'] = {
        'date': args.date,
        'aggregated_at': datetime.now().isoformat(),
        'total_experiments': aggregated_data.get('rag_metrics', {}).get('total_sessions', 0)
    }

    # -------------- 결과 저장 -------------- #
    output_path = args.output

    if output_path.endswith('.json'):
        save_as_json(aggregated_data, output_path)
    elif output_path.endswith('.csv'):
        save_as_csv(aggregated_data, output_path)
    else:
        print(f"\n지원하지 않는 파일 형식입니다: {output_path}")
        print("JSON(.json) 또는 CSV(.csv) 형식만 지원됩니다.")
        return

    # -------------- 요약 출력 -------------- #
    print(f"\n{'='*80}")
    print("집계 요약")
    print(f"{'='*80}")

    if aggregated_data.get('rag_metrics'):
        rag = aggregated_data['rag_metrics']
        print(f"\n[RAG 평가 지표]")
        print(f"  총 실험: {rag.get('total_sessions', 0)}개")
        print(f"  평균 Recall@5: {rag.get('recall_at_5', {}).get('mean', 0):.3f}")
        print(f"  평균 Faithfulness: {rag.get('faithfulness', {}).get('mean', 0):.3f}")

    if aggregated_data.get('agent_accuracy'):
        agent = aggregated_data['agent_accuracy']
        print(f"\n[Agent 정확도]")
        print(f"  총 실험: {agent.get('total_sessions', 0)}개")
        print(f"  도구 선택 정확도: {agent.get('routing_accuracy', 0):.1%}")

    if aggregated_data.get('latency'):
        latency = aggregated_data['latency']
        print(f"\n[응답 시간]")
        print(f"  총 실험: {latency.get('total_sessions', 0)}개")
        print(f"  평균 응답 시간: {latency.get('total_time_ms', {}).get('mean', 0):.1f} ms")
        print(f"  p95 응답 시간: {latency.get('total_time_ms', {}).get('p95', 0):.1f} ms")

    if aggregated_data.get('cost'):
        cost = aggregated_data['cost']
        print(f"\n[비용 분석]")
        print(f"  총 실험: {cost.get('total_sessions', 0)}개")
        print(f"  총 토큰 사용: {cost.get('total_tokens', {}).get('sum', 0):,}")
        print(f"  총 비용 (USD): ${cost.get('total_cost_usd', {}).get('sum', 0):.4f}")
        print(f"  총 비용 (KRW): ₩{cost.get('total_cost_krw', {}).get('sum', 0):.2f}")

    print(f"\n{'='*80}\n")


# ---------------------- 스크립트 직접 실행 시 ---------------------- #
if __name__ == "__main__":
    main()
