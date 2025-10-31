#!/usr/bin/env python3
# ---------------------- 실험 검색 스크립트 ---------------------- #
"""
metadata.json 기반 실험 검색 도구

주요 기능:
- 난이도별 실험 검색 (easy/hard)
- 사용 도구별 검색 (rag_paper, web_search 등)
- 날짜별 검색 (YYYYMMDD)
- 응답 시간 기준 검색 (min/max)
- 복합 조건 검색
"""

# ------------------------- 표준 라이브러리 ------------------------- #
import json                                    # JSON 파일 처리
import argparse                                # 명령줄 인자 처리
from pathlib import Path                       # 파일 경로 처리
from typing import Optional, List, Dict        # 타입 힌팅


# ==================== 검색 함수 ==================== #
# ---------------------- 실험 검색 함수 ---------------------- #
def find_experiments(
    difficulty: Optional[str] = None,
    tool: Optional[str] = None,
    date: Optional[str] = None,
    min_response_time: Optional[int] = None,
    max_response_time: Optional[int] = None,
    min_success: Optional[bool] = None
) -> List[Dict]:
    """
    metadata.json 기반 실험 검색

    Args:
        difficulty: 난이도 필터 (easy/hard)
        tool: 도구 필터 (rag_paper, web_search 등)
        date: 날짜 필터 (YYYYMMDD)
        min_response_time: 최소 응답 시간 (ms)
        max_response_time: 최대 응답 시간 (ms)
        min_success: 성공 여부 필터 (True/False)

    Returns:
        검색된 실험 목록 (경로 및 메타데이터)
    """
    results = []                               # 검색 결과 리스트

    # -------------- 검색 경로 설정 -------------- #
    if date:
        # 특정 날짜 검색
        search_path = Path(f"experiments/{date}")
        if not search_path.exists():
            print(f"날짜 폴더가 존재하지 않습니다: experiments/{date}")
            return []
    else:
        # 전체 검색
        search_path = Path("experiments")
        if not search_path.exists():
            print("experiments 폴더가 존재하지 않습니다")
            return []

    # -------------- 모든 metadata.json 찾기 -------------- #
    # 재귀적으로 모든 metadata.json 파일 검색
    for meta_file in search_path.rglob("metadata.json"):
        try:
            # metadata.json 파일 읽기
            with open(meta_file, encoding='utf-8') as f:
                meta = json.load(f)
        except Exception as e:
            print(f"메타데이터 읽기 실패: {meta_file} - {e}")
            continue

        # -------------- 필터 적용 -------------- #
        # 난이도 필터
        if difficulty and meta.get('difficulty') != difficulty:
            continue

        # 도구 필터
        if tool and meta.get('tool_used') != tool:
            continue

        # 최소 응답 시간 필터
        if min_response_time and meta.get('response_time_ms', 0) < min_response_time:
            continue

        # 최대 응답 시간 필터
        if max_response_time and meta.get('response_time_ms', float('inf')) > max_response_time:
            continue

        # 성공 여부 필터
        if min_success is not None and meta.get('success') != min_success:
            continue

        # 검색 조건을 모두 통과한 실험 추가
        results.append({
            'path': str(meta_file.parent),     # 실험 폴더 경로
            'metadata': meta                   # 메타데이터
        })

    # 시작 시간 기준으로 정렬
    results.sort(key=lambda x: x['metadata'].get('start_time', ''))

    return results


# ---------------------- 검색 결과 출력 함수 ---------------------- #
def print_results(results: List[Dict], verbose: bool = False):
    """
    검색 결과 출력

    Args:
        results: 검색 결과 리스트
        verbose: 상세 출력 여부
    """
    if not results:
        print("\n검색 결과가 없습니다.")
        return

    print(f"\n총 {len(results)}개의 실험을 찾았습니다.\n")
    print("=" * 80)

    for idx, exp in enumerate(results, 1):
        meta = exp['metadata']
        path = exp['path']

        # 기본 정보 출력
        print(f"\n[{idx}] {Path(path).name}")
        print(f"경로: {path}")
        print(f"Session ID: {meta.get('session_id', 'N/A')}")
        print(f"시작 시간: {meta.get('start_time', 'N/A')}")
        print(f"난이도: {meta.get('difficulty', 'N/A')}")
        print(f"사용 도구: {meta.get('tool_used', 'N/A')}")
        print(f"사용자 질문: {meta.get('user_query', 'N/A')}")

        # 상세 정보 출력 (verbose 모드)
        if verbose:
            print(f"종료 시간: {meta.get('end_time', 'N/A')}")
            print(f"성공 여부: {meta.get('success', 'N/A')}")
            print(f"응답 시간: {meta.get('response_time_ms', 'N/A')} ms")
            print(f"응답 길이: {meta.get('response_length', 'N/A')} 자")

            # 토큰 사용량 출력
            tokens = meta.get('tokens_used', {})
            if tokens:
                print(f"토큰 사용: {tokens.get('total', 'N/A')} (프롬프트: {tokens.get('prompt', 'N/A')}, 완료: {tokens.get('completion', 'N/A')})")

        print("-" * 80)


# ==================== 메인 실행부 ==================== #
# ---------------------- 메인 함수 ---------------------- #
def main():
    """메인 실행 함수"""
    # 명령줄 인자 파서 생성
    parser = argparse.ArgumentParser(
        description='metadata.json 기반 실험 검색',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
                사용 예시:
                # 난이도별 검색
                python scripts/find_experiments.py --difficulty easy

                # 도구별 검색
                python scripts/find_experiments.py --tool rag_paper

                # 날짜별 검색
                python scripts/find_experiments.py --date 20251031

                # 응답 시간 기준 검색 (빠른 실험만)
                python scripts/find_experiments.py --max-time 3000

                # 복합 조건 검색
                python scripts/find_experiments.py --difficulty easy --tool rag_paper --min-time 2000 --max-time 5000

                # 상세 정보 출력
                python scripts/find_experiments.py --difficulty easy --verbose
                """
    )

    # 필터 옵션 추가
    parser.add_argument(
        '--difficulty',
        choices=['easy', 'hard'],
        help='난이도 필터 (easy 또는 hard)'
    )

    parser.add_argument(
        '--tool',
        help='사용 도구 필터 (rag_paper, rag_glossary, web_search, summary_paper, file_save, general)'
    )

    parser.add_argument(
        '--date',
        help='날짜 필터 (YYYYMMDD 형식, 예: 20251031)'
    )

    parser.add_argument(
        '--min-time',
        type=int,
        help='최소 응답 시간 (밀리초)'
    )

    parser.add_argument(
        '--max-time',
        type=int,
        help='최대 응답 시간 (밀리초)'
    )

    parser.add_argument(
        '--success-only',
        action='store_true',
        help='성공한 실험만 검색'
    )

    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='상세 정보 출력'
    )

    # 인자 파싱
    args = parser.parse_args()

    # 검색 실행
    print("\n실험 검색 중...")
    results = find_experiments(
        difficulty=args.difficulty,
        tool=args.tool,
        date=args.date,
        min_response_time=args.min_time,
        max_response_time=args.max_time,
        min_success=True if args.success_only else None
    )

    # 결과 출력
    print_results(results, verbose=args.verbose)


# ---------------------- 스크립트 직접 실행 시 ---------------------- #
if __name__ == "__main__":
    main()
