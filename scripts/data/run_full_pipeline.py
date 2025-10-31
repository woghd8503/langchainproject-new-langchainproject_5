"""전체 데이터 파이프라인을 순차적으로 실행하는 스크립트.

사용법:
    python scripts/run_full_pipeline.py
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def run_command(script_name: str, description: str) -> bool:
    """스크립트를 실행하고 결과를 반환합니다."""
    print("=" * 60)
    print(f"{description}")
    print("=" * 60)
    
    script_path = ROOT / "scripts" / script_name
    
    try:
        result = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=str(ROOT),
            check=True,
            capture_output=False,
        )
        print(f"✅ {description} 완료\n")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} 실패: {e}\n")
        return False
    except Exception as e:
        print(f"❌ 오류: {e}\n")
        return False


def main() -> int:
    """전체 파이프라인을 실행합니다."""
    
    print("=" * 60)
    print("전체 데이터 파이프라인 실행")
    print("=" * 60)
    print()
    
    steps = [
        ("collect_arxiv_papers.py", "Phase 1: arXiv 논문 수집"),
        ("setup_database.py", "Phase 2: PostgreSQL 데이터베이스 초기화"),
        ("process_documents.py", "Phase 3: PDF 문서 로드 및 청크 분할"),
        ("load_embeddings.py", "Phase 4: 임베딩 생성 및 Vector DB 저장"),
    ]
    
    for script, description in steps:
        success = run_command(script, description)
        if not success:
            print(f"⚠️ {description} 단계에서 실패했습니다.")
            print("다음 단계는 건너뜁니다.")
            return 1
    
    print("=" * 60)
    print("✅ 전체 파이프라인 실행 완료!")
    print("=" * 60)
    
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

