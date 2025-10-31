"""PDF 문서를 로드하고 청크로 분할하는 스크립트.

사용법:
    python scripts/process_documents.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.document_loader import PaperDocumentLoader


def main() -> int:
    """PDF 문서를 로드하고 청크로 분할합니다."""
    
    # 설정
    pdf_dir = ROOT / "data/raw/pdfs"
    metadata_path = ROOT / "data/raw/arxiv_papers_metadata.json"
    
    # 파일 존재 확인
    if not pdf_dir.exists():
        print(f"❌ PDF 디렉토리가 없습니다: {pdf_dir}")
        return 1
    
    if not metadata_path.exists():
        print(f"❌ 메타데이터 파일이 없습니다: {metadata_path}")
        print("먼저 scripts/collect_arxiv_papers.py를 실행하세요.")
        return 1
    
    # Document 로더 초기화
    loader = PaperDocumentLoader(chunk_size=1000, chunk_overlap=200)
    
    print("=" * 60)
    print("PDF 문서 로드 및 청크 분할")
    print("=" * 60)
    print(f"PDF 디렉토리: {pdf_dir}")
    print(f"메타데이터: {metadata_path}")
    print()
    
    # PDF 로드 및 분할
    print("PDF 파일 로드 중...")
    try:
        chunks = loader.load_all_pdfs(pdf_dir, metadata_path)
        print(f"✅ 총 {len(chunks)}개 청크 생성 완료")
        
        # 샘플 출력
        if chunks:
            print("\n첫 번째 청크 예시:")
            print("-" * 60)
            print(chunks[0].page_content[:300] + "...")
            print(f"\n메타데이터: {chunks[0].metadata}")
        
        return 0
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

