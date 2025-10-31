"""데이터 파이프라인 테스트 스크립트.

각 단계별로 기본 동작을 확인합니다.
"""

import sys
import os
from pathlib import Path

# Windows 콘솔 인코딩 설정
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def test_collector():
    """ArxivPaperCollector 클래스 테스트."""
    print("=" * 60)
    print("1. ArxivPaperCollector 테스트")
    print("=" * 60)
    
    try:
        from scripts.collect_arxiv_papers import ArxivPaperCollector
        
        collector = ArxivPaperCollector()
        print(f"[OK] ArxivPaperCollector 초기화 성공")
        print(f"   저장 디렉토리: {collector.save_dir}")
        print(f"   디렉토리 존재: {collector.save_dir.exists()}")
        
        # 메서드 확인
        assert hasattr(collector, 'collect_papers'), "collect_papers 메서드 없음"
        assert hasattr(collector, 'collect_by_keywords'), "collect_by_keywords 메서드 없음"
        assert hasattr(collector, 'remove_duplicates'), "remove_duplicates 메서드 없음"
        print("[OK] 모든 메서드 존재 확인")
        
        return True
    except Exception as e:
        print(f"[ERROR] 오류: {e}")
        return False


def test_document_loader():
    """PaperDocumentLoader 클래스 테스트."""
    print("\n" + "=" * 60)
    print("2. PaperDocumentLoader 테스트")
    print("=" * 60)
    
    try:
        from src.data.document_loader import PaperDocumentLoader
        
        loader = PaperDocumentLoader(chunk_size=1000, chunk_overlap=200)
        print("[OK] PaperDocumentLoader 초기화 성공")
        print(f"   chunk_size: {loader.text_splitter._chunk_size}")
        print(f"   chunk_overlap: {loader.text_splitter._chunk_overlap}")
        
        # 메서드 확인
        assert hasattr(loader, 'load_pdf'), "load_pdf 메서드 없음"
        assert hasattr(loader, 'load_and_split'), "load_and_split 메서드 없음"
        assert hasattr(loader, 'load_all_pdfs'), "load_all_pdfs 메서드 없음"
        print("[OK] 모든 메서드 존재 확인")
        
        return True
    except Exception as e:
        print(f"❌ 오류: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_embeddings():
    """PaperEmbeddingManager 클래스 테스트."""
    print("\n" + "=" * 60)
    print("3. PaperEmbeddingManager 테스트")
    print("=" * 60)
    
    try:
        from src.data.embeddings import PaperEmbeddingManager
        
        # 환경 변수가 없어도 초기화 구조는 확인 가능
        print("[OK] PaperEmbeddingManager 모듈 임포트 성공")
        
        # 메서드 확인 (클래스 구조 확인)
        import inspect
        methods = [m for m in dir(PaperEmbeddingManager) if not m.startswith('_')]
        print(f"[OK] 사용 가능한 메서드: {', '.join(methods)}")
        
        assert 'add_documents' in methods, "add_documents 메서드 없음"
        assert 'add_documents_with_paper_id' in methods, "add_documents_with_paper_id 메서드 없음"
        print("[OK] 필수 메서드 존재 확인")
        
        return True
    except Exception as e:
        print(f"❌ 오류: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_setup_database():
    """setup_database 스크립트 테스트."""
    print("\n" + "=" * 60)
    print("4. setup_database 스크립트 테스트")
    print("=" * 60)
    
    try:
        from scripts.setup_database import (
            DDL_CREATE_TABLES,
            DDL_CREATE_INDEXES,
            ensure_pgvector,
            insert_glossary_data,
            insert_paper_metadata,
            save_paper_id_mapping,
        )
        
        print("[OK] 모든 함수 임포트 성공")
        print(f"   DDL_CREATE_TABLES 길이: {len(DDL_CREATE_TABLES)} 문자")
        print(f"   DDL_CREATE_INDEXES 길이: {len(DDL_CREATE_INDEXES)} 문자")
        
        # DDL 내용 확인
        assert "CREATE TABLE IF NOT EXISTS papers" in DDL_CREATE_TABLES
        assert "CREATE TABLE IF NOT EXISTS glossary" in DDL_CREATE_TABLES
        assert "CREATE INDEX IF NOT EXISTS idx_papers_title" in DDL_CREATE_TABLES
        print("[OK] DDL 내용 확인 완료")
        
        return True
    except Exception as e:
        print(f"❌ 오류: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_data_files():
    """데이터 파일 존재 확인."""
    print("\n" + "=" * 60)
    print("5. 데이터 파일 확인")
    print("=" * 60)
    
    try:
        import json
        
        # 메타데이터 파일 확인
        metadata_path = ROOT / "data/raw/arxiv_papers_metadata.json"
        if metadata_path.exists():
            with open(metadata_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            print(f"[OK] 메타데이터 파일: {len(metadata)}개 논문")
        else:
            print(f"[WARN] 메타데이터 파일 없음: {metadata_path}")
        
        # PDF 디렉토리 확인
        pdf_dir = ROOT / "data/raw/pdfs"
        if pdf_dir.exists():
            pdf_files = list(pdf_dir.glob("*.pdf"))
            print(f"[OK] PDF 디렉토리: {len(pdf_files)}개 PDF 파일")
        else:
            print(f"[WARN] PDF 디렉토리 없음: {pdf_dir}")
        
        # 매핑 파일 확인
        mapping_path = ROOT / "data/processed/paper_id_mapping.json"
        if mapping_path.exists():
            with open(mapping_path, 'r', encoding='utf-8') as f:
                mapping = json.load(f)
            print(f"[OK] 매핑 파일: {len(mapping)}개 매핑")
        else:
            print(f"[WARN] 매핑 파일 없음 (setup_database.py 실행 후 생성됨)")
        
        return True
    except Exception as e:
        print(f"[ERROR] 오류: {e}")
        return False


def main():
    """전체 테스트 실행."""
    print("\n" + "=" * 60)
    print("데이터 파이프라인 구조 테스트")
    print("=" * 60 + "\n")
    
    results = []
    results.append(("ArxivPaperCollector", test_collector()))
    results.append(("PaperDocumentLoader", test_document_loader()))
    results.append(("PaperEmbeddingManager", test_embeddings()))
    results.append(("setup_database", test_setup_database()))
    results.append(("데이터 파일", test_data_files()))
    
    print("\n" + "=" * 60)
    print("테스트 결과 요약")
    print("=" * 60)
    
    for name, result in results:
        status = "[OK] 통과" if result else "[FAIL] 실패"
        print(f"{name}: {status}")
    
    all_passed = all(r for _, r in results)
    
    if all_passed:
        print("\n[OK] 모든 테스트 통과!")
    else:
        print("\n[WARN] 일부 테스트 실패 (환경 변수 또는 의존성 문제일 수 있음)")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())

