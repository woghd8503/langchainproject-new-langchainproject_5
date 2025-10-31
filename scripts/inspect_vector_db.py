"""FAISS 벡터 데이터베이스 내용 확인 스크립트

사용법:
    python scripts/inspect_vector_db.py
    
벡터 DB에 저장된 문서, 메타데이터, 인덱스 정보를 확인합니다.
"""

import os
import sys
from pathlib import Path

# Windows 콘솔 인코딩 설정
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# 프로젝트 루트 경로 추가
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# 환경 변수 설정 (PyTorch 경고 방지)
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message=".*torch.classes.*")

# LangChain 임포트
from langchain_core.documents import Document
try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# 경로 설정
FAISS_DIR = ROOT / "data" / "vectordb" / "papers_faiss"


def inspect_vector_db():
    """FAISS 벡터 DB 내용을 확인합니다."""
    print("=" * 80)
    print("FAISS 벡터 데이터베이스 열람")
    print("=" * 80)
    
    # 인덱스 존재 확인
    if not FAISS_DIR.exists():
        print(f"❌ 벡터 DB가 존재하지 않습니다: {FAISS_DIR}")
        print("   먼저 앱에서 논문을 검색하고 색인을 실행하세요.")
        return
    
    index_files = list(FAISS_DIR.glob("*.faiss")) + list(FAISS_DIR.glob("*.pkl"))
    if not index_files:
        print(f"❌ 인덱스 파일이 없습니다: {FAISS_DIR}")
        return
    
    print(f"📁 벡터 DB 경로: {FAISS_DIR}")
    print(f"📊 인덱스 파일 수: {len(index_files)}")
    print()
    
    # 임베딩 모델 로드
    print("🔄 임베딩 모델 로딩 중...")
    try:
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        print("✅ 임베딩 모델 로드 완료")
    except Exception as e:
        print(f"❌ 임베딩 모델 로드 실패: {e}")
        return
    
    # FAISS 벡터 스토어 로드
    print("🔄 FAISS 벡터 스토어 로드 중...")
    try:
        vs = FAISS.load_local(
            FAISS_DIR.as_posix(), 
            embeddings, 
            allow_dangerous_deserialization=True
        )
        print("✅ 벡터 스토어 로드 완료")
    except Exception as e:
        print(f"❌ 벡터 스토어 로드 실패: {e}")
        return
    
    print()
    print("=" * 80)
    print("📊 인덱스 통계")
    print("=" * 80)
    
    # 인덱스 기본 정보
    try:
        total_docs = vs.index.ntotal
        print(f"총 저장된 문서 수: {total_docs}")
        
        if hasattr(vs.index, 'd'):
            print(f"벡터 차원: {vs.index.d}")
        
        print(f"인덱스 타입: {type(vs.index).__name__}")
    except Exception as e:
        print(f"⚠️ 인덱스 정보 확인 실패: {e}")
        total_docs = 0
    
    print()
    
    # 저장된 문서 확인
    print("=" * 80)
    print("📄 저장된 문서 목록")
    print("=" * 80)
    
    if total_docs == 0:
        print("⚠️ 저장된 문서가 없습니다.")
        return
    
    # 모든 문서 조회 시도 (FAISS는 직접 문서 접근이 제한적)
    # 대신 샘플 검색으로 문서 확인
    print("\n🔍 샘플 검색으로 저장된 문서 확인...\n")
    
    # 일반적인 키워드로 검색
    sample_queries = ["paper", "research", "arxiv", "title", "summary"]
    
    all_retrieved_docs = []
    seen_contents = set()
    
    for query in sample_queries:
        try:
            docs = vs.similarity_search(query, k=5)
            for doc in docs:
                # 중복 제거
                content_hash = hash(doc.page_content[:100])
                if content_hash not in seen_contents:
                    seen_contents.add(content_hash)
                    all_retrieved_docs.append(doc)
        except Exception as e:
            print(f"⚠️ 검색 중 오류: {e}")
    
    # 실제 발견된 고유 문서 수
    unique_docs = len(all_retrieved_docs)
    print(f"발견된 고유 문서 수: {unique_docs}")
    print()
    
    # 상위 10개 문서 상세 출력
    print("=" * 80)
    print(f"📋 상위 {min(10, unique_docs)}개 문서 상세")
    print("=" * 80)
    
    for idx, doc in enumerate(all_retrieved_docs[:10], 1):
        print(f"\n[문서 {idx}]")
        print("-" * 80)
        
        # 메타데이터 출력
        if doc.metadata:
            print("📌 메타데이터:")
            for key, value in doc.metadata.items():
                print(f"   {key}: {value}")
            print()
        
        # 본문 일부 출력 (최대 300자)
        content = doc.page_content
        if len(content) > 300:
            print(f"📝 본문 (처음 300자):\n{content[:300]}...")
        else:
            print(f"📝 본문:\n{content}")
        
        print()
    
    # 검색 테스트
    print()
    print("=" * 80)
    print("🔎 검색 기능 테스트")
    print("=" * 80)
    
    test_queries = [
        "machine learning",
        "neural network", 
        "transformer",
    ]
    
    for query in test_queries:
        print(f"\n검색어: '{query}'")
        try:
            results = vs.similarity_search_with_score(query, k=3)
            print(f"  → {len(results)}개 결과 발견")
            
            for i, (doc, score) in enumerate(results[:3], 1):
                print(f"\n  [{i}] 유사도 점수: {score:.4f}")
                if doc.metadata:
                    title = doc.metadata.get("Title") or doc.metadata.get("title") or "제목 없음"
                    print(f"      제목: {title[:80]}...")
                print(f"      본문: {doc.page_content[:150]}...")
        except Exception as e:
            print(f"  ❌ 검색 실패: {e}")
    
    print()
    print("=" * 80)
    print("✅ 열람 완료")
    print("=" * 80)


if __name__ == "__main__":
    inspect_vector_db()
