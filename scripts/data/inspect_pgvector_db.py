"""PGVector 벡터 데이터베이스 내용 확인 스크립트.

사용법:
    python scripts/data/inspect_pgvector_db.py
    
PGVector에 저장된 문서, 메타데이터, 인덱스 정보를 확인합니다.
"""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_postgres.vectorstores import PGVector

# 프로젝트 루트 경로 추가
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

# Windows 콘솔 인코딩 설정
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# 환경 변수 로드
load_dotenv(ROOT / ".env")


def inspect_pgvector_db():
    """PGVector 벡터 DB 내용을 확인합니다."""
    print("=" * 80)
    print("PGVector 벡터 데이터베이스 열람")
    print("=" * 80)
    
    # 환경 변수 확인
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        print("❌ DATABASE_URL 환경 변수가 설정되지 않았습니다.")
        print("   .env 파일에 DATABASE_URL을 설정하세요.")
        return
    
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        print("❌ OPENAI_API_KEY 환경 변수가 설정되지 않았습니다.")
        print("   .env 파일에 OPENAI_API_KEY를 설정하세요.")
        return
    
    print(f"📁 데이터베이스: {database_url.split('@')[-1] if '@' in database_url else database_url}")
    print()
    
    # 임베딩 모델 초기화
    print("🔄 임베딩 모델 로딩 중...")
    try:
        embeddings = OpenAIEmbeddings(
            model=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"),
            api_key=openai_api_key,
        )
        print("✅ 임베딩 모델 로드 완료")
    except Exception as e:
        print(f"❌ 임베딩 모델 로드 실패: {e}")
        return
    
    # PGVector 벡터 스토어 초기화
    print("🔄 PGVector 벡터 스토어 연결 중...")
    try:
        vectorstore = PGVector(
            collection_name="paper_chunks",
            connection=database_url,
            embeddings=embeddings,
        )
        print("✅ 벡터 스토어 연결 완료")
    except Exception as e:
        print(f"❌ 벡터 스토어 연결 실패: {e}")
        return
    
    print()
    print("=" * 80)
    print("📊 컬렉션 정보")
    print("=" * 80)
    
    # 컬렉션 정보 확인 (Langchain PGVector 내부 테이블 조회)
    try:
        import psycopg2
        
        # 연결 문자열 파싱
        conn = psycopg2.connect(database_url)
        cur = conn.cursor()
        
        # 컬렉션 개수 확인
        cur.execute("""
            SELECT COUNT(*) 
            FROM langchain_pg_collection 
            WHERE collection_name = 'paper_chunks'
        """)
        collection_count = cur.fetchone()[0]
        print(f"컬렉션 존재 여부: {'✅ 있음' if collection_count > 0 else '❌ 없음'}")
        
        # 벡터 수 확인
        cur.execute("""
            SELECT COUNT(*) 
            FROM langchain_pg_embedding 
            WHERE collection_id IN (
                SELECT uuid FROM langchain_pg_collection 
                WHERE collection_name = 'paper_chunks'
            )
        """)
        total_vectors = cur.fetchone()[0]
        print(f"총 저장된 벡터 수: {total_vectors}")
        
        cur.close()
        conn.close()
        
        if total_vectors == 0:
            print("\n⚠️ 저장된 벡터가 없습니다.")
            print("   먼저 load_embeddings.py를 실행하여 벡터를 저장하세요.")
            return
        
    except Exception as e:
        print(f"⚠️ 데이터베이스 조회 중 오류: {e}")
        total_vectors = 0
    
    print()
    
    # 샘플 검색으로 문서 확인
    print("=" * 80)
    print("📄 저장된 문서 샘플 확인")
    print("=" * 80)
    
    sample_queries = ["paper", "research", "arxiv", "transformer", "neural"]
    
    all_retrieved_docs = []
    seen_contents = set()
    
    for query in sample_queries:
        try:
            docs = vectorstore.similarity_search(query, k=5)
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
    
    # 상위 5개 문서 상세 출력
    print("=" * 80)
    print(f"📋 상위 {min(5, unique_docs)}개 문서 상세")
    print("=" * 80)
    
    for idx, doc in enumerate(all_retrieved_docs[:5], 1):
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
            results = vectorstore.similarity_search_with_score(query, k=3)
            print(f"  → {len(results)}개 결과 발견")
            
            for i, (doc, score) in enumerate(results[:3], 1):
                print(f"\n  [{i}] 유사도 점수: {score:.4f}")
                if doc.metadata:
                    paper_id = doc.metadata.get("paper_id") or "없음"
                    arxiv_id = doc.metadata.get("arxiv_id") or "없음"
                    print(f"      paper_id: {paper_id}, arxiv_id: {arxiv_id}")
                print(f"      본문: {doc.page_content[:150]}...")
        except Exception as e:
            print(f"  ❌ 검색 실패: {e}")
    
    print()
    print("=" * 80)
    print("✅ 열람 완료")
    print("=" * 80)


if __name__ == "__main__":
    inspect_pgvector_db()

