from __future__ import annotations

"""OpenAI Embeddings 생성 및 pgvector 저장.

LangChain PGVector를 사용하여 PostgreSQL(pgvector)에 문서를 적재합니다.
"""

import logging
import os
import time
from pathlib import Path
from typing import Dict, List, Optional

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_postgres.vectorstores import PGVector

logger = logging.getLogger(__name__)


class PaperEmbeddingManager:
    """논문 임베딩 및 Vector DB 저장 클래스."""

    def __init__(self, collection_name: str = "paper_chunks", skip_extension_check: bool = False) -> None:
        load_dotenv(Path(__file__).resolve().parents[2] / ".env")
        self.embeddings = OpenAIEmbeddings(
            model=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"),
            api_key=os.getenv("OPENAI_API_KEY"),
        )
        conn = os.getenv("DATABASE_URL")
        if not conn:
            raise RuntimeError("DATABASE_URL이 설정되지 않았습니다.")
        
        try:
            # pgvector extension 사전 확인 및 설치 시도
            try:
                import psycopg2
                test_conn = psycopg2.connect(conn)
                test_conn.autocommit = True
                test_cur = test_conn.cursor()
                test_cur.execute("SELECT 1 FROM pg_extension WHERE extname = 'vector';")
                extension_exists = test_cur.fetchone() is not None
                test_cur.close()
                test_conn.close()
                
                if not extension_exists:
                    print("\n[INFO] pgvector extension이 설치되지 않았습니다.")
                    print("       설치를 시도합니다...")
                    try:
                        install_conn = psycopg2.connect(conn)
                        install_conn.autocommit = True
                        install_cur = install_conn.cursor()
                        install_cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
                        install_cur.close()
                        install_conn.close()
                        print("[OK] pgvector extension 설치 완료")
                    except Exception as install_err:
                        print(f"[ERROR] 자동 설치 실패: {install_err}")
                        error_msg = (
                            "\n❌ pgvector extension이 설치되지 않았습니다.\n\n"
                            "다음 명령을 실행하여 수동 설치하세요:\n"
                            "  1. PostgreSQL 접속:\n"
                            "     psql -U postgres -d papers\n\n"
                            "  2. Extension 설치:\n"
                            "     CREATE EXTENSION IF NOT EXISTS vector;\n\n"
                            "  3. 확인:\n"
                            "     \\dx\n\n"
                            "자세한 내용은 docs/pgvector_quick_fix.md를 참조하세요.\n"
                        )
                        raise RuntimeError(error_msg) from install_err
            except ImportError:
                # psycopg2가 없으면 건너뜀 (langchain_postgres가 처리함)
                pass
            
            self.vectorstore = PGVector(
                collection_name=collection_name,
                connection=conn,
                embeddings=self.embeddings,
            )
        except Exception as e:
            if "vector" in str(e).lower() and "extension" in str(e).lower():
                error_msg = (
                    "\n❌ pgvector extension이 설치되지 않았습니다.\n\n"
                    "다음 명령을 실행하여 수동 설치하세요:\n"
                    "  1. PostgreSQL 접속:\n"
                    "     psql -U postgres -d papers\n\n"
                    "  2. Extension 설치:\n"
                    "     CREATE EXTENSION IF NOT EXISTS vector;\n\n"
                    "  3. 확인:\n"
                    "     \\dx\n\n"
                    "자세한 내용은 docs/pgvector_quick_fix.md를 참조하세요.\n"
                )
                raise RuntimeError(error_msg) from e
            raise

    def add_documents(self, documents: List[Document], batch_size: int = 50) -> int:
        """문서 리스트를 배치로 나누어 Vector DB에 저장합니다.

        Args:
            documents: 저장할 Document 리스트
            batch_size: 배치 크기 (기본값: 50)

        Returns:
            저장된 문서 수
        """
        total = 0
        num_batches = (len(documents) + batch_size - 1) // batch_size
        
        logger.info(f"Starting to add {len(documents)} documents in {num_batches} batches")
        
        for i in range(0, len(documents), batch_size):
            batch = documents[i : i + batch_size]
            batch_num = i // batch_size + 1
            
            try:
                self.vectorstore.add_documents(batch)
                total += len(batch)
                logger.info(f"Batch {batch_num}/{num_batches}: Added {len(batch)} documents (total: {total})")
                
                # Rate Limit 대응: 배치 간 대기
                if i + batch_size < len(documents):
                    time.sleep(0.1)  # 100ms 대기
                    
            except Exception as e:  # noqa: BLE001
                logger.error(f"Batch {batch_num}/{num_batches} failed: {e}")
                # Rate Limit 오류 시 더 긴 대기
                if "rate limit" in str(e).lower() or "429" in str(e):
                    logger.warning("Rate limit detected, waiting 5 seconds...")
                    time.sleep(5)
                continue
        
        logger.info(f"Completed: {total}/{len(documents)} documents added")
        return total

    def add_documents_with_paper_id(
        self, 
        documents: List[Document], 
        paper_id_mapping: Dict[str, int],
        batch_size: int = 50
    ) -> int:
        """문서 리스트를 paper_id 메타데이터와 함께 배치로 저장합니다.

        Args:
            documents: 저장할 Document 리스트 (metadata에 'arxiv_id' 또는 'entry_id' 포함)
            paper_id_mapping: arxiv_id → paper_id 매핑 딕셔너리
            batch_size: 배치 크기 (기본값: 50)

        Returns:
            저장된 문서 수
        """
        # 메타데이터에 paper_id 추가
        enriched_docs = []
        for doc in documents:
            # arxiv_id 추출
            arxiv_id = doc.metadata.get("arxiv_id")
            if not arxiv_id:
                # entry_id에서 추출
                entry_id = doc.metadata.get("entry_id", "")
                arxiv_id = entry_id.split("/")[-1] if entry_id else None
            
            # paper_id 매핑
            if arxiv_id and arxiv_id in paper_id_mapping:
                doc.metadata["paper_id"] = paper_id_mapping[arxiv_id]
                enriched_docs.append(doc)
            else:
                logger.warning(f"Paper ID not found for arxiv_id: {arxiv_id}, skipping document")
        
        logger.info(f"Enriched {len(enriched_docs)}/{len(documents)} documents with paper_id")
        
        # 배치 저장 실행
        return self.add_documents(enriched_docs, batch_size=batch_size)


