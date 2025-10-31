from __future__ import annotations

"""PostgreSQL 초기 스키마 및 pgvector 확장 설정 스크립트."""

import os
from pathlib import Path

import psycopg2
from dotenv import load_dotenv
from typing import Dict

import json

ROOT = Path(__file__).resolve().parents[1]


DDL_CREATE_TABLES = """
CREATE TABLE IF NOT EXISTS papers (
    paper_id SERIAL PRIMARY KEY,
    arxiv_id VARCHAR(64),
    title TEXT NOT NULL,
    authors TEXT,
    publish_date DATE,
    source VARCHAR(32),
    url TEXT UNIQUE,
    category TEXT,
    citation_count INT,
    abstract TEXT,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS glossary (
    term_id SERIAL PRIMARY KEY,
    term TEXT UNIQUE,
    definition TEXT,
    easy_explanation TEXT,
    hard_explanation TEXT,
    category TEXT,
    difficulty_level INT,
    related_terms TEXT,
    examples TEXT,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_papers_title ON papers USING gin (to_tsvector('simple', title));
"""

DDL_ALTER_PAPERS_COLUMNS = """
ALTER TABLE papers ADD COLUMN IF NOT EXISTS category TEXT;
ALTER TABLE papers ADD COLUMN IF NOT EXISTS publish_date DATE;
ALTER TABLE papers ADD COLUMN IF NOT EXISTS citation_count INT;
"""

DDL_CREATE_INDEXES = """
CREATE INDEX IF NOT EXISTS idx_papers_category ON papers (category);
CREATE INDEX IF NOT EXISTS idx_papers_date ON papers (publish_date);
CREATE INDEX IF NOT EXISTS idx_glossary_term ON glossary (term);
"""


def ensure_pgvector(conn, cur) -> None:
    """pgvector 확장을 설치합니다(권한 필요)."""

    try:
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
    except Exception as e:  # noqa: BLE001
        # 권한 없을 수 있으므로 경고만 출력 후 트랜잭션 정리
        print(f"WARN: pgvector extension create failed: {e}")
        conn.rollback()


def insert_glossary_data(conn, cur):
    terms = [
        {"term": "Attention Mechanism", "definition": "특정 정보에 더 집중하는 신경망 기법.", "category": "Deep Learning", "difficulty_level": 1},
        {"term": "Fine-tuning", "definition": "사전학습된 모델을 특정 task에 맞게 추가 학습.", "category": "Transfer Learning", "difficulty_level": 1},
        {"term": "BLEU Score", "definition": "기계번역 등 텍스트 생성 성능 지표.", "category": "Metric", "difficulty_level": 1},
        # 필요한 만큼 추가...
    ]
    insert_sql = """
        INSERT INTO glossary (term, definition, category, difficulty_level, created_at)
        VALUES (%s, %s, %s, %s, NOW())
        ON CONFLICT (term) DO NOTHING;
    """
    for t in terms:
        try:
            cur.execute(insert_sql, (t["term"], t["definition"], t["category"], t["difficulty_level"]))
        except Exception as e:
            print("WARN: insert glossary term failed:", t["term"], e)
    conn.commit()
    print(f"Glossary seed data inserted ({len(terms)} terms).")


def insert_paper_metadata(conn, cur) -> Dict[str, int]:
    """JSON 메타데이터를 papers 테이블에 INSERT하고 arxiv_id → paper_id 매핑을 반환합니다.
    
    Args:
        conn: PostgreSQL 연결 객체
        cur: 커서 객체
        
    Returns:
        arxiv_id → paper_id 매핑 딕셔너리
    """
    meta_path = ROOT / "data/raw/arxiv_papers_metadata.json"
    if not meta_path.exists():
        print(f"SKIP: {meta_path} not found")
        return {}
    
    with open(meta_path, "r", encoding="utf-8") as f:
        papers = json.load(f)
    
    insert_sql = """
        INSERT INTO papers (arxiv_id, title, authors, publish_date, source, url, category, abstract)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (url) DO NOTHING
        RETURNING paper_id, arxiv_id;
    """
    
    mapping = {}
    inserted_count = 0
    
    for paper in papers:
        arxiv_id = paper.get("entry_id", "").split("/")[-1] if "entry_id" in paper else None
        title = paper.get("title", "")
        authors = ", ".join(paper.get("authors", [])) if isinstance(paper.get("authors"), list) else str(paper.get("authors", ""))
        publish_date = paper.get("published_date") or None
        source = "arxiv"
        url = paper.get("pdf_url") or paper.get("entry_id", "")
        category = paper.get("primary_category") or (paper.get("categories", [None])[0] if paper.get("categories") else None)
        abstract = paper.get("summary", "")
        
        try:
            cur.execute(
                insert_sql,
                (arxiv_id, title, authors, publish_date, source, url, category, abstract)
            )
            result = cur.fetchone()
            if result:
                paper_id, arxiv_id_returned = result
                if arxiv_id_returned:
                    mapping[arxiv_id_returned] = paper_id
                inserted_count += 1
        except Exception as e:  # noqa: BLE001
            print(f"WARN: insert paper failed: {title[:50]}... {e}")
            continue
    
    conn.commit()
    print(f"Inserted {inserted_count} papers into database.")
    return mapping


def save_paper_id_mapping(conn, mapping: Dict[str, int] = None):
    """arxiv_id → paper_id 매핑을 파일로 저장합니다.
    
    Args:
        conn: PostgreSQL 연결 객체
        mapping: 이미 생성된 매핑 딕셔너리 (None이면 DB에서 조회)
    """
    out_path = ROOT / "data/processed/paper_id_mapping.json"
    
    if mapping is None:
        # 기존 방식: DB에서 조회
        meta_path = ROOT / "data/raw/arxiv_papers_metadata.json"
        if not meta_path.exists():
            print(f"SKIP: {meta_path} not found")
            return
        with open(meta_path, "r", encoding="utf-8") as f:
            papers = json.load(f)
        arxiv_ids = [x.get("entry_id", "").split("/")[-1] for x in papers if "entry_id" in x]
        sql = "SELECT paper_id, arxiv_id FROM papers WHERE arxiv_id = ANY(%s)"
        mapping = {}
        with conn.cursor() as cur:
            cur.execute(sql, (arxiv_ids,))
            for pid, aid in cur.fetchall():
                mapping[aid] = pid
    
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(mapping, f, indent=2, ensure_ascii=False)
    print(f"Saved arxiv_id→paper_id mapping: {out_path} ({len(mapping)}개)")


def main() -> int:
    load_dotenv(ROOT / ".env")
    dsn = os.getenv("DATABASE_URL")
    if not dsn:
        raise RuntimeError("DATABASE_URL이 설정되지 않았습니다.")

    # psycopg2는 sqlalchemy dsn과 호환되도록 처리 필요
    # 예: postgresql+psycopg2:// → postgresql:// 로 치환
    dsn = dsn.replace("postgresql+psycopg2://", "postgresql://")

    with psycopg2.connect(dsn) as conn:
        with conn.cursor() as cur:
            ensure_pgvector(conn, cur)
            cur.execute(DDL_CREATE_TABLES)
            cur.execute(DDL_ALTER_PAPERS_COLUMNS)
            cur.execute(DDL_CREATE_INDEXES)
            insert_glossary_data(conn, cur)
            # JSON 메타데이터를 papers 테이블에 삽입
            mapping = insert_paper_metadata(conn, cur)
        conn.commit()
        # 매핑 파일 저장
        save_paper_id_mapping(conn, mapping if mapping else None)

    print("Database setup completed (tables, indexes, glossary seed, id mapping).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


