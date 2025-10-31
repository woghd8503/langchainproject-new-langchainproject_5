"""pgvector extension 설치 스크립트.

PostgreSQL에 직접 접속하여 pgvector extension을 설치합니다.
"""

import os
import sys
from pathlib import Path

import psycopg2
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")

def install_pgvector():
    """PostgreSQL에 pgvector extension을 설치합니다."""
    
    dsn = os.getenv("DATABASE_URL")
    if not dsn:
        print("[ERROR] DATABASE_URL이 설정되지 않았습니다.")
        sys.exit(1)
    
    try:
        conn = psycopg2.connect(dsn)
        conn.autocommit = True
        cur = conn.cursor()
        
        # pgvector extension 설치 시도
        print("pgvector extension 설치 시도 중...")
        try:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            print("[OK] pgvector extension 설치 완료")
        except psycopg2.errors.FeatureNotSupported as e:
            print(f"[ERROR] pgvector extension을 설치할 수 없습니다.")
            print(f"       {e}")
            print()
            print("해결 방법:")
            print("1. PostgreSQL에 관리자 권한으로 접속:")
            print("   psql -U postgres -d papers")
            print("2. 다음 명령 실행:")
            print("   CREATE EXTENSION IF NOT EXISTS vector;")
            print()
            print("자세한 내용은 docs/pgvector_install_windows.md를 참조하세요.")
            cur.close()
            conn.close()
            sys.exit(1)
        except Exception as e:
            print(f"[ERROR] 오류 발생: {e}")
            cur.close()
            conn.close()
            sys.exit(1)
        
        # 설치 확인
        cur.execute("SELECT 1 FROM pg_extension WHERE extname = 'vector';")
        if cur.fetchone():
            print("[OK] pgvector extension 확인 완료")
        else:
            print("[WARN] pgvector extension이 설치되었지만 확인할 수 없습니다.")
        
        cur.close()
        conn.close()
        
    except psycopg2.Error as e:
        print(f"[ERROR] PostgreSQL 연결 오류: {e}")
        sys.exit(1)


if __name__ == "__main__":
    install_pgvector()

