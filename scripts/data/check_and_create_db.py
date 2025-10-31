"""papers 데이터베이스가 존재하는지 확인하고 없으면 생성합니다."""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv
import psycopg2

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")

# postgres 데이터베이스에 연결 (기본 데이터베이스)
dsn = os.getenv("DATABASE_URL")
if not dsn:
    print("[ERROR] DATABASE_URL이 설정되지 않았습니다.")
    sys.exit(1)

# postgres 데이터베이스에 연결하기 위해 URL 수정
dsn_postgres = dsn.replace("/papers", "/postgres")

try:
    conn = psycopg2.connect(dsn_postgres)
    conn.autocommit = True
    cur = conn.cursor()
    
    # papers 데이터베이스 존재 확인
    cur.execute("SELECT 1 FROM pg_database WHERE datname = 'papers'")
    exists = cur.fetchone() is not None
    
    if not exists:
        print("[INFO] papers 데이터베이스가 없습니다. 생성 중...")
        cur.execute('CREATE DATABASE papers')
        print("[OK] papers 데이터베이스 생성 완료")
    else:
        print("[OK] papers 데이터베이스가 이미 존재합니다.")
    
    cur.close()
    conn.close()
    
except psycopg2.Error as e:
    print(f"[ERROR] 오류 발생: {e}")
    sys.exit(1)

