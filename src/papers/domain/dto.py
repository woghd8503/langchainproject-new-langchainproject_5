from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional


@dataclass
class PaperDTO:
    """논문 메타데이터 전송 객체.

    Google-style docstring을 따릅니다.

    Attributes:
        arxiv_id: arXiv 고유 식별자 (예: 2101.00001)
        title: 논문 제목
        abstract: 초록 텍스트
        authors: 저자 목록 (표기명)
        pdf_url: PDF 다운로드 URL
        published_at: 최초 공개 일시 (UTC)
        version: 버전 문자열 (예: v1)
        source: 수집 출처 식별자 (예: "arxiv")
        lang: 언어 코드 (예: "en")
    """

    arxiv_id: str
    title: str
    abstract: str
    authors: List[str]
    pdf_url: Optional[str]
    published_at: Optional[datetime]
    version: Optional[str] = None
    source: str = "arxiv"
    lang: Optional[str] = None


@dataclass
class ResultStats:
    """수집/저장 결과 요약.

    Attributes:
        inserted: 신규 삽입된 레코드 수
        updated: 기존 레코드 업데이트 수
        skipped: 중복 등으로 스킵된 수
        failed: 실패한 항목 수
        detail: 부가 정보 메시지 목록
    """

    inserted: int = 0
    updated: int = 0
    skipped: int = 0
    failed: int = 0
    detail: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        """dict 형태로 변환합니다."""
        return {
            "inserted": self.inserted,
            "updated": self.updated,
            "skipped": self.skipped,
            "failed": self.failed,
            "detail": list(self.detail),
        }


