from __future__ import annotations

import logging
import os
from datetime import datetime
from typing import List

from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from ..domain.dto import PaperDTO

logger = logging.getLogger(__name__)


class ArxivClient:
    """arXiv 메타데이터 수집 클라이언트.

    네트워크 오류에 대해 지수 백오프로 재시도합니다.
    QPM 제한은 호출 상위에서 조절하거나, 호출 간 슬립으로 제어합니다.
    """

    def __init__(self, max_results_default: int = 20) -> None:
        self.max_results_default = max_results_default

    @retry(
        retry=retry_if_exception_type(Exception),
        wait=wait_exponential(multiplier=0.5, min=1, max=8),
        stop=stop_after_attempt(3),
        reraise=True,
    )
    def search(self, query: str, max_results: int | None = None) -> List[PaperDTO]:
        """쿼리로 arXiv를 검색하여 PaperDTO 리스트를 반환합니다.

        Args:
            query: 검색어
            max_results: 최대 결과 수
        """

        # 지연 임포트로 런타임 충돌 최소화
        import arxiv  # type: ignore

        limit = max_results or self.max_results_default
        results: List[PaperDTO] = []

        logger.info("arXiv 검색 시작: query=%s, max_results=%s", query, limit)
        search = arxiv.Search(
            query=query,
            max_results=limit,
            sort_by=arxiv.SortCriterion.Relevance,
        )

        for entry in search.results():  # type: ignore[attr-defined]
            arxiv_id = getattr(entry, "entry_id", "").rsplit("/", 1)[-1]
            title = getattr(entry, "title", "").strip()
            abstract = getattr(entry, "summary", "").strip()
            authors = [a.name for a in getattr(entry, "authors", [])]
            pdf_url = None
            try:
                pdf_url = next((l.href for l in entry.links if l.title == "pdf"), None)  # type: ignore[attr-defined]
            except Exception:
                pdf_url = None
            published_at = None
            try:
                published_at = getattr(entry, "published", None)
                if isinstance(published_at, datetime):
                    pass
                elif published_at:
                    published_at = datetime.fromisoformat(str(published_at))
            except Exception:
                published_at = None

            version = None
            if arxiv_id and "v" in arxiv_id:
                # 예: 2101.00001v2 → v2 추출
                version = "v" + arxiv_id.split("v")[-1]

            results.append(
                PaperDTO(
                    arxiv_id=arxiv_id,
                    title=title,
                    abstract=abstract,
                    authors=authors,
                    pdf_url=pdf_url,
                    published_at=published_at,
                    version=version,
                    source="arxiv",
                    lang="en",
                )
            )

        logger.info("arXiv 검색 완료: 결과 %d건", len(results))
        return results


