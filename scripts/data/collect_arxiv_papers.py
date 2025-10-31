"""arXiv 논문 수집 스크립트.

ArxivClient를 사용하여 논문을 검색하고 PDF를 다운로드하며 메타데이터를 저장합니다.
ExperimentManager를 사용하여 실험을 추적합니다.

사용법:
    python scripts/data/collect_arxiv_papers.py
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

# 프로젝트 루트 경로 추가
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.papers.domain.dto import PaperDTO
from src.papers.infra.arxiv_client import ArxivClient
from src.utils.experiment_manager import ExperimentManager


class ArxivPaperCollector:
    """arXiv 논문 수집 클래스.

    ArxivClient를 래핑하여 PDF 다운로드 및 메타데이터 JSON 저장 기능을 제공합니다.
    """

    def __init__(
        self,
        save_dir: str | Path = "data/raw/pdfs",
        metadata_dir: str | Path = "data/raw/json",
        exp_manager: Optional[ExperimentManager] = None,
    ) -> None:
        """ArxivPaperCollector 초기화.

        Args:
            save_dir: PDF 파일 저장 디렉토리
            metadata_dir: 메타데이터 JSON 저장 디렉토리
            exp_manager: ExperimentManager 인스턴스 (선택적)
        """
        self.save_dir = Path(save_dir)
        self.metadata_dir = Path(metadata_dir)
        self.exp_manager = exp_manager
        
        # 디렉토리 생성
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_dir.mkdir(parents=True, exist_ok=True)
        
        # ArxivClient 초기화
        self.client = ArxivClient(max_results_default=50)

    def collect_papers(self, query: str, max_results: int = 50) -> List[Dict]:
        """arXiv에서 논문을 수집하고 PDF를 다운로드합니다.

        Args:
            query: 검색 쿼리
            max_results: 최대 수집 논문 수

        Returns:
            논문 메타데이터 리스트 (JSON 직렬화 가능한 형태)
        """
        if self.exp_manager:
            self.exp_manager.logger.write(f"논문 수집 시작: query='{query}', max_results={max_results}")

        # ArxivClient를 사용하여 검색
        paper_dtos = self.client.search(query=query, max_results=max_results)
        
        papers_data = []
        downloaded_count = 0
        failed_count = 0

        # 지연 임포트
        import arxiv  # type: ignore

        for dto in paper_dtos:
            try:
                # 메타데이터 수집
                paper_info = {
                    "title": dto.title,
                    "authors": dto.authors,
                    "published_date": dto.published_at.strftime("%Y-%m-%d") if dto.published_at else None,
                    "summary": dto.abstract,
                    "pdf_url": dto.pdf_url or f"https://arxiv.org/pdf/{dto.arxiv_id}.pdf",
                    "entry_id": f"https://arxiv.org/abs/{dto.arxiv_id}",
                    "categories": [],  # ArxivClient에서 카테고리 정보가 없음
                    "primary_category": None,
                }

                # PDF 다운로드
                pdf_filename = self.save_dir / f"{dto.arxiv_id}.pdf"
                
                if pdf_filename.exists():
                    if self.exp_manager:
                        self.exp_manager.logger.write(f"PDF 이미 존재: {dto.arxiv_id}.pdf")
                    papers_data.append(paper_info)
                    continue

                try:
                # PDF URL로 직접 다운로드
                import urllib.request
                
                pdf_url = dto.pdf_url or f"https://arxiv.org/pdf/{dto.arxiv_id}.pdf"
                
                # 재시도 로직
                max_retries = 3
                for attempt in range(max_retries):
                    try:
                        urllib.request.urlretrieve(pdf_url, pdf_filename)
                        break
                    except Exception as retry_e:  # noqa: BLE001
                        if attempt == max_retries - 1:
                            raise retry_e
                        time.sleep(1)
                    
                    downloaded_count += 1
                    if self.exp_manager:
                        self.exp_manager.logger.write(f"PDF 다운로드 완료: {dto.arxiv_id}.pdf")
                    
                    # 다운로드 후 짧은 대기 (API Rate Limit 방지)
                    time.sleep(0.5)
                    
                except Exception as e:  # noqa: BLE001
                    failed_count += 1
                    if self.exp_manager:
                        self.exp_manager.logger.write(f"PDF 다운로드 실패: {dto.arxiv_id}.pdf - {e}")
                    # 다운로드 실패해도 메타데이터는 저장
                    continue

                papers_data.append(paper_info)

            except Exception as e:  # noqa: BLE001
                failed_count += 1
                if self.exp_manager:
                    self.exp_manager.logger.write(f"논문 처리 실패: {dto.arxiv_id} - {e}")
                continue

        if self.exp_manager:
            self.exp_manager.logger.write(
                f"논문 수집 완료: 총 {len(paper_dtos)}건, "
                f"성공 {downloaded_count}건, 실패 {failed_count}건"
            )

        return papers_data

    def collect_by_keywords(
        self, keywords: List[str], per_keyword: int = 15
    ) -> List[Dict]:
        """여러 키워드로 논문을 수집하고 중복을 제거합니다.

        Args:
            keywords: 검색 키워드 리스트
            per_keyword: 키워드당 수집할 논문 수

        Returns:
            중복 제거된 논문 메타데이터 리스트
        """
        if self.exp_manager:
            self.exp_manager.logger.write(
                f"키워드별 수집 시작: {len(keywords)}개 키워드, "
                f"키워드당 {per_keyword}편"
            )

        all_papers = []
        
        for idx, keyword in enumerate(keywords, 1):
            if self.exp_manager:
                self.exp_manager.logger.write(
                    f"[{idx}/{len(keywords)}] 키워드 수집 중: '{keyword}'"
                )
            
            papers = self.collect_papers(query=keyword, max_results=per_keyword)
            all_papers.extend(papers)
            
            # 키워드 간 대기 (API Rate Limit 방지)
            if idx < len(keywords):
                time.sleep(2)

        # 중복 제거
        unique_papers = self.remove_duplicates(all_papers)

        if self.exp_manager:
            self.exp_manager.logger.write(
                f"키워드별 수집 완료: 총 {len(all_papers)}건, "
                f"중복 제거 후 {len(unique_papers)}건"
            )

        return unique_papers

    @staticmethod
    def remove_duplicates(papers: List[Dict]) -> List[Dict]:
        """제목 기준으로 중복 논문을 제거합니다.

        Args:
            papers: 논문 메타데이터 리스트

        Returns:
            중복 제거된 논문 리스트
        """
        seen_titles = set()
        unique_papers = []

        for paper in papers:
            # 제목을 소문자로 정규화하여 중복 확인
            title_lower = paper.get("title", "").lower().strip()
            
            if title_lower and title_lower not in seen_titles:
                seen_titles.add(title_lower)
                unique_papers.append(paper)

        return unique_papers

    def save_metadata(self, papers: List[Dict], filename: str = "arxiv_papers_metadata.json") -> Path:
        """메타데이터를 JSON 파일로 저장합니다.

        Args:
            papers: 논문 메타데이터 리스트
            filename: 저장할 파일명

        Returns:
            저장된 파일 경로
        """
        metadata_path = self.metadata_dir / filename
        
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(papers, f, indent=2, ensure_ascii=False)

        if self.exp_manager:
            self.exp_manager.logger.write(
                f"메타데이터 저장 완료: {metadata_path} ({len(papers)}건)"
            )

        return metadata_path


def main() -> int:
    """메인 실행 함수."""
    
    # AI/ML 키워드 리스트
    keywords = [
        "transformer attention",
        "BERT GPT",
        "large language model",
        "retrieval augmented generation",
        "neural machine translation",
        "question answering",
        "AI agent",
    ]

    # ExperimentManager로 실험 추적
    with ExperimentManager() as exp:
        exp.logger.write("arXiv 논문 수집 파이프라인 시작")
        exp.update_metadata(
            user_query="arXiv 논문 수집",
            difficulty="easy",
            tool_used="arxiv_collector"
        )

        # Collector 초기화
        collector = ArxivPaperCollector(
            save_dir=ROOT / "data/raw/pdfs",
            metadata_dir=ROOT / "data/raw/json",
            exp_manager=exp,
        )

        # 키워드별 수집
        papers = collector.collect_by_keywords(keywords, per_keyword=15)

        # 메타데이터 저장
        metadata_path = collector.save_metadata(papers)
        
        exp.logger.write(f"총 {len(papers)}개 고유 논문 수집 완료")
        exp.update_metadata(
            success=True,
            response_time_ms=None,
        )

        print(f"\n✅ 논문 수집 완료: {len(papers)}개")
        print(f"📄 메타데이터 저장: {metadata_path}")
        print(f"📁 PDF 파일 저장: {collector.save_dir}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

