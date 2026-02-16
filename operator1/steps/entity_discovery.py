"""Steps A/B -- Linked entity discovery and resolution.

Uses Gemini to propose related entities (competitors, suppliers, customers,
etc.) and resolves each to an Eulerpool record via fuzzy matching with
scoring.  Tracks search budgets and checkpoints progress to disk.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Any

from operator1.clients.eulerpool import EulerportClient, EulerportAPIError
from operator1.clients.gemini import GeminiClient
from operator1.config_loader import get_global_config
from operator1.constants import CACHE_DIR, MATCH_SCORE_THRESHOLD, SECTOR_PEER_FALLBACK_COUNT

logger = logging.getLogger(__name__)

_PROGRESS_PATH = os.path.join(CACHE_DIR, "progress.json")

# Relationship groups expected from Gemini
RELATIONSHIP_GROUPS = (
    "competitors",
    "suppliers",
    "customers",
    "financial_institutions",
    "logistics",
    "regulators",
)


@dataclass
class LinkedEntity:
    """A resolved linked entity with match metadata."""

    isin: str
    ticker: str
    name: str
    country: str
    sector: str
    relationship_group: str
    match_score: int
    market_cap: float | None = None


@dataclass
class DiscoveryResult:
    """Container for the full discovery output."""

    linked: dict[str, list[LinkedEntity]] = field(default_factory=dict)
    search_calls_used: int = 0
    dropped_low_score: list[dict[str, Any]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def _score_match(
    query: str,
    candidate: dict[str, Any],
    target_country: str,
    target_sector: str,
) -> int:
    """Score a search candidate against the original query.

    Scoring weights (out of 100):
      - Ticker exact match:   40 pts
      - Name similarity:      30 pts (SequenceMatcher ratio * 30)
      - Country match:        15 pts
      - Sector match:         15 pts
    """
    score = 0

    # Ticker exact match
    cand_ticker = (candidate.get("ticker") or "").upper()
    if cand_ticker and cand_ticker == query.upper():
        score += 40

    # Name similarity
    cand_name = (candidate.get("name") or "").lower()
    ratio = SequenceMatcher(None, query.lower(), cand_name).ratio()
    score += int(ratio * 30)

    # Country match
    cand_country = (candidate.get("country") or "").upper()
    if cand_country and cand_country == target_country.upper():
        score += 15

    # Sector match
    cand_sector = (candidate.get("sector") or "").lower()
    if cand_sector and cand_sector == target_sector.lower():
        score += 15

    return score


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------

def _load_progress() -> dict[str, Any]:
    """Load discovery progress from disk."""
    if os.path.exists(_PROGRESS_PATH):
        try:
            with open(_PROGRESS_PATH, "r", encoding="utf-8") as fh:
                return json.load(fh)
        except (json.JSONDecodeError, OSError):
            pass
    return {"resolved": {}, "search_calls": 0}


def _save_progress(progress: dict[str, Any]) -> None:
    """Checkpoint discovery progress to disk."""
    os.makedirs(os.path.dirname(_PROGRESS_PATH), exist_ok=True)
    with open(_PROGRESS_PATH, "w", encoding="utf-8") as fh:
        json.dump(progress, fh, indent=2)


# ---------------------------------------------------------------------------
# Resolution
# ---------------------------------------------------------------------------

def _resolve_entity(
    query: str,
    group: str,
    eulerpool_client: EulerportClient,
    target_country: str,
    target_sector: str,
) -> LinkedEntity | None:
    """Search Eulerpool for *query* and return the best match above threshold.

    Returns ``None`` if no match scores >= MATCH_SCORE_THRESHOLD.
    """
    try:
        candidates = eulerpool_client.search(query)
    except EulerportAPIError as exc:
        logger.warning("Search failed for '%s': %s", query, exc)
        return None

    if not candidates:
        logger.debug("No search results for '%s'", query)
        return None

    # Score and rank
    scored = []
    for c in candidates:
        s = _score_match(query, c, target_country, target_sector)
        scored.append((s, c))
    scored.sort(key=lambda x: x[0], reverse=True)

    best_score, best = scored[0]
    if best_score < MATCH_SCORE_THRESHOLD:
        logger.debug(
            "Best match for '%s' scored %d (< %d) -- dropped",
            query, best_score, MATCH_SCORE_THRESHOLD,
        )
        return None

    return LinkedEntity(
        isin=best.get("isin", ""),
        ticker=best.get("ticker", ""),
        name=best.get("name", ""),
        country=best.get("country", ""),
        sector=best.get("sector", ""),
        relationship_group=group,
        match_score=best_score,
        market_cap=best.get("market_cap"),
    )


# ---------------------------------------------------------------------------
# Fallback: sector peers
# ---------------------------------------------------------------------------

def _fallback_sector_peers(
    target_isin: str,
    eulerpool_client: EulerportClient,
    count: int = SECTOR_PEER_FALLBACK_COUNT,
) -> list[LinkedEntity]:
    """Fallback when no competitors found: use Eulerpool peers sorted by market cap."""
    logger.info("Competitor fallback: fetching sector peers via Eulerpool ...")
    try:
        peer_isins = eulerpool_client.get_peers(target_isin)
    except EulerportAPIError as exc:
        logger.warning("Peer fallback failed: %s", exc)
        return []

    peers: list[LinkedEntity] = []
    for isin in peer_isins:
        if isin == target_isin:
            continue
        try:
            profile = eulerpool_client.get_profile(isin)
            peers.append(LinkedEntity(
                isin=isin,
                ticker=profile.get("ticker", ""),
                name=profile.get("name", ""),
                country=profile.get("country", ""),
                sector=profile.get("sector", ""),
                relationship_group="competitors",
                match_score=100,  # direct peer, full confidence
                market_cap=None,
            ))
        except EulerportAPIError:
            continue
        if len(peers) >= count:
            break

    logger.info("Peer fallback yielded %d competitors", len(peers))
    return peers


# ---------------------------------------------------------------------------
# Main discovery function
# ---------------------------------------------------------------------------

def discover_linked_entities(
    target_isin: str,
    target_profile: dict[str, Any],
    eulerpool_client: EulerportClient,
    gemini_client: GeminiClient | None = None,
    force_rebuild: bool | None = None,
) -> DiscoveryResult:
    """Discover and resolve linked entities for the target company.

    Parameters
    ----------
    target_isin:
        ISIN of the verified target.
    target_profile:
        Full profile dict from Eulerpool (used for Gemini hints).
    eulerpool_client:
        Initialised Eulerpool client.
    gemini_client:
        Optional Gemini client.  If ``None``, skips LLM proposals and
        goes straight to peer fallback for competitors.
    force_rebuild:
        Override FORCE_REBUILD config.

    Returns
    -------
    DiscoveryResult
        Contains linked entities grouped by relationship type.
    """
    cfg = get_global_config()
    if force_rebuild is None:
        force_rebuild = cfg.get("FORCE_REBUILD", False)

    budget_per_group: int = cfg.get("search_budget_per_group", 10)
    budget_global: int = cfg.get("search_budget_global", 50)

    target_country = (target_profile.get("country") or "").upper()
    target_sector = (target_profile.get("sector") or "").lower()

    # Load checkpoint (resume if partially complete)
    progress = _load_progress() if not force_rebuild else {"resolved": {}, "search_calls": 0}
    global_calls = progress.get("search_calls", 0)

    result = DiscoveryResult(search_calls_used=global_calls)

    # Restore previously resolved entities
    for group, entities_raw in progress.get("resolved", {}).items():
        result.linked[group] = [
            LinkedEntity(**e) for e in entities_raw
        ]

    # ------------------------------------------------------------------
    # 1. Get Gemini proposals
    # ------------------------------------------------------------------
    proposals: dict[str, list[str]] = {}
    if gemini_client is not None:
        sector_hints = f"{target_sector}, country={target_country}"
        proposals = gemini_client.propose_linked_entities(
            target_profile, sector_hints=sector_hints,
        )
        logger.info(
            "Gemini proposed entities for %d groups: %s",
            len(proposals),
            {g: len(v) for g, v in proposals.items()},
        )
    else:
        logger.info("No Gemini client -- skipping LLM entity proposals")

    # ------------------------------------------------------------------
    # 2. Resolve each proposal via Eulerpool search
    # ------------------------------------------------------------------
    for group in RELATIONSHIP_GROUPS:
        if group in result.linked:
            logger.debug("Group '%s' already resolved from checkpoint", group)
            continue

        names = proposals.get(group, [])
        resolved: list[LinkedEntity] = []
        group_calls = 0

        for name in names:
            if group_calls >= budget_per_group:
                logger.info("Budget exhausted for group '%s'", group)
                break
            if global_calls >= budget_global:
                logger.info("Global search budget exhausted")
                break

            entity = _resolve_entity(
                name, group, eulerpool_client, target_country, target_sector,
            )
            group_calls += 1
            global_calls += 1

            if entity is not None:
                # Deduplicate by ISIN
                if entity.isin and entity.isin != target_isin:
                    existing_isins = {e.isin for e in resolved}
                    if entity.isin not in existing_isins:
                        resolved.append(entity)
                        logger.info(
                            "  [%s] Resolved: %s (%s) score=%d",
                            group, entity.name, entity.isin, entity.match_score,
                        )
            else:
                result.dropped_low_score.append({
                    "query": name,
                    "group": group,
                    "reason": "below_threshold",
                })

        result.linked[group] = resolved

        # Checkpoint after each group
        progress["resolved"][group] = [
            {
                "isin": e.isin, "ticker": e.ticker, "name": e.name,
                "country": e.country, "sector": e.sector,
                "relationship_group": e.relationship_group,
                "match_score": e.match_score, "market_cap": e.market_cap,
            }
            for e in resolved
        ]
        progress["search_calls"] = global_calls
        _save_progress(progress)

    result.search_calls_used = global_calls

    # ------------------------------------------------------------------
    # 3. Fallback: if no competitors found, use sector peers
    # ------------------------------------------------------------------
    competitors = result.linked.get("competitors", [])
    if not competitors:
        logger.warning("No competitors resolved -- triggering peer fallback")
        peers = _fallback_sector_peers(target_isin, eulerpool_client)
        result.linked["competitors"] = peers
        progress["resolved"]["competitors"] = [
            {
                "isin": e.isin, "ticker": e.ticker, "name": e.name,
                "country": e.country, "sector": e.sector,
                "relationship_group": e.relationship_group,
                "match_score": e.match_score, "market_cap": e.market_cap,
            }
            for e in peers
        ]
        _save_progress(progress)

    # Summary
    total = sum(len(v) for v in result.linked.values())
    logger.info(
        "Discovery complete: %d entities across %d groups, %d search calls, %d dropped",
        total,
        sum(1 for v in result.linked.values() if v),
        result.search_calls_used,
        len(result.dropped_low_score),
    )

    return result


def get_all_linked_isins(result: DiscoveryResult) -> list[str]:
    """Extract a flat list of unique ISINs from a discovery result."""
    seen: set[str] = set()
    isins: list[str] = []
    for entities in result.linked.values():
        for e in entities:
            if e.isin and e.isin not in seen:
                seen.add(e.isin)
                isins.append(e.isin)
    return isins
