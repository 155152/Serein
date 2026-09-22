"""Bounded extra Scout candidates; retrieval signals never establish Arc membership."""

import asyncio
import re
import sqlite3

from ..adapters.embedding import EmbeddingClient
from ..core.reader import Reader
from ..configured_models import effective_settings
from ..recall.index import Search
from ..tagging_entities import current_entities, entity_key
from .germany.narrative_revision_scout import _material_key, _material_text, _term_allowed


EXTRA_PER_ROUTE = 4
MAX_CORRIDOR_CANDIDATES = 32  # Existing lexical maximum 24 + two extra routes of four.
VERSION = 'scout-candidates-v1'


def add_current_entities(settings, inventory):
    """Read validated names only; suggested aliases do not establish identity."""
    result = []
    with Reader(settings.database) as reader:
        for item in inventory:
            names = []
            if item['source_type'] in {'event', 'scene'}:
                obj = reader.read(item['source_id'], kind=item['source_type'], with_evidence=False)
                if obj['readable'] and obj['document']['metadata'].get('tagged_entities'):
                    names = sorted({entity['name'] for entity in current_entities(reader.store, obj['document'])
                                    if _term_allowed(entity['name'])})
            result.append({**item, 'entity_names': names})
    return result


def index_revision(settings):
    """An index becoming available/refreshed must invalidate an unchanged scan."""
    try:
        settings = effective_settings(settings)
        stat = settings.index.stat()
        wal = settings.index.with_name(settings.index.name + '-wal')
        wal_stat = wal.stat() if wal.exists() else None
        return [stat.st_mtime_ns, stat.st_size,
                wal_stat.st_mtime_ns if wal_stat else None, wal_stat.st_size if wal_stat else None]
    except (AttributeError, OSError, ValueError, sqlite3.Error):
        return None


def _eligible(item, seed_key):
    return _material_key(item) != seed_key and not item.get('bound_narrative_ids')


def entity_candidates(inventory, seeds):
    # A name grounded in any current extraction can also find an untagged diary.
    vocabulary = {entity_key(name): name for item in inventory for name in item.get('entity_names', [])}
    names_by_item = {}
    texts = {_material_key(item): entity_key(_material_text(item)) for item in inventory}
    matcher = re.compile('|'.join(
        (r'(?<![A-Za-z0-9_])' + re.escape(key) + r'(?![A-Za-z0-9_])') if key.isascii()
        else re.escape(key) for key in sorted(vocabulary, key=lambda key: (-len(key), key)))) if vocabulary else None
    for item in inventory:
        key = _material_key(item)
        names = {entity_key(name) for name in item.get('entity_names', [])}
        if matcher:
            names.update(match.group() for match in matcher.finditer(texts[key]))
        names_by_item[key] = names
    result = {}
    for seed in seeds:
        seed_key = _material_key(seed)
        ranked = []
        for item in inventory:
            shared = names_by_item[seed_key] & names_by_item[_material_key(item)]
            if shared and _eligible(item, seed_key):
                ranked.append((item, sorted(vocabulary[name] for name in shared)))
        ranked.sort(key=lambda row: (-len(row[1]), _material_key(row[0])))
        result[seed_key] = [{**item, 'matched_entities': names} for item, names in ranked[:EXTRA_PER_ROUTE]]
    return result


def _semantic_query(settings, seed, eligible):
    # Query embedding uses the existing query instruction/profile, not document vectors.
    query = '\n'.join(filter(None, [str(seed.get('title') or ''),
                                    str(seed.get('search_text') or seed.get('summary') or '')]))[:4000]
    if not query.strip():
        return []
    embedding = EmbeddingClient(settings.database, settings.index, **settings.embedding).query(query)
    with Search(settings.database, settings.index) as search:
        hits = search.search(query, mode='lookup', limit=EXTRA_PER_ROUTE, query_embedding=embedding,
                             min_cosine=0.3, use_passages=True,
                             candidate_ids={item['source_id'] for item in eligible.values()
                                            if item['source_type'] in {'event', 'scene'}})['items']
    result = []
    for hit in hits:
        key = f"{hit['kind']}:{hit['id']}"
        if key in eligible:
            result.append({**eligible[key], 'semantic_score': hit['score']})
        if len(result) == EXTRA_PER_ROUTE:
            break
    return result


async def semantic_candidates(settings, inventory, seeds):
    try:
        settings = effective_settings(settings)
    except (ValueError, OSError, sqlite3.Error):
        return {}, {'status': 'failed', 'failed_queries': len(seeds)}
    if not settings.index or not settings.embedding:
        return {}, {'status': 'unconfigured', 'failed_queries': 0}
    semaphore = asyncio.Semaphore(3)

    async def one(seed):
        key = _material_key(seed)
        eligible = {_material_key(item): item for item in inventory if _eligible(item, key)
                    and item['source_type'] in {'event', 'scene'}}
        if not eligible:
            return key, [], False
        async with semaphore:
            try:
                return key, await asyncio.to_thread(_semantic_query, settings, seed, eligible), False
            except (ValueError, OSError, sqlite3.Error, ImportError):
                # No provider error text, private material, or credential in scan receipts.
                return key, [], True

    rows = await asyncio.gather(*(one(seed) for seed in seeds))
    failures = sum(failed for _, _, failed in rows)
    status = ('failed' if failures == len(rows) else 'partial') if failures else 'ok'
    return {key: hits for key, hits, _ in rows}, {'status': status, 'failed_queries': failures}


def merge_candidates(corridors, entities, semantic):
    """Keep all existing lexical candidates; deduplicate while retaining route provenance."""
    merged = []
    for corridor in corridors:
        seed_key = _material_key(corridor['seed'])
        rows = {}
        for route, candidates in [('keyword', corridor['candidates']),
                                  ('entity', entities.get(seed_key, [])[:EXTRA_PER_ROUTE]),
                                  ('semantic', semantic.get(seed_key, [])[:EXTRA_PER_ROUTE])]:
            for rank, candidate in enumerate(candidates, 1):
                if not _eligible(candidate, seed_key):
                    continue
                key = _material_key(candidate)
                row = rows.setdefault(key, {**candidate, 'candidate_sources': [], 'candidate_ranks': {}})
                row['candidate_sources'].append(route)
                row['candidate_ranks'][route] = rank
                for field in ('matched_keywords', 'matched_entities', 'semantic_score'):
                    if field in candidate:
                        row[field] = candidate[field]
        merged.append({**corridor, 'candidates': list(rows.values())[:MAX_CORRIDOR_CANDIDATES]})
    return merged


async def supplement_candidates(settings, inventory, corridors):
    # The existing Scout prompt consumes at most 24 seeds.
    seeds = [corridor['seed'] for corridor in corridors[:24]]
    entities = entity_candidates(inventory, seeds)
    semantic, status = await semantic_candidates(settings, inventory, seeds)
    merged = merge_candidates(corridors, entities, semantic)
    counts = {route: sum(route in row['candidate_sources'] for corridor in merged[:24]
                         for row in corridor['candidates']) for route in ('keyword', 'entity', 'semantic')}
    return merged, {'version': VERSION, 'semantic': status, 'candidate_pairs_by_route': counts}
