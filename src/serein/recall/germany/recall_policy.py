"""Read-only Germany policy slice; provenance: PROVENANCE.md."""
from __future__ import annotations

import re
from functools import lru_cache
from typing import Any
from .memory_relevance import MemoryRelevanceOptions
from .memory_relevance import memory_relevance_options_from_config
from .memory_relevance import content_terms_for_query
from .identity import identity_names
from .query_terms import RECALL_SYSTEM_META_TERMS, WEAK_AUTOMATION_TOPIC_TERMS
WEAK_RECALL_TOPIC_TERMS = frozenset({*RECALL_SYSTEM_META_TERMS, '进度', '偏好', '情况', '状态', '事情', '东西', '内容', '相关', '记忆', '回忆', '总结', '记录', '查询', '搜索', '最近', '之前', '过去', '现在', '当前', '安排', '计划', '问题', '目标', '好', '喜欢', '好喜欢', '舒服', '好舒服', 'anything', 'current', 'find', 'memory', 'memories', 'recent', 'related', 'search', 'something', 'status', 'thing', 'things', 'topic'})
WEAK_RECALL_TOPIC_TERMS = WEAK_RECALL_TOPIC_TERMS | WEAK_AUTOMATION_TOPIC_TERMS
GENERIC_RECALL_CONTEXT_TERMS = frozenset({'ai_name', 'assistant', 'display_name', 'human_name', 'user', 'user_alias', 'user_aliases', 'user_display_name', 'user_name', 'username', '对方', '用户'})
RELATIONSHIP_BACKGROUND_QUERY_FILLERS = frozenset({'我', '你', '他', '她', '它', '我们', '你们', '他们', '她们', '哥哥', '老公', '老婆', '宝宝', '宝贝', '亲爱的', '自己', '可以', '能不能', '可不可以', '那个', '这个', '作为', '怎么样', '话说'})

class RecallPolicy:

    def __init__(self, options: MemoryRelevanceOptions | None=None, *, semantic_threshold: float=0.72, rerank_threshold: float=0.65, ai_reaction_names: list[str] | tuple[str, ...] | None=None, relationship_names: list[str] | tuple[str, ...] | None=None) -> None:
        self.options = options or memory_relevance_options_from_config()
        self.semantic_threshold = _safe_float(semantic_threshold, 0.72)
        self.rerank_threshold = _safe_float(rerank_threshold, 0.65)
        self.ai_reaction_names = self._normalize_reaction_names(ai_reaction_names if ai_reaction_names is not None else [identity_names().get('ai_name')])
        default_relationship_names = identity_names().get('relationship_terms') or []
        self.relationship_names = self._normalize_recall_context_terms(relationship_names if relationship_names is not None else default_relationship_names)
        self.relationship_background_fillers = self._normalize_recall_context_terms([*RELATIONSHIP_BACKGROUND_QUERY_FILLERS, *self.relationship_names])
        self.recall_context_terms = self._normalize_recall_context_terms([*self.options.context_terms, *GENERIC_RECALL_CONTEXT_TERMS])

    @staticmethod
    def _normalize_reaction_names(values: list[str] | tuple[str, ...] | None) -> set[str]:
        names: set[str] = set()
        for value in values or []:
            compact = re.sub('\\s+', '', str(value or '').lower())
            key = re.sub('[^0-9a-z\\u4e00-\\u9fff]+', '', compact)
            if key:
                names.add(key)
        return names

    @staticmethod
    def _normalize_recall_context_terms(values) -> set[str]:
        terms: set[str] = set()
        for value in values or []:
            key = re.sub('\\s+', ' ', str(value or '').strip().lower())
            if key:
                terms.add(key)
            compact = re.sub('[^0-9a-z\\u4e00-\\u9fff]+', '', key)
            if compact:
                terms.add(compact)
        return terms

    def _is_recall_context_term(self, term: str) -> bool:
        key = re.sub('\\s+', ' ', str(term or '').strip().lower())
        compact = re.sub('[^0-9a-z\\u4e00-\\u9fff]+', '', key)
        return key in self.recall_context_terms or compact in self.recall_context_terms

    def _is_relationship_background_filler(self, term: str) -> bool:
        key = re.sub('\\s+', ' ', str(term or '').strip().lower())
        compact = re.sub('[^0-9a-z\\u4e00-\\u9fff]+', '', key)
        if key in self.relationship_background_fillers or compact in self.relationship_background_fillers:
            return True
        return bool(compact and len(compact) >= 2 and any(
            compact in filler for filler in self.relationship_background_fillers if len(filler) > len(compact)))

    def specific_query_terms(self, query: str) -> list[str]:
        raw = str(query or '')
        return list(self._specific_query_terms_cached(raw))

    @lru_cache(maxsize=512)
    def _specific_query_terms_cached(self, raw: str) -> tuple[str, ...]:
        terms = list(content_terms_for_query(raw, self.options))
        terms.extend(re.findall('\\d+(?:\\.\\d+)+', raw))
        terms.extend(re.findall('[A-Za-z]+[A-Za-z0-9_.:-]*\\d[A-Za-z0-9_.:-]*', raw))
        kept = []
        seen = set()
        for term in terms:
            cleaned = str(term or '').strip()
            if not cleaned:
                continue
            key = cleaned.lower()
            if key in seen:
                continue
            if key in WEAK_RECALL_TOPIC_TERMS:
                continue
            if key in RECALL_SYSTEM_META_TERMS:
                continue
            if self._is_recall_context_term(cleaned):
                continue
            if self._is_relationship_background_filler(cleaned):
                continue
            if re.fullmatch('[a-z0-9_.:-]+', key) and len(key) < 3 and (not re.fullmatch('\\d+(?:\\.\\d+)+', key)):
                continue
            if re.fullmatch('[\\u4e00-\\u9fff]+', cleaned) and len(cleaned) < 2:
                continue
            if any((_term_subsumes(existing.lower(), key) for existing in kept)):
                continue
            kept = [existing for existing in kept if not _term_subsumes(key, existing.lower())]
            seen = {existing.lower() for existing in kept}
            seen.add(key)
            kept.append(cleaned)
        return tuple(kept)

def _term_subsumes(container: str, contained: str) -> bool:
    if container == contained:
        return True
    if not container or not contained:
        return False
    if not re.search('\\d', contained):
        return False
    return contained in container

def _maybe_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None

def _safe_float(value: Any, default: float) -> float:
    number = _maybe_float(value)
    return default if number is None else number
