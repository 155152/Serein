"""Bounded diary discovery for MCP; exact reads retain their existing adapters."""
from datetime import datetime, timezone
import re

from ..compat.germany.diary_store import _normalize_date
from ..core.store import Store


def validate_read(diary_id, query, limit, offset):
    if diary_id is not None and diary_id <= 0:
        raise ValueError('diary_id must be a positive integer')
    if not 1 <= limit <= 20 or offset < 0:
        raise ValueError('limit must be 1..20 and offset must be non-negative')
    if diary_id is not None and (query.strip() or offset):
        raise ValueError('Use query/offset to find diaries, then diary_id alone to read one')


def _preview(body, query):
    text = re.sub(r'\s+', ' ', body).strip()
    match = re.search(re.escape(query), text, re.IGNORECASE) if query else None
    start = max(0, match.start() - 45) if match else 0
    prefix = '…' if start else ''
    available = 150 - len(prefix)
    suffix = '…' if len(text) - start > available else ''
    return prefix + text[start:start + available - len(suffix)] + suffix


def diary_directory(database, *, query='', date='', limit=5, offset=0, include_archived=False):
    """Search visible prose only, before pagination; never search sealed bodies."""
    validate_read(None, query, limit, offset)
    query = query.strip()
    day = _normalize_date(date) if date.strip() else ''
    clock = datetime.now(timezone.utc)

    def unlocked(value):
        if not value:
            return True
        try:
            stamp = datetime.fromisoformat(value.replace('Z', '+00:00'))
            return stamp.tzinfo is not None and stamp <= clock
        except (ValueError, TypeError):
            return False

    with Store(database, read_only=True) as store:
        legacy = store.conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='diaries'").fetchone()
        table, day_column, body_column = ('diaries', 'date', 'content') if legacy else ('diary_entries', 'day', 'body_md')
        store.conn.create_function('diary_unlocked', 1, unlocked)
        visibility = "visibility IN ('active','archived')" if include_archived else "visibility='active'"
        clauses = [visibility, "COALESCE(deleted_at,'')=''", 'diary_unlocked(unlock_at)']
        values = []
        if day:
            clauses.append(f'{day_column}=?')
            values.append(day)
        if query:
            clauses.append(f"(instr(lower(COALESCE(title,'')),lower(?))>0 OR instr(lower({body_column}),lower(?))>0)")
            values.extend((query, query))
        order = f'{day_column} DESC,id DESC'
        if query:
            order = "(instr(lower(COALESCE(title,'')),lower(?))>0) DESC," + order
            values.append(query)
        values.extend((limit + 1, offset))
        rows = store.conn.execute(
            f'SELECT id,title,{day_column} AS day,{body_column} AS body FROM {table} WHERE '
            + ' AND '.join(clauses) + f' ORDER BY {order} LIMIT ? OFFSET ?', values).fetchall()
    has_more = len(rows) > limit
    lines = ['Diary directory', f'count: {min(len(rows), limit)}', f'offset: {offset}']
    for row in rows[:limit]:
        title = re.sub(r'\s+', ' ', row['title'] or '').strip()
        lines.extend(('', f"diary_id: {row['id']}", f"date: {row['day']}",
                      f'title: {title[:150]}', f"excerpt: {_preview(row['body'] or '', query)}"))
    lines.append(f'has_more: {str(has_more).lower()}')
    if has_more:
        lines.extend((f'next_offset: {offset + limit}', 'Keep the same query, date and limit when paging.'))
    lines.append('Use read_diary(diary_id=ID) to read one complete diary and its comments.')
    return '\n'.join(lines)
