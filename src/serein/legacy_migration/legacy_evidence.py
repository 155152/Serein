"""Preserve Haven-Ombre evidence layers without promoting them to Scenes."""

from contextlib import closing
from pathlib import Path
import sqlite3

from ..core.store import Conflict, Store, digest, encode
from .originals import COLUMNS


def import_source_archive(database, plan):
    """Store legacy source-record buckets and source-only graph edges off the recall surface."""
    records = plan.get("source_records") or []
    edges = plan.get("source_edges") or []
    if not records and not edges:
        return {"source_records": 0, "source_edges": 0}

    origin = "ombre-legacy:" + plan["fingerprint"]
    with Store(database) as store, store.transaction(immediate=True):
        for record in records:
            metadata = {
                "source_system": "ombre_legacy_source_record",
                "legacy_id": record["old_id"],
                "legacy_path": record["path"],
                "legacy_title": record["title"],
                "import_source_hash": record["source_hash"],
                "legacy_metadata": record.get("metadata") or {},
            }
            source_key = encode(["ombre_legacy_source_record", plan["fingerprint"], record["old_id"]])
            source_id = store.add_source(source_key, record["body"], metadata=metadata)
            saved = store.conn.execute(
                "SELECT source_key,content,content_sha256,metadata_json FROM sources WHERE id=?",
                (source_id,),
            ).fetchone()
            expected = (source_key, record["body"], digest(record["body"]), encode(metadata))
            if not saved or tuple(saved) != expected:
                raise Conflict("Legacy source record differs from the imported source: " + record["old_id"])

        for edge in edges:
            record_id = digest(encode(edge))
            values = (
                origin,
                "ombre_source_edge",
                record_id,
                "legacy_source_record_endpoint",
                encode(edge),
            )
            store.conn.execute("INSERT OR IGNORE INTO detached_import_records VALUES (?,?,?,?,?)", values)
            saved = store.conn.execute(
                "SELECT reason,metadata_json FROM detached_import_records "
                "WHERE origin=? AND record_type=? AND record_id=?",
                values[:3],
            ).fetchone()
            if not saved or tuple(saved) != values[3:]:
                raise Conflict("Legacy source edge differs from the imported archive")
    return {"source_records": len(records), "source_edges": len(edges)}


def _wanted_raw_ids(items):
    result = []
    seen = set()
    for item in items:
        for value in item.get("source_raw_event_ids") or []:
            key = int(value)
            if key not in seen:
                seen.add(key)
                result.append(key)
    return result


def _legacy_rows(path, wanted):
    if not wanted:
        return {}
    if not path:
        raise ValueError("旧 Scene 含 source_raw_event_ids，但旧原文库未找到")
    path = Path(path).resolve()
    rows = {}
    with closing(sqlite3.connect(path.as_uri() + "?mode=ro", uri=True)) as conn:
        conn.row_factory = sqlite3.Row
        for start in range(0, len(wanted), 800):
            chunk = wanted[start:start + 800]
            marks = ",".join("?" for _ in chunk)
            for row in conn.execute(
                "SELECT id," + ",".join(COLUMNS) + " FROM raw_events WHERE id IN (" + marks + ")",
                chunk,
            ):
                rows[int(row["id"])] = dict(row)
    missing = [key for key in wanted if key not in rows]
    if missing:
        raise ValueError("旧 Scene provenance 引用了不存在的 raw event: " + str(missing[0]))
    return rows


def resolve_raw_evidence(database, plan):
    """Map old integer raw IDs to the exact imported Serein raw-message identities."""
    wanted = _wanted_raw_ids(plan.get("items") or [])
    if not wanted:
        return {}
    legacy = _legacy_rows((plan.get("originals") or {}).get("path"), wanted)
    resolved = {}
    compare = [key for key in COLUMNS if key not in ("ingested_at", "metadata_json", "client")]
    with Store(database, read_only=True) as store:
        for old_id in wanted:
            row = legacy[old_id]
            matches = store.conn.execute(
                "SELECT * FROM raw_events WHERE source=? "
                "AND (event_hash=? OR (source_event_id!='' AND source_event_id=?))",
                (row["source"], row["event_hash"], row["source_event_id"]),
            ).fetchall()
            exact = [candidate for candidate in matches if all(candidate[key] == row[key] for key in compare)]
            if len(exact) != 1:
                raise ValueError("无法把旧 raw event 确定映射到 Serein 原话: " + str(old_id))
            current = exact[0]
            message_id = current["source_event_id"] or str(current["id"])
            metadata = {
                "source_system": current["source"],
                "session_id": current["session_id"],
                "thread_id": "",
                "conversation_id": current["conversation_id"],
                "message_id": message_id,
                "raw_id": current["id"],
                "role": current["role"],
                "created_at": current["created_at"],
                "content_sha256": digest(current["text"]),
                "evidence_kind": "primary",
                "binding_method": "legacy_ombre_raw_event_id",
                "legacy_raw_event_id": old_id,
            }
            resolved[old_id] = {
                "source_key": encode([current["source"], current["session_id"], message_id]),
                "content": current["text"],
                "metadata": metadata,
            }
    return resolved


def bind_scene_raw_evidence(store, scene_id, item, resolved, fingerprint):
    count = 0
    for old_id in item.get("source_raw_event_ids") or []:
        source = resolved[int(old_id)]
        source_id = store.add_source(source["source_key"], source["content"], metadata=source["metadata"])
        binding_id = "legacy_ombre_" + digest(
            encode([fingerprint, item["old_id"], int(old_id)])
        )
        store.bind(
            scene_id,
            source_id,
            binding_id=binding_id,
            metadata=source["metadata"],
            active=True,
            record_action=False,
        )
        count += 1
    return count
