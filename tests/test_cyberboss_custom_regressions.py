import asyncio
import json

import pytest

from test_public_features import settings, ingest, synthetic_runner
from serein.compat.events import reference_blockers
from serein.core.store import Store
from serein.extensions import pipeline as p
from serein.model_runtime import non_thinking_options


def test_pipeline_event_stays_event_without_automatic_scene(settings):
    ingest(settings)
    result = asyncio.run(p.advance(settings.database, include_recent=True, runner=synthetic_runner))
    assert result['events'] == 1
    with Store(settings.database, read_only=True) as store:
        event_id = store.conn.execute(
            "SELECT id FROM documents WHERE kind='event' AND lifecycle='active'"
        ).fetchone()[0]
        assert store.conn.execute(
            "SELECT count(*) FROM documents WHERE kind='scene'"
        ).fetchone()[0] == 0
        assert store.surface_state(event_id)['can_surface'] is True


def test_authored_scene_still_blocks_event_supersession(settings):
    ingest(settings)
    assert asyncio.run(
        p.advance(settings.database, include_recent=True, runner=synthetic_runner)
    )['events'] == 1
    with Store(settings.database) as store:
        event_id = store.conn.execute(
            "SELECT id FROM documents WHERE kind='event' AND lifecycle='active'"
        ).fetchone()[0]
        source_rows = store.conn.execute(
            "SELECT source_id,metadata_json FROM evidence_bindings "
            "WHERE document_id=? AND active=1 ORDER BY id",
            (event_id,),
        ).fetchall()
        store.create(
            'manual-scene',
            'scene',
            'Manual memory',
            'Independently authored memory',
            metadata={'memory_value_source': 'authored_scene'},
        )
        for source in source_rows:
            store.bind(
                'manual-scene',
                source['source_id'],
                metadata=json.loads(source['metadata_json']),
            )
        assert 'active_scene_dependency' in reference_blockers(store.conn, event_id)


def test_api_prompt_compacts_frozen_writer_transport_without_changing_evidence():
    frozen_row = {
        'source_message_id': 7,
        'created_at': '2025-01-01T00:00:00Z',
        'speaker': '她',
        'text': 'exact source text',
        'saved_snowflake': False,
        'memory_event_source': False,
        'attachment_refs': [],
        'evidence_role': 'owned',
        'activity_role': 'primary_activity',
    }
    prompt = (
        'head\n\nRULES\n\n<event_reading_block_json>\n'
        + json.dumps([frozen_row], ensure_ascii=False)
        + '\n</event_reading_block_json>\n'
        + '<materialized_track_cards_json>\n[]\n</materialized_track_cards_json>\n'
        + '<track_context_events_json>\n[]\n</track_context_events_json>\n'
        + '<previous_events_json>\n[]\n</previous_events_json>\n'
    )
    compact = p.api_prompt_for_model(
        {'role': 'event_writer', 'rules': 'RULES', 'prompt': prompt}
    )
    assert '\n\nRULES\n\n' not in compact
    row = json.loads(
        compact.split('<event_reading_block_json>\n', 1)[1]
        .split('\n</event_reading_block_json>', 1)[0]
    )[0]
    assert row['text'] == 'exact source text'
    assert row['source_message_id'] == 7
    assert 'saved_snowflake' not in row
    assert 'memory_event_source' not in row
    assert 'attachment_refs' not in row


def test_parse_model_json_tolerates_fences_and_prose():
    assert p.parse_model_json('{"a":1}') == {'a': 1}
    assert p.parse_model_json('```json\n{"a":1}\n```') == {'a': 1}
    assert p.parse_model_json('```\n{"a":1}\n```') == {'a': 1}
    assert p.parse_model_json('Looking at this batch:\n{"a":1}\nmore notes') == {'a': 1}
    with pytest.raises(json.JSONDecodeError):
        p.parse_model_json('not json at all, no braces either')


def test_slow_stage_timeout_is_extended_without_changing_small_router_or_curator():
    assert p.api_timeout_seconds('track_router', 49999, 600) == 600
    assert p.api_timeout_seconds('track_router', 50000, 600) == 1200
    assert p.api_timeout_seconds('event_curator', 50000, 600) == 600
    assert p.api_timeout_seconds('event_writer', 1000, 600) == 1200
    assert p.api_timeout_seconds('event_writer', 50000, 1000) == 1800
    assert p.api_timeout_seconds('event_writer', 50000, 1800) == 1800


def test_sensenova_writer_disables_reasoning():
    assert non_thinking_options({
        'model': 'sensenova-6.8-flash-lite',
        'base_url': 'https://token.sensenova.cn/v1',
    }) == {'reasoning_effort': 'none'}
