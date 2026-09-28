import asyncio
import json

import pytest

from test_public_features import settings, ingest, output_for, synthetic_runner
from serein.compat.events import reference_blockers
from serein.core.store import Store, promoted_scene_id
from serein.extensions import pipeline as p
from serein.extensions import pipeline_latest as latest
from serein.model_runtime import non_thinking_options


def test_auto_scene_does_not_block_event_extension_and_successor_retires_it(settings):
    ingest(settings)

    async def scene_runner(role, request):
        output = output_for(role, request)
        if role == 'event_writer':
            output['scene_worthy'] = True
        return output

    first = asyncio.run(p.advance(settings.database, include_recent=True, runner=scene_runner))
    assert first['events'] == 1
    with Store(settings.database, read_only=True) as store:
        first_event_id = store.conn.execute(
            "SELECT id FROM documents WHERE kind='event' AND lifecycle='active'"
        ).fetchone()[0]
        first_scene_id = promoted_scene_id(first_event_id)
        assert store.read(first_scene_id)['lifecycle'] == 'active'
        assert 'active_scene_dependency' not in reference_blockers(store.conn, first_event_id)

    ingest(settings, 2)
    second = asyncio.run(p.advance(settings.database, include_recent=True, runner=scene_runner))
    assert second['events'] == 1

    with Store(settings.database, read_only=True) as store:
        active_event_id = store.conn.execute(
            "SELECT id FROM documents WHERE kind='event' AND lifecycle='active'"
        ).fetchone()[0]
        assert active_event_id != first_event_id
        assert store.read(first_event_id)['lifecycle'] == 'superseded'
        assert store.read(first_scene_id)['lifecycle'] == 'archived'
        successor_scene = store.read(promoted_scene_id(active_event_id))
        assert successor_scene is not None
        assert successor_scene['lifecycle'] == 'active'
        assert successor_scene['metadata']['promoted_from_event']['id'] == active_event_id


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


def test_writer_result_caps_auxiliary_kept_details_without_changing_body():
    result = output_for('event_writer', {'messages': [{'id': 1, 'content': 'A book was returned'}]})
    result['event_draft'] = '书还了。'
    result['kept_details'] = [f'anchor-{i}' for i in range(8)]
    body = result['event_draft']
    result.pop('claim_groups', None)
    result.pop('sentence_evidence', None)
    latest.normalize_event_writer_result(result)
    assert result['kept_details'] == [f'anchor-{i}' for i in range(6)]
    assert result['event_draft'] == body
    assert latest.validate_event_writer_result(result) == []


def test_writer_repair_prompt_makes_detail_limit_explicit():
    failed = {'kept_details': [str(i) for i in range(8)], 'discarded_details': []}
    prompt = latest.build_event_writer_repair_prompt(
        'base', failed, ['kept_details 必须有 1–6 项：8']
    )
    assert 'kept_details must contain at most 6 items' in prompt
    assert 'Keep only the 6 most essential anchors' in prompt


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


def test_writer_transcript_omits_default_empty_transport_fields_but_keeps_signals():
    rows = latest.writer_transcript_payload([
        {
            'id': 1,
            'role': 'user',
            'content': 'plain',
            'created_at': '2025-01-01T00:00:00Z',
            'metadata': {},
        },
        {
            'id': 2,
            'role': 'assistant',
            'content': 'with attachment',
            'created_at': '2025-01-01T00:01:00Z',
            'metadata': {
                'memory_event_source': True,
                'attachments': [
                    {
                        'id': 'img-1',
                        'kind': 'image',
                        'name': 'photo.png',
                        'mime_type': 'image/png',
                    }
                ],
            },
        },
    ])
    assert 'saved_snowflake' not in rows[0]
    assert 'memory_event_source' not in rows[0]
    assert 'attachment_refs' not in rows[0]
    assert rows[1]['memory_event_source'] is True
    assert rows[1]['attachment_refs'][0]['attachment_id'] == 'img-1'


def test_parse_model_json_tolerates_fences_and_prose():
    assert p.parse_model_json('{"a":1}') == {'a': 1}
    assert p.parse_model_json('```json\n{"a":1}\n```') == {'a': 1}
    assert p.parse_model_json('```\n{"a":1}\n```') == {'a': 1}
    assert p.parse_model_json('Looking at this batch:\n{"a":1}\nmore notes') == {'a': 1}
    with pytest.raises(json.JSONDecodeError):
        p.parse_model_json('not json at all, no braces either')


def test_track_router_truncates_overlong_subject_and_throughline():
    long_subject = '题' * 200
    long_throughline = '尾' * 700
    output = {
        'message_assignments': [
            {
                'source_message_id': 1,
                'primary_track_ref': 'new:1',
                'context_track_refs': [],
                'routing_role': 'primary_activity',
            },
            {
                'source_message_id': 2,
                'primary_track_ref': 'new:1',
                'context_track_refs': [],
                'routing_role': 'primary_activity',
            },
        ],
        'track_updates': [
            {
                'track_ref': 'new:1',
                'subject': long_subject,
                'throughline': long_throughline,
                'event_policy': 'default',
                'status': 'active',
            }
        ],
    }
    _, cards, _ = p.normalize_event_track_message_output(
        output, [{'id': 1}, {'id': 2}], [], session_id=1, next_track_ordinal=1
    )
    card = cards[0]
    assert len(card['subject']) == 160 and card['subject'].endswith('…')
    assert len(card['throughline']) == 598 and card['throughline'].endswith('…')


def test_router_fills_only_missing_existing_track_updates():
    output = {
        'message_assignments': [
            {
                'source_message_id': 1,
                'primary_track_ref': 'session_test_track_0001',
                'context_track_refs': ['new:1'],
                'routing_role': 'bridge',
            }
        ],
        'track_updates': [
            {
                'track_ref': 'new:1',
                'subject': 'new',
                'throughline': 'new arc',
                'event_policy': 'default',
                'status': 'active',
            }
        ],
    }
    tracks = [
        {
            'track_id': 'session_test_track_0001',
            'subject': 'old',
            'throughline': 'old arc',
            'event_policy': 'rolling_engineering',
            'status': 'parked',
        }
    ]
    p.fill_missing_existing_track_updates(output, tracks)
    by_ref = {item['track_ref']: item for item in output['track_updates']}
    assert set(by_ref) == {'session_test_track_0001', 'new:1'}
    assert by_ref['session_test_track_0001'] == {
        'track_ref': 'session_test_track_0001',
        'subject': 'old',
        'throughline': 'old arc',
        'event_policy': 'rolling_engineering',
        'status': 'parked',
    }


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
