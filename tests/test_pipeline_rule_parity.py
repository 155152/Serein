import asyncio
import copy

import pytest

from test_public_features import settings, ingest, output_for, raw_archive
from serein.extensions import pipeline as p
from serein.extensions.pipeline_admission import count_rounds
from serein.extensions.pipeline_audit import curator_receipt_errors
from serein.extensions.pipeline_images import MissingOriginalImage


@pytest.mark.parametrize('status', [404, 410, 503])
def test_download_only_classifies_gone_http_status_as_missing(monkeypatch, status):
    from urllib.parse import urlsplit
    from serein.extensions import pipeline_images as images
    monkeypatch.setattr(images, '_public_image_target', lambda url: (
        urlsplit(url), 80, [(None, None, None, None, ('93.184.216.34', 80))]))
    monkeypatch.setattr(images.socket, 'create_connection', lambda *args, **kwargs: object())
    class Connection:
        def __init__(self, *args, **kwargs): pass
        def request(self, *args): pass
        def getresponse(self): return type('Response', (), {'status': status})()
        def close(self): pass
    monkeypatch.setattr(images.http.client, 'HTTPConnection', Connection)
    expected = MissingOriginalImage if status in (404, 410) else ValueError
    with pytest.raises(expected) as caught:
        images.image_bytes('http://example.com/image.png')
    assert isinstance(caught.value, MissingOriginalImage) == (status in (404, 410))


def test_missing_image_does_not_drop_usable_sibling_and_is_not_refetched(settings, monkeypatch):
    from serein.extensions import pipeline_images as images
    uri = 'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+aX1cAAAAASUVORK5CYII='
    calls = []
    def gone(url):
        calls.append(url)
        raise MissingOriginalImage('synthetic HTTP 410')
    monkeypatch.setattr(images, '_remote_image_bytes', gone)
    raw_archive(settings).ingest([
        {'source_event_id': 'u', 'session_id': 'one', 'role': 'user', 'text': 'Compare these covers',
         'created_at': '2025-01-01T00:00:00Z', 'metadata': {'attachments': [
             {'kind': 'image', 'url': 'https://example.com/gone.png'}, {'kind': 'image', 'url': uri}]}},
        {'source_event_id': 'a', 'session_id': 'one', 'role': 'assistant', 'text': 'The blue cover is useful',
         'created_at': '2025-01-01T00:01:00Z'}], source='test')
    async def runner(role, request):
        if request['execution']['task'] == 'image_transcription':
            assert request['images'][0]['position'] == 2
            return {'image_transcriptions': [{'input_image': 1, 'text': 'Blue cover', 'unreadable': False}]}
        return output_for(role, request)
    result = asyncio.run(p.advance(settings.database, include_recent=True, runner=runner))
    assert result['events'] == 1 and len(result['missing_images']) == 1
    assert len(calls) == 1


@pytest.mark.parametrize('url', [None, 'missing-original', 'https://example.com/gone.png'])
def test_missing_original_skips_attachment_and_keeps_text(settings, monkeypatch, url):
    def gone(_):
        raise MissingOriginalImage('synthetic HTTP 404')
    monkeypatch.setattr('serein.extensions.pipeline_images.image_bytes', gone)
    raw_archive(settings).ingest([
        {'source_event_id': 'u', 'session_id': 'one', 'role': 'user', 'text': 'We chose the blue cover.',
         'created_at': '2025-01-01T00:00:00Z', 'metadata': {'attachments': [{'kind': 'image', 'url': url}]}},
        {'source_event_id': 'a', 'session_id': 'one', 'role': 'assistant', 'text': 'I will bind it.',
         'created_at': '2025-01-01T00:01:00Z'}], source='test')
    seen = []
    async def runner(role, request):
        seen.append(role)
        assert request['execution']['task'] != 'image_transcription'
        if role != 'track_router':
            assert request['images'] == []
            assert '原图缺失' in request['prompt']
        return output_for(role, request)
    result = asyncio.run(p.advance(settings.database, include_recent=True, runner=runner))
    assert result['events'] == 1 and result['processed_originals'] == 2
    assert result['missing_images'][0]['reason'] == 'original_missing'
    assert seen == list(p.ROLES)
    assert asyncio.run(p.advance(settings.database, include_recent=True, runner=runner))['status'] == 'current'


def test_temporary_download_failure_is_not_a_missing_original(settings, monkeypatch):
    raw_archive(settings).ingest([
        {'source_event_id': 'u', 'session_id': 'one', 'role': 'user', 'text': 'Read this picture',
         'created_at': '2025-01-01T00:00:00Z',
         'metadata': {'attachments': [{'kind': 'image', 'url': 'https://example.com/slow.png'}]}},
        {'source_event_id': 'a', 'session_id': 'one', 'role': 'assistant', 'text': 'Let us read it',
         'created_at': '2025-01-01T00:01:00Z'}], source='test')
    def timeout(_):
        raise TimeoutError('synthetic download timeout')
    monkeypatch.setattr('serein.extensions.pipeline_images.image_bytes', timeout)
    async def runner(role, request):
        return output_for(role, request)
    with pytest.raises(TimeoutError):
        asyncio.run(p.advance(settings.database, include_recent=True, runner=runner))


def test_reply_to_other_user_does_not_count_as_an_owned_round():
    messages = [{'id': 1, 'session_id': 'one', 'role': 'user'},
                {'id': 2, 'session_id': 'one', 'role': 'assistant',
                 'metadata': {'reply_to_user_message_id': 99}}]
    assert count_rounds([1, 2], messages) == 0
    messages[1]['metadata']['reply_to_user_message_id'] = 1
    assert count_rounds([1, 2], messages) == 1
    messages[1]['metadata'] = {}
    assert count_rounds([1, 2], messages) == 1


def test_boundary_quote_recovery_remains_in_exclusive_cited_original():
    component = {'context_messages': [{'id': 1, 'content': 'We picked the blue shelf.'},
                                      {'id': 2, 'content': 'The lamp needs a new switch.'}],
                 'memberships': [], 'parked_context_source_ids': []}
    plan = {'events': [{'primary_track_id': 'home', 'source_bindings': [{'source_message_id': 1}]},
                       {'primary_track_id': 'home', 'source_bindings': [{'source_message_id': 2}]}],
            'skip_source_message_ids': [], 'defer_source_message_ids': []}
    review = {'events': [{'event_index': 0, 'reason': 'Shelf choice'}, {'event_index': 1, 'reason': 'Lamp repair'}],
              'boundaries': [{'left_event_index': 0, 'right_event_index': 1, 'reason': 'Different activities',
                              'evidence': [{'source_message_id': 1, 'quote': 'We picked blue shelf.'},
                                           {'source_message_id': 2, 'quote': 'new switch'}]}],
              'dispositions': []}
    assert curator_receipt_errors(review, plan, component) == []
    assert review['boundaries'][0]['evidence'][0]['quote'] == component['context_messages'][0]['content']
    wrong = copy.deepcopy(review)
    wrong['boundaries'][0]['evidence'][0]['quote'] = 'Unrelated mountains and oceans'
    assert curator_receipt_errors(wrong, plan, component)
    wrong = copy.deepcopy(review)
    wrong['boundaries'][0]['evidence'][1]['source_message_id'] = 1
    assert curator_receipt_errors(wrong, plan, component)


def test_failed_writer_resume_reuses_successful_router_and_curator(settings):
    ingest(settings)
    seen = []
    async def fails(role, request):
        seen.append(role)
        if role == 'event_writer':
            raise ValueError('synthetic Writer failure')
        return output_for(role, request)
    with pytest.raises(ValueError):
        asyncio.run(p.advance(settings.database, include_recent=True, runner=fails))
    resumed = []
    async def succeeds(role, request):
        resumed.append(role)
        return output_for(role, request)
    assert asyncio.run(p.advance(settings.database, include_recent=True, runner=succeeds))['events'] == 1
    assert resumed == ['event_writer']
