import asyncio
import copy
import json
from datetime import datetime
import pytest
from test_public_features import settings, ingest, output_for, synthetic_runner, raw_archive
from serein.core.store import Store, promoted_scene_id
from serein.compat.events import reference_blockers
from serein.deployment import save_settings
from serein.extensions import pipeline as p
from serein.extensions import pipeline_latest as latest


def curator_task(settings):
    task=asyncio.run(p.advance(settings.database,include_recent=True))
    while task['role']=='track_router':
        p.submit(settings.database,task['job_id'],output_for(task['role'],task['request']))
        task=asyncio.run(p.advance(settings.database,include_recent=True))
    assert task['role']=='event_curator'
    return task


def test_parked_correction_is_readable_but_not_owned(settings):
    raw_archive(settings).ingest([
        {'source_event_id':'u','session_id':'one','role':'user','text':'Meet at three','created_at':'2025-02-02T02:00:00+08:00'},
        {'source_event_id':'a','session_id':'one','role':'assistant','text':'Agreed','created_at':'2025-02-02T02:01:00+08:00'},
        {'source_event_id':'tail','session_id':'one','role':'user','text':'Wait, I cannot make three','created_at':'2025-02-02T02:55:00+08:00'}],source='test')
    p.initialize(settings.database)
    batch=p.new_batch(settings.database,False,datetime.fromisoformat('2025-02-02T04:00:00+08:00'))
    task=curator_task(settings);component=task['request']['component']
    rendered=latest.event_curator_model_input(component)
    assert [u['scope'] for u in rendered['units']]==['stable','stable','parked']
    assert 'cannot make' in rendered['units'][-1]['messages'][0]['text']
    output=output_for('event_curator',task['request']);output['events'][0]['owned_unit_roots'].append(component['parked_context_source_ids'][0])
    with pytest.raises(ValueError):p.validate(task['request'],output)
    deferred={'events':[],'skip_unit_roots':[],'defer_unit_roots':[m['id'] for m in component['messages']]}
    assert len(latest.normalize_event_curator_output(deferred,component)['defer_source_message_ids'])==2


def test_latest_rolling_policy_does_not_force_unrelated_leaves_and_defers_blockers(settings):
    ingest(settings);asyncio.run(p.advance(settings.database,include_recent=True,runner=synthetic_runner));ingest(settings,2)
    task=curator_task(settings);component=copy.deepcopy(task['request']['component'])
    base=component['base_event_candidates'][0]
    unrelated={**base,'event_id':'unrelated','source_message_ids':[90,91]}
    component['base_event_candidates'].append(unrelated)
    plan=output_for('event_curator',task['request'])
    normalized=latest.normalize_event_curator_output(plan,component)
    assert normalized['events'][0]['base_event_ids']==[base['event_id']]
    base['protected']=True
    normalized=latest.normalize_event_curator_output(plan,component)
    assert normalized['events']==[] and len(normalized['defer_source_message_ids'])==2
    assert normalized['hard_skips'][0]['blocking_flags']==['protected']


def test_three_stages_and_writer_sees_exact_predecessor_originals(settings):
    ingest(settings);seen=[]
    async def runner(role,request):seen.append(role);return output_for(role,request)
    assert asyncio.run(p.advance(settings.database,include_recent=True,runner=runner))['events']==1
    assert seen==list(p.ROLES)==['track_router','event_curator','event_writer']
    ingest(settings,2)
    task=curator_task(settings);p.submit(settings.database,task['job_id'],output_for(task['role'],task['request']))
    task=asyncio.run(p.advance(settings.database,include_recent=True));prompt=task['request']['prompt']
    assert len(task['request']['messages'])==4
    assert task['role']=='event_writer' and 'Book club plan 1' in prompt and 'Book club plan 2' in prompt
    assert '<previous_events_json>' in prompt and '正文最多 1000 字，这是写作硬上限而非目标' in prompt
    with Store(settings.database) as store:
        detail=json.loads(store.conn.execute('SELECT details_json FROM pipeline_event_details').fetchone()[0])
        assert 'evidence' not in detail
        assert set(detail['source_activity_roles'])=={'1','2'}


def test_scene_worthy_event_is_atomically_promoted_to_scene(settings):
    ingest(settings)

    async def runner(role, request):
        output = output_for(role, request)
        if role == 'event_writer':
            output['scene_worthy'] = True
        return output

    result = asyncio.run(p.advance(settings.database, include_recent=True, runner=runner))
    assert result['events'] == 1

    with Store(settings.database, read_only=True) as store:
        event_id = store.conn.execute(
            "SELECT id FROM documents WHERE kind='event' ORDER BY created_at DESC LIMIT 1"
        ).fetchone()[0]
        scene_id = promoted_scene_id(event_id)
        event = store.read(event_id)
        scene = store.read(scene_id)
        assert scene is not None
        assert scene['kind'] == 'scene'
        assert scene['title'] == event['title']
        assert scene['body_md'] == event['body_md']
        assert scene['metadata']['memory_value_source'] == 'automatic_event_scene'
        assert scene['metadata']['write_contract'] == 'event-to-scene-auto-v1'
        assert scene['metadata']['scene_cues'] == ['Book club plan']
        assert scene['metadata']['promoted_from_event']['id'] == event_id
        event_sources = store.conn.execute(
            "SELECT source_id FROM evidence_bindings WHERE document_id=? AND active=1 ORDER BY source_id",
            (event_id,),
        ).fetchall()
        scene_sources = store.conn.execute(
            "SELECT source_id FROM evidence_bindings WHERE document_id=? AND active=1 ORDER BY source_id",
            (scene_id,),
        ).fetchall()
        assert [row[0] for row in scene_sources] == [row[0] for row in event_sources]
        surface = store.surface_state(event_id)
        assert 'promoted_to_scene' in surface['reasons']
        assert 'covered_by_scene' in surface['reasons']
        detail = json.loads(store.conn.execute(
            "SELECT details_json FROM pipeline_event_details WHERE event_id=?",
            (event_id,),
        ).fetchone()[0])
        assert detail['auto_scene'] == {'status': 'promoted', 'scene_id': scene_id}


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
    assert asyncio.run(p.advance(settings.database, include_recent=True, runner=synthetic_runner))['events'] == 1
    with Store(settings.database) as store:
        event_id = store.conn.execute(
            "SELECT id FROM documents WHERE kind='event' AND lifecycle='active'"
        ).fetchone()[0]
        source_rows = store.conn.execute(
            "SELECT source_id,metadata_json FROM evidence_bindings WHERE document_id=? AND active=1 ORDER BY id",
            (event_id,),
        ).fetchall()
        store.create(
            'manual-scene', 'scene', 'Manual memory', 'Independently authored memory',
            metadata={'memory_value_source': 'authored_scene'},
        )
        for source in source_rows:
            store.bind(
                'manual-scene', source['source_id'], metadata=json.loads(source['metadata_json'])
            )
        assert 'active_scene_dependency' in reference_blockers(store.conn, event_id)


def test_settled_event_is_queued_only_when_arc_linker_is_selected(settings):
    save_settings(settings.database,{
        'models':[{'id':'local','model':'synthetic','base_url':'http://127.0.0.1:9/v1'}],
        'assignments':{'arc_linker':'local'}})
    ingest(settings)
    result=asyncio.run(p.advance(settings.database,include_recent=True,runner=synthetic_runner))
    assert result['events']==1
    with Store(settings.database,read_only=True) as store:
        row=store.conn.execute('SELECT event_id,event_fingerprint,status FROM pipeline_arc_links').fetchone()
        fact=store.conn.execute('SELECT item_id,fingerprint FROM fact_events WHERE status=\'active\'').fetchone()
        assert tuple(row)==(fact['item_id'],fact['fingerprint'],'pending')


def test_writer_body_uses_1000_guidance_with_1500_tolerance():
    request={'messages':[{'id':1,'content':'A book was returned'}]}
    output=output_for('event_writer',request)
    output['event_draft']='书还了。'
    assert latest.validate_event_writer_result(output)==[]
    output['event_draft']='书'*1500
    assert latest.validate_event_writer_result(output)==[]
    output['event_draft']='书'*1501
    assert '正文超过容错上限 1500 字：1501 字' in ' '.join(latest.validate_event_writer_result(output))
    output['title']=''
    assert '标题为空' in latest.validate_event_writer_result(output)


def test_writer_result_caps_auxiliary_kept_details_without_changing_body():
    result=output_for('event_writer',{'messages':[{'id':1,'content':'A book was returned'}]})
    result['event_draft']='书还了。'
    result['kept_details']=[f'anchor-{i}' for i in range(8)]
    body=result['event_draft']
    latest.normalize_event_writer_result(result)
    assert result['kept_details']==[f'anchor-{i}' for i in range(6)]
    assert result['event_draft']==body
    assert latest.validate_event_writer_result(result)==[]


def test_writer_repair_prompt_makes_detail_limit_explicit():
    failed={'kept_details':[str(i) for i in range(8)],'discarded_details':[]}
    prompt=latest.build_event_writer_repair_prompt('base',failed,['kept_details 必须有 1–6 项：8'])
    assert 'kept_details must contain at most 6 items' in prompt
    assert 'Keep only the 6 most essential anchors' in prompt


def test_public_writer_materializes_source_grounded_rules_with_configured_names():
    with latest.identity_scope({'ai_name': 'Atlas', 'user_name': 'Lin'}):
        rules = latest.materialize_agent_rules('event_writer')
    assert 'Atlas 在回复中对Lin的话作出的展开' in rules
    assert '最小完整语义单位' in rules and '局部回应不能改变前句' in rules
    assert '不额外补出理解、判断、解释等动作' in rules
    assert '不把某一种归属句式当成模板' in rules
    assert '原文停留在“想、打算、建议' in rules
    assert '反例三' in rules and '台灯' in rules
    assert '我把这句话理解成' not in rules
    assert 'Haven' not in rules and '小雨' not in rules


def test_model_counting_tolerance_settles_without_truncation(settings):
    ingest(settings)
    curator=curator_task(settings)
    p.submit(settings.database,curator['job_id'],output_for(curator['role'],curator['request']))
    task=asyncio.run(p.advance(settings.database,include_recent=True))
    output=output_for('event_writer',task['request'])
    output['event_draft']='书'*1500
    p.submit(settings.database,task['job_id'],output)
    assert asyncio.run(p.advance(settings.database,include_recent=True))['events']==1
    with Store(settings.database,read_only=True) as store:
        saved=store.conn.execute('SELECT body FROM fact_events').fetchone()[0]
        assert saved==output['event_draft'] and len(saved)==1500


def test_api_prompt_compacts_frozen_writer_transport_without_changing_evidence():
    frozen_row={'source_message_id':7,'created_at':'2025-01-01T00:00:00Z','speaker':'她','text':'exact source text',
                'saved_snowflake':False,'memory_event_source':False,'attachment_refs':[],
                'evidence_role':'owned','activity_role':'primary_activity'}
    prompt='head\n\nRULES\n\n<event_reading_block_json>\n'+json.dumps([frozen_row],ensure_ascii=False)+'\n</event_reading_block_json>\n<materialized_track_cards_json>\n[]\n</materialized_track_cards_json>\n<track_context_events_json>\n[]\n</track_context_events_json>\n<previous_events_json>\n[]\n</previous_events_json>\n'
    compact=p.api_prompt_for_model({'role':'event_writer','rules':'RULES','prompt':prompt})
    assert '\n\nRULES\n\n' not in compact
    match=json.loads(compact.split('<event_reading_block_json>\n',1)[1].split('\n</event_reading_block_json>',1)[0])
    assert match[0]['text']=='exact source text'
    assert match[0]['source_message_id']==7
    assert 'saved_snowflake' not in match[0]
    assert 'memory_event_source' not in match[0]
    assert 'attachment_refs' not in match[0]


def test_writer_transcript_omits_default_empty_transport_fields_but_keeps_signals():
    rows=latest.writer_transcript_payload([
        {'id':1,'role':'user','content':'plain','created_at':'2025-01-01T00:00:00Z','metadata':{}},
        {'id':2,'role':'assistant','content':'with attachment','created_at':'2025-01-01T00:01:00Z','metadata':{
            'memory_event_source':True,
            'attachments':[{'id':'img-1','kind':'image','name':'photo.png','mime_type':'image/png'}],
        }},
    ])
    assert 'saved_snowflake' not in rows[0]
    assert 'memory_event_source' not in rows[0]
    assert 'attachment_refs' not in rows[0]
    assert rows[1]['memory_event_source'] is True
    assert rows[1]['attachment_refs'][0]['attachment_id']=='img-1'


def test_writer_prompt_examples_match_both_evidence_outcomes():
    prompt=latest.build_event_writer_prompt('2025-01-01','',[{'id':1,'role':'user','content':'A book was returned'}])
    samples=[json.loads(line) for line in prompt.splitlines() if line.startswith('{"evidence_sufficient":')]
    assert len(samples)==2
    sufficient,insufficient=samples
    for sample in samples:
        assert latest.validate_event_writer_result(sample)==[]
    sufficient['recallable']=False
    assert latest.validate_event_writer_result(sufficient)==[]
    assert insufficient['evidence_sufficient'] is False and insufficient['recallable'] is False
    assert insufficient['title']==insufficient['event_draft']==''
    assert insufficient['kept_details']==insufficient['discarded_details']==[]


def test_writer_prompt_requires_grounded_affective_and_embodied_fidelity():
    prompt=latest.build_event_writer_prompt('2025-01-01','',[{'id':1,'role':'user','content':'Synthetic owned source'}])
    assert '证据约束同时意味着保真义务' in prompt
    assert '不得因为压缩而默认视为修辞、气氛或旁支' in prompt
    assert '产生明显情绪反应' in prompt and '发生亲密互动' in prompt
    assert '普通事务或技术处理没有上述情绪、关系或身体内容时保持简洁' in prompt
    assert '不为了“有温度”强行补入感受' in prompt
    assert '由任一方或双方明确表达的欲望、边界、回应和转折' in prompt
    assert '不得因为“更有情感”而补写' in prompt
    assert '她去拿快递，我牵挂着她出门的每一步' in prompt
    assert '事实表面正确，但把原文已经明确存在的情绪转折和具体欲望抽象掉了' in prompt
    assert '不得为了凑成双方反应替我增加身体感受' in prompt


@pytest.mark.parametrize('accepted',[False,True])
def test_old_pending_evidence_job_is_bypassed_and_history_preserved(settings,accepted):
    from fastapi.testclient import TestClient
    from serein.api.http import create_app
    from serein.core.store import encode
    ingest(settings)
    task=curator_task(settings)
    p.submit(settings.database,task['job_id'],output_for(task['role'],task['request']))
    batch_id=task['request']['batch_id'];old_id=batch_id+':event_evidence:0:0'
    old_request={'role':'event_evidence','batch_id':batch_id,'prompt':'Retired task','identity':task['request']['identity']}
    old_output=encode({'evidence_points':['historical receipt']}) if accepted else None
    with Store(settings.database) as store:
        store.conn.execute('INSERT INTO pipeline_jobs(id,batch_id,role,request_json,output_json) VALUES (?,?,?,?,?)',
            (old_id,batch_id,'event_evidence:0:0',encode(old_request),old_output))
        store.conn.execute("UPDATE background_state SET value_json=json_set(value_json,'$.stage','event_evidence','$.status','awaiting_agent') WHERE name='work:pipeline'")
    client=TestClient(create_app(settings,token='test',live=True),headers={'Authorization':'Bearer test'})
    assert client.get('/v1/pipeline/status').json()['stage']!='event_evidence'
    with pytest.raises(ValueError,match='retired'):p.submit(settings.database,old_id,{'anything':'old client'})
    writer=asyncio.run(p.advance(settings.database,include_recent=True))
    assert writer['role']=='event_writer' and len(writer['request']['messages'])==2
    p.submit(settings.database,writer['job_id'],output_for('event_writer',writer['request']))
    assert asyncio.run(p.advance(settings.database,include_recent=True))['events']==1
    with Store(settings.database,read_only=True) as store:
        assert store.conn.execute('SELECT output_json FROM pipeline_jobs WHERE id=?',(old_id,)).fetchone()[0]==old_output
        assert store.conn.execute('SELECT count(*) FROM pipeline_jobs').fetchone()[0]==4
        assert store.conn.execute('SELECT count(*) FROM raw_processing').fetchone()[0]==2
        assert 'evidence' not in json.loads(store.conn.execute('SELECT details_json FROM pipeline_event_details').fetchone()[0])


def test_one_bounded_context_request_and_no_foreign_ownership(settings):
    ingest(settings);task=curator_task(settings);component=task['request']['component'];root=min(m['id'] for m in component['messages'])
    query={'context_request':{'track_id':component['track_ids'][0],'before_message_id':root,'reason':'missing_subject'}}
    p.submit(settings.database,task['job_id'],query)
    next_task=asyncio.run(p.advance(settings.database,include_recent=True))
    assert next_task['request']['context_read'] and next_task['request']['component']['context_receipt']['read_source_ids']==[]
    with pytest.raises(ValueError):p.submit(settings.database,next_task['job_id'],query)
    p.submit(settings.database,next_task['job_id'],output_for(next_task['role'],next_task['request']))


def test_daytime_routing_does_not_create_events_or_consume_originals(settings,monkeypatch):
    for i in range(1,6):ingest(settings,i)
    save_settings(settings.database,{'models':[{'id':'local','model':'synthetic','base_url':'http://127.0.0.1:9/v1'}],'assignments':{'track_router':'local'}})
    async def complete(model,payload):
        with Store(settings.database,read_only=True) as store:
            request=json.loads(store.conn.execute('SELECT request_json FROM pipeline_jobs WHERE output_json IS NULL ORDER BY rowid DESC LIMIT 1').fetchone()[0])
        assert request['role']=='track_router'
        return {'choices':[{'message':{'content':json.dumps(output_for('track_router',request))}}]}
    monkeypatch.setattr('serein.model_runtime.complete',complete)
    asyncio.run(p.flush_routes(settings.database))
    with Store(settings.database) as store:
        assert store.conn.execute('SELECT COUNT(*) FROM pipeline_routes').fetchone()[0]==10
        assert store.conn.execute('SELECT COUNT(*) FROM raw_processing').fetchone()[0]==0
        assert store.conn.execute("SELECT COUNT(*) FROM documents WHERE kind='event'").fetchone()[0]==0
    assert asyncio.run(p.advance(settings.database,include_recent=True))['role']=='event_curator'


def test_identity_rendering_never_rewrites_source_words(settings):
    names={'user_name':'Nori','ai_name':'Atlas'};save_settings(settings.database,{'identity':names})
    original='Literal User and AI are words in the original.'
    with latest.identity_scope(names):
        prompt=latest.build_event_writer_prompt('2025-01-01','',[{'id':1,'role':'user','content':original}])
    assert original in prompt and 'Nori' in prompt and 'Atlas' in prompt and '{ai_name}' not in prompt
    assert 'Nori把台灯送修' in prompt


def test_configured_names_are_literal_values_not_recursive_templates(settings):
    from serein.semantic_setup import render_example
    names={'user_name':'Reader {ai_name}', 'ai_name':'Guide "A"'}
    template='{user_name} asks {ai_name}'
    expected='Reader {ai_name} asks Guide "A"'
    assert render_example(template,names)==expected
    with latest.identity_scope(names):
        assert latest._identity_text(template)==expected
    # Freshly loaded Writer examples use the current saved instance names.
    save_settings(settings.database, {'identity':{'user_name':'NewReader','ai_name':'NewGuide'}})
    rules=p.rules('event_writer',settings.database)
    assert 'NewReader把台灯送修' in rules and 'NewGuide' in rules


def test_images_keep_ownership_and_only_curator_receives_pixels(settings,monkeypatch):
    uri='data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+aX1cAAAAASUVORK5CYII='
    raw_archive(settings).ingest([
        {'source_event_id':'u','session_id':'image','role':'user','text':'This is the book','created_at':'2025-01-01T00:00:00Z','metadata':{'attachments':[{'kind':'image','url':uri,'mime_type':'image/png'}]}},
        {'source_event_id':'a','session_id':'image','role':'assistant','text':'The blue book','created_at':'2025-01-01T00:01:00Z'}],source='test')
    save_settings(settings.database,{'models':[{'id':'local','model':'synthetic','base_url':'http://127.0.0.1:9/v1'}],'assignments':{r:'local' for r in p.ROLES}})
    async def complete(model,payload):
        with Store(settings.database,read_only=True) as store:request=json.loads(store.conn.execute('SELECT request_json FROM pipeline_jobs WHERE output_json IS NULL ORDER BY rowid DESC LIMIT 1').fetchone()[0])
        if request['role']=='event_curator':
            assert request['images'][0]['evidence_role']=='stable'
            assert payload['messages'][1]['content'][1]['image_url']['url']==uri
        if request['role']=='event_writer':
            assert request['images']==[]
            assert request['curator_image_transcriptions'][0]['evidence_role']=='owned'
            assert isinstance(payload['messages'][1]['content'],str) and uri not in payload['messages'][1]['content']
        return {'choices':[{'message':{'content':json.dumps(output_for(request['role'],request))}}]}
    monkeypatch.setattr('serein.model_runtime.complete',complete)
    assert asyncio.run(p.advance(settings.database,include_recent=True))['events']==1


def test_parse_model_json_tolerates_fences_and_prose():
    from serein.extensions.pipeline import parse_model_json
    assert parse_model_json('{"a":1}') == {'a':1}
    assert parse_model_json('```json\n{"a":1}\n```') == {'a':1}
    assert parse_model_json('```\n{"a":1}\n```') == {'a':1}
    assert parse_model_json('Looking at this batch:\n{"a":1}\nmore notes') == {'a':1}
    assert parse_model_json('\n\n```json\n{"message_assignments":[]}\n```') == {'message_assignments':[]}
    with pytest.raises(json.JSONDecodeError):
        parse_model_json('not json at all, no braces either')


def test_track_router_truncates_overlong_subject_and_throughline():
    long_subject = '题' * 200
    long_throughline = '尾' * 700
    output = {
        'message_assignments': [
            {'source_message_id':1,'primary_track_ref':'new:1','context_track_refs':[],'routing_role':'primary_activity'},
            {'source_message_id':2,'primary_track_ref':'new:1','context_track_refs':[],'routing_role':'primary_activity'},
        ],
        'track_updates': [
            {'track_ref':'new:1','subject':long_subject,'throughline':long_throughline,'event_policy':'default','status':'active'},
        ],
    }
    assignments, cards, _ = p.normalize_event_track_message_output(
        output, [{'id':1},{'id':2}], [], session_id=1, next_track_ordinal=1)
    card = cards[0]
    assert len(card['subject']) == 160 and card['subject'].endswith('…')
    assert len(card['throughline']) == 598 and card['throughline'].endswith('…')
    assert card['subject'] == long_subject[:159] + '…'
    assert card['throughline'] == long_throughline[:597] + '…'
