import json
import sqlite3
import hashlib
import pytest
from serein.bootstrap import initialize
from serein.config import Settings
from serein.core.store import Store, encode
from serein.compat.germany.raw_events import RawEventStore
from serein.compat.originals import Originals
from serein.legacy_migration.originals import scan_originals, import_originals
from serein.legacy_migration.scan import scan
from serein.legacy_migration.workflow import Migration
from serein.legacy_migration.history import run_history_repair
from serein.legacy_migration.dates import legacy_dates, repair_dates


@pytest.fixture
def setup(tmp_path):
    root=tmp_path/'old';(root/'buckets/dynamic').mkdir(parents=True)
    (root/'buckets/dynamic/a.md').write_text('---\nid: a\nname: Bookshop\ndate: 2026-01-02\ncreated: 2026-02-03T12:30:00+08:00\n---\nOriginal Scene','utf-8')
    archive=RawEventStore({'state_dir':str(root/'state')})
    archive.ingest([{'source':'old-chat','source_event_id':'m1','role':'user','text':'Hello bookshop.','created_at':'2026-01-02T10:01:00+08:00','session_id':'s','conversation_id':'c'},
                    {'source':'old-chat','source_event_id':'m2','role':'assistant','text':'See you there.','created_at':'2026-01-02T10:02:00+08:00','session_id':'s','conversation_id':'c'}])
    settings=Settings(tmp_path/'new/memory.db',tmp_path/'new/index.db',writable=True);initialize(settings)
    return root,settings,archive


def test_raw_archive_keeps_new_messages_and_does_not_schedule_models(setup):
    root,settings,_=setup
    current=RawEventStore({'raw_events':{'db_path':str(settings.database)}})
    current.ingest([{'source':'new-chat','source_event_id':'m1','role':'user','text':'New message','created_at':'2026-03-01T10:01:00+08:00'}])
    source=scan_originals(root)
    result=import_originals(settings.database,source)
    assert result['inserted']==2 and not result['held']
    assert import_originals(settings.database,source)['duplicate']==2
    api=Originals(settings.database)
    found=api.source_message_search(query='bookshop')['items'][0]
    message=api.source_message_read([found['id']])['items'][0]
    assert message['content']=='Hello bookshop.'
    assert message['metadata']['created_at']=='2026-01-02T10:01:00+08:00'
    assert message['metadata']['session_id']=='s' and message['metadata']['message_id']=='m1'
    with Store(settings.database) as store:
        assert store.conn.execute('SELECT count(*) FROM raw_events').fetchone()[0]==3
        assert store.conn.execute('SELECT count(*) FROM raw_processing').fetchone()[0]==2
        assert store.conn.execute("SELECT count(*) FROM raw_events r WHERE NOT EXISTS (SELECT 1 FROM raw_processing p WHERE p.raw_id=r.id)").fetchone()[0]==1
        assert store.conn.execute("SELECT count(*) FROM raw_events_fts WHERE raw_events_fts MATCH 'bookshop'").fetchone()[0]==1
        assert store.conn.execute('SELECT count(*) FROM evidence_bindings').fetchone()[0]==0


def test_changed_archive_rolls_back_and_identity_conflicts_report(setup):
    root,settings,archive=setup;source=scan_originals(root)
    with sqlite3.connect(archive.db_path) as conn:conn.execute("UPDATE raw_events SET text='Changed' WHERE source_event_id='m2'")
    with pytest.raises(ValueError,match='回滚'):import_originals(settings.database,source)
    with Store(settings.database) as store:assert store.conn.execute('SELECT count(*) FROM raw_events').fetchone()[0]==0
    source=scan_originals(root);import_originals(settings.database,source)
    with sqlite3.connect(archive.db_path) as conn:conn.execute("UPDATE raw_events SET text='Another version' WHERE source_event_id='m2'")
    report=import_originals(settings.database,scan_originals(root))
    assert report['duplicate']==1 and len(report['held'])==1 and report['inserted']==0
    with Store(settings.database) as store:
        assert store.conn.execute("SELECT text FROM raw_events WHERE source_event_id='m2'").fetchone()[0]=='Changed'


@pytest.mark.parametrize('meta,expected',[
    ({'date':'2026-01-02','created':'2026-02-03'},'2026-01-02'),
    ({'date':'','created_at':'2026-03-04T12:00:00+08:00'},'2026-03-04'),
    ({'date':'2026-99-99','created':'2026-04-05'},'2026-04-05'),({},'')])
def test_date_priority(meta,expected):assert legacy_dates(meta)['date']==expected


def test_new_dates_and_old_repair_keep_identity_and_body(setup):
    root,settings,_=setup;plan=scan(root)
    assert 'source_records' not in plan and 'source_edges' not in plan
    assert all('source_raw_event_ids' not in item and 'legacy_provenance' not in item for item in plan['items'])
    # Verify the exact old fingerprint recipe, excluding only newly added fields.
    items=[{k:v for k,v in i.items() if k not in ('date','legacy_created','legacy_comments','source_raw_event_ids','legacy_provenance')} for i in plan['items']]
    old=hashlib.sha256(json.dumps({'items':items,'edges':plan['edges'],'skipped':plan['skipped']},sort_keys=True,ensure_ascii=False).encode()).hexdigest()
    assert plan['fingerprint']==old
    migration=Migration(settings,plan,{'user_name':'User','ai_name':'AI','aliases':[]})
    try:migration.import_bodies();key=migration.ids['a']
    finally:migration.close()
    with Store(settings.database) as store:
        doc=store.read(key)
        assert doc['metadata']['date']=='2026-01-02' and doc['created_at'].startswith('2026-02-03')
        # Simulate a first-release import missing both date metadata fields.
        meta={k:v for k,v in doc['metadata'].items() if k not in ('date','created')}
        store.revise(key,expected_revision=doc['revision'],title=doc['title'],body_md=doc['body_md'],metadata=meta)
    report=run_history_repair(settings,plan)
    assert report['originals']['inserted']==2 and report['dates']['updated']==1
    with Store(settings.database) as store:
        repaired=store.read(key);assert repaired['metadata']['date']=='2026-01-02'
        assert repaired['body_md']==doc['body_md'] and repaired['title']==doc['title']
        assert store.conn.execute('SELECT count(*) FROM documents').fetchone()[0]==1
        meta={**repaired['metadata'],'date':'2026-05-06'}
        store.revise(key,expected_revision=repaired['revision'],title=doc['title'],body_md=doc['body_md'],metadata=meta)
    assert repair_dates(settings.database,plan)['updated']==0
    assert run_history_repair(settings,plan)['originals']['duplicate']==2


def test_raw_source_must_stay_inside_backup(setup,tmp_path):
    root,_,_=setup
    (root/'config.yaml').write_text('raw_events:\n  db_path: '+str(tmp_path/'elsewhere.db').replace('\\','/')+'\n','utf-8')
    assert scan_originals(root)['summary']['messages']==2
    assert scan_originals(root)['summary']['warnings']
