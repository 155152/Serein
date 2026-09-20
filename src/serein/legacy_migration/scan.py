import hashlib
import json
from pathlib import Path, PurePosixPath, PureWindowsPath
import re
import sqlite3
import tarfile
import yaml
from .dates import legacy_dates
from .comments import capture_comments, comment_summary

BUCKETS={'dynamic','permanent','archive','archived','feel','whisper'}
AFFECT={'reflection','affect_anchor','和弦情绪','情绪和弦','情绪锚点','和弦'}
DAILY_IMPRESSION_TAGS={'relationship_weather','daily_impression','weekly_impression'}
LEGACY_PROVENANCE_FIELDS={
    'source_summary_ids','historical_evidence_dates','historical_candidate_id','historical_backfill_version',
    'historical_promotion_target','historical_memory_scope','historical_upstream_scope','promotion_notes',
    'protected_historical_event','from_historical_backfill','daily_chat_memory_candidate_id',
    'daily_chat_memory_reason','from_daily_chat','source_conversation_turn_ids',
}


def is_old_self_anchor(meta):
    """Recognize the retired self-only bucket without making it a new category."""
    markers={'自我','self_anchor','first_person_anchor','first-person-anchor'}
    def marked(value):
        if isinstance(value,str):
            values=value.split(',')
        elif isinstance(value,(list,tuple,set)):
            values=value
        else:
            values=[value]
        return any(str(part).strip().lower() in markers for part in values)
    return meta.get('self_anchor') is True or any(marked(meta.get(key)) for key in
        ('tags','bucket_tags','domain','profile_kind','bucket_profile_kind','anchor_kind','kind','type','source'))


def is_daily_impression(meta):
    """Recognize retired relationship-weather records without scanning prose."""
    old_id=str(meta.get('id') or '').strip().lower()
    if old_id.startswith('reflection_daily_'):
        return True
    tags=meta.get('tags')
    if isinstance(tags,str):
        tags=tags.split(',')
    elif not isinstance(tags,(list,tuple,set)):
        tags=[]
    return bool({str(tag).strip().lower() for tag in tags} & DAILY_IMPRESSION_TAGS)


def _tags(meta):
    values=meta.get('tags') or []
    if isinstance(values,str):values=values.split(',')
    elif not isinstance(values,(list,tuple,set)):values=[]
    return {str(value).strip().lower() for value in values}


def is_source_record(meta):
    """Recognize Haven-Ombre evidence buckets without treating them as memories."""
    return (str(meta.get('target_representation') or '').strip().lower()=='bucket_source_record'
            or 'source_record' in _tags(meta)
            or (str(meta.get('type') or '').strip().lower()=='source' and bool(meta.get('source_type'))))


def _jsonable(value):
    if value is None or isinstance(value,(str,int,float,bool)):return value
    if isinstance(value,dict):return {str(key):_jsonable(item) for key,item in value.items()}
    if isinstance(value,(list,tuple,set)):return [_jsonable(item) for item in value]
    return str(value)


def _source_raw_event_ids(meta,body):
    value=meta.get('source_raw_event_ids')
    origin='frontmatter'
    if value is None and 'raw_event_refs' in _tags(meta):
        match=re.search(r'(?m)^\s*raw_event_ids:\s*([0-9][0-9,\s]*)\s*$',body)
        if match:
            value=[part.strip() for part in match[1].split(',') if part.strip()]
            origin='body_raw_event_ids'
    if value is None:return [],''
    if not isinstance(value,(list,tuple,set)):raise ValueError('source_raw_event_ids 必须是数组')
    result=[]
    for raw in value:
        try:key=int(raw)
        except (TypeError,ValueError):raise ValueError('source_raw_event_ids 必须是正整数') from None
        if key<=0:raise ValueError('source_raw_event_ids 必须是正整数')
        if key not in result:result.append(key)
    return result,origin


def _legacy_provenance(meta,raw_ids,raw_id_origin):
    result={key:_jsonable(meta[key]) for key in LEGACY_PROVENANCE_FIELDS if key in meta}
    if raw_ids:
        result['source_raw_event_ids']=list(raw_ids)
        result['source_raw_event_ids_origin']=raw_id_origin
    return result


def clean_body(text):
    output=[];excluded=None;fence=None;removed=0
    for line in text.splitlines():
        marker=re.match(r'^\s*(`{3,}|~{3,})',line)
        if marker:
            if fence is None:fence=marker[1][0]
            elif marker[1][0]==fence:fence=None
        heading=None if fence or marker else re.match(r'^(#{1,6})\s+(.+?)\s*#*\s*$',line)
        if heading:
            level=len(heading[1]);name=heading[2].strip().lower().replace('-','_').replace(' ','_')
            if excluded is not None and level<=excluded:excluded=None
            if name in AFFECT:excluded=level;removed+=1
            if excluded is not None or level==3:continue
        if excluded is None:output.append(line)
    return '\n'.join(output).strip(),removed


def unpack(source, workspace):
    source=Path(source).resolve()
    if source.is_dir():return source
    if not tarfile.is_tarfile(source):raise ValueError('旧库必须是目录或 tar/tar.gz 备份')
    sha=hashlib.sha256()
    with source.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):sha.update(block)
    root=Path(workspace)/'sources'/sha.hexdigest()[:20]
    done=root/'.complete'
    if done.is_file() and not done.is_symlink() and done.read_text('utf-8')=='safe-v2':return root
    root.mkdir(parents=True,exist_ok=True)
    root=root.resolve()
    total=0;count=0
    with tarfile.open(source) as archive:
        for entry in archive:
            count+=1;total+=entry.size
            if count>100_000 or total>4*1024**3:raise ValueError('备份解压超过 4 GiB 或 10 万条目，请使用已核对的目录导入')
            path=PurePosixPath(entry.name)
            if (path.is_absolute() or '..' in path.parts or '\\' in entry.name
                or PureWindowsPath(entry.name).drive or ':' in entry.name
                or any(re.fullmatch(r'(CON|PRN|AUX|NUL|COM[1-9¹²³]|LPT[1-9¹²³])(?:\..*)?',part,re.I) for part in path.parts)
                or any(part.endswith((' ','.')) for part in path.parts)
                or path.name=='.complete' or entry.issym() or entry.islnk()):
                raise ValueError('备份包含不安全的路径或链接')
            if not entry.isfile():continue
            if entry.size>1024**3:raise ValueError('备份单文件超过 1 GiB，请先手动解压并核对')
            target=root.joinpath(*path.parts)
            if target.is_symlink() or not target.resolve().is_relative_to(root):
                raise ValueError('备份包含不安全的路径或链接')
            target.parent.mkdir(parents=True,exist_ok=True)
            with archive.extractfile(entry) as src,target.open('wb') as dst:
                while block:=src.read(1024*1024):dst.write(block)
    done.write_text('safe-v2','utf-8')
    return root


def locate_root(path):
    path=Path(path).resolve()
    if (path/'buckets').is_dir():return path,path/'buckets'
    if any((path/k).is_dir() for k in BUCKETS):return path.parent,path
    roots=[p for p in path.iterdir() if p.is_dir() and (p/'buckets').is_dir()]
    if len(roots)==1:return roots[0],roots[0]/'buckets'
    raise ValueError('没有找到 buckets 下的 dynamic/permanent/feel/archive 等记忆目录')


def scan(path):
    root,buckets=locate_root(path)
    items=[];source_records=[];errors=[];skipped=[];seen=set()
    for group in sorted(BUCKETS):
        directory=buckets/group
        if not directory.is_dir():continue
        for file in sorted(directory.rglob('*.md')):
            if file.is_symlink() or not file.resolve().is_relative_to(buckets):continue
            relative=file.relative_to(root).as_posix()
            try:
                raw=file.read_text('utf-8-sig')
                match=re.match(r'\A---\s*\n(.*?)\n---\s*(?:\n|$)',raw,re.S)
                if not match:raise ValueError('缺少 YAML 元数据，不能推断旧记忆身份')
                meta=yaml.safe_load(match[1])
                if not isinstance(meta,dict):raise ValueError('元数据格式错误')
                old_id=str(meta.get('id') or '').strip()
                if not old_id:raise ValueError('缺少 id')
                if old_id in seen:raise ValueError('重复 id，需要先确认保留哪一份')
                seen.add(old_id)
                if is_old_self_anchor(meta):
                    skipped.append({'path':relative,'old_id':old_id,'reason':'self_anchor_discarded',
                        'source_hash':hashlib.sha256(raw.encode()).hexdigest()})
                    continue
                if is_daily_impression(meta):
                    skipped.append({'path':relative,'old_id':old_id,'reason':'daily_impression_discarded',
                        'source_hash':hashlib.sha256(raw.encode()).hexdigest()})
                    continue
                source_body=raw[match.end():].strip()
                if is_source_record(meta):
                    if not source_body:raise ValueError('旧 source record 没有正文')
                    source_records.append({'old_id':old_id,'path':relative,
                        'title':str(meta.get('name') or meta.get('title') or old_id),'body':source_body,
                        'source_hash':hashlib.sha256(raw.encode()).hexdigest(),'metadata':_jsonable(meta)})
                    continue
                body,removed=clean_body(raw[match.end():])
                if not body and removed:
                    skipped.append({'path':relative,'old_id':old_id,'reason':'empty_after_section_removal',
                        'source_hash':hashlib.sha256(raw.encode()).hexdigest()})
                    continue
                if not body:raise ValueError('原记录没有正文')
                kind='diary' if group in ('feel','whisper') or meta.get('type') in ('feel','whisper') else 'scene'
                archived=group in ('archive','archived') or meta.get('type') in ('archive','archived')
                raw_ids,raw_id_origin=_source_raw_event_ids(meta,raw[match.end():]) if kind=='scene' else ([], '')
                provenance=_legacy_provenance(meta,raw_ids,raw_id_origin)
                preserved=set(LEGACY_PROVENANCE_FIELDS) if provenance else set()
                if raw_ids:preserved.add('source_raw_event_ids')
                item={'old_id':old_id,'path':relative,'kind':kind,
                    'title':str(meta.get('name') or meta.get('title') or old_id),
                    'body':body,'archived':archived,'created':str(meta.get('created') or ''),
                    **legacy_dates(meta),
                    "legacy_comments":capture_comments(meta),
                    'source_hash':hashlib.sha256(raw.encode()).hexdigest(),'removed_affect_sections':removed,
                    'metadata_fields_discarded':sorted(set(map(str,meta))-preserved)}
                if raw_ids:item['source_raw_event_ids']=raw_ids
                if provenance:item['legacy_provenance']=provenance
                items.append(item)
            except (ValueError,UnicodeError,yaml.YAMLError) as exc:errors.append({'path':relative,'error':str(exc)})
    from .edges import endpoints,scan_edges
    all_edges,edge_sources,edge_errors=scan_edges(root)
    errors.extend(edge_errors)
    source_ids={item['old_id'] for item in source_records}
    source_edges=[];edges=[]
    for edge in all_edges:
        left,right=(str(value or '') for value in endpoints(edge))
        (source_edges if left in source_ids or right in source_ids else edges).append(edge)
    from .companion import scan_companion
    try:
        companion=scan_companion(root)
    except (ValueError, OSError, sqlite3.Error) as exc:
        errors.append({'path':'companion_state','error':str(exc)})
        companion={'summary':{'error':'旧 Persona／备忘状态扫描失败'}}
    # Keep existing memory IDs and ledgers stable; companion state has its own fingerprint.
    # Extra date fields must not change old IDs or make a completed batch import twice.
    identity_items=[]
    for item in items:
        identity_item={k:v for k,v in item.items() if k not in
            ('date','legacy_created','legacy_comments','source_raw_event_ids','legacy_provenance')}
        if item.get('source_raw_event_ids'):identity_item['source_raw_event_ids']=item['source_raw_event_ids']
        if item.get('legacy_provenance'):identity_item['legacy_provenance']=item['legacy_provenance']
        identity_items.append(identity_item)
    identity={'items':identity_items,'edges':edges,'skipped':skipped}
    # Preserve the historical fingerprint recipe for ordinary Ombre libraries so
    # an existing migration can resume after upgrading Serein. Haven-Ombre's
    # extra evidence layer participates only when it is actually present.
    if source_records:identity['source_records']=source_records
    if source_edges:identity['source_edges']=source_edges
    fingerprint=hashlib.sha256(json.dumps(identity,sort_keys=True,ensure_ascii=False).encode()).hexdigest()
    from .originals import scan_originals
    try:originals=scan_originals(root)
    except (ValueError,OSError,sqlite3.Error) as exc:
        errors.append({'path':'raw_events','error':str(exc)})
        originals={'summary':{'messages':0,'warnings':['旧原文库扫描失败，请核对备份。']}}
    from .history import scan_history
    try:
        history=scan_history(root)
    except (ValueError, OSError, sqlite3.Error, KeyError, TypeError) as exc:
        errors.append({'path':'history','error':str(exc)})
        history={'summary':{'warnings':['历史数据扫描失败，请修正后重新预览。']}}
    summary={'scenes':sum(i['kind']=='scene' for i in items),'diaries':sum(i['kind']=='diary' for i in items),
        'archived':sum(i['archived'] for i in items),'old_edges':len(edges),'edge_sources':edge_sources,
        'edge_warning':('' if edge_sources else
            '所选目录没有可识别的旧关系边文件。正文可以单独迁入；这不表示旧库一定没有关系。请核对旧服务实际使用的 state 目录及 Docker 挂载，完整备份应包含 state/memory_edges.jsonl 或存有关系表的 SQLite 库。只有代码和 buckets 的备份可能缺少状态数据。'),
        'errors':len(errors),'empty_after_cleanup_skipped':len(skipped),
        'affect_sections_removed':sum(i['removed_affect_sections'] for i in items),
        'self_anchors_discarded':sum(i['reason']=='self_anchor_discarded' for i in skipped),
        'daily_impressions_discarded':sum(i['reason']=='daily_impression_discarded' for i in skipped),
        'whole_vector_file_present':(buckets/'embeddings.db').is_file(),
        'whole_vectors_without_input_proof':'旧 embeddings 表通常没有原文/哈希，无法验证的向量会重新生成',
        'daily_impressions':'不导入日印象；不扫描 state 内删除备份和迁移预览', 'companion':companion['summary'], 'history':history['summary'],'originals':originals['summary'],'memory_comments':comment_summary(items)}
    result={'root':str(root),'buckets':str(buckets),'fingerprint':fingerprint,'items':items,
        'edges':edges,'errors':errors,'skipped':skipped,'companion':companion,'history':history,'originals':originals,
        'summary':summary}
    evidence_items=[item for item in items if item.get('source_raw_event_ids')]
    if source_records:
        result['source_records']=source_records;summary['source_records']=len(source_records)
    if source_edges:
        result['source_edges']=source_edges;summary['source_edges_archived']=len(source_edges)
    if source_records or source_edges:
        summary['old_edges_scanned']=len(all_edges)
    if evidence_items:
        evidence_refs=[int(value) for item in evidence_items for value in item['source_raw_event_ids']]
        summary.update(scenes_with_raw_evidence=len(evidence_items),raw_evidence_refs=len(evidence_refs),
            unique_raw_evidence_refs=len(set(evidence_refs)),
            body_raw_ref_scenes=sum((item.get('legacy_provenance') or {}).get('source_raw_event_ids_origin')=='body_raw_event_ids' for item in evidence_items))
    return result
