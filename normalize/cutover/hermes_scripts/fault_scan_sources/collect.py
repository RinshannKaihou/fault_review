#!/usr/bin/python3
"""NPU issue/PR 共享来源采集器；stdout 仅输出紧凑 JSON。"""
import argparse
from datetime import datetime, timedelta, timezone
import fcntl
import hashlib
from http.client import IncompleteRead
import json
import os
from pathlib import Path
import re
import tempfile
import urllib.error
import urllib.parse
import urllib.request

HERE = Path(__file__).resolve().parent
DEFAULT_OUTPUT = Path('/root/.hermes/profiles/third/cache/fault_scan_sources')
SCOPES = ('inference', 'training')
ERRORS = (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, ConnectionError,
          IncompleteRead, json.JSONDecodeError, UnicodeDecodeError)
API = {'github': 'https://api.github.com/repos/',
       'gitcode': 'https://api.gitcode.com/api/v5/repos/',
       'gitee': 'https://gitee.com/api/v5/repos/'}


class FetchError(ValueError):
    """结构、时间窗口或分页不可信；不得视为零发现。"""


def moment(value):
    try:
        parsed = datetime.fromisoformat(value.replace('Z', '+00:00'))
    except (ValueError, AttributeError) as exc:
        raise FetchError('invalid timestamp: ' + str(value)) from exc
    if parsed.tzinfo is None:
        raise FetchError('timestamp missing timezone')
    return parsed.astimezone(timezone.utc)


def stamp(value):
    return value.astimezone(timezone.utc).isoformat(timespec='seconds').replace('+00:00', 'Z')


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, separators=(',', ':')), encoding='utf-8')


def atomic_write_text(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix='.tmp-' + path.name + '-', dir=path.parent)
    os.close(fd)
    tmp_path = Path(tmp)
    try:
        tmp_path.write_text(content, encoding='utf-8')
        os.replace(tmp_path, path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def number_of(item):
    value = item.get('number')
    if type(value) not in (str, int) or not re.fullmatch(r'[A-Za-z0-9_-]+', str(value)):
        raise FetchError('invalid or missing original number')
    return str(value)


def github_next(headers):
    for url, rel in re.findall(r'<([^>]+)>;\s*rel="([^"]+)"', headers.get('link', '')):
        if rel == 'next':
            parsed = urllib.parse.urlsplit(url)
            if parsed.scheme != 'https' or parsed.netloc != 'api.github.com' or parsed.fragment:
                raise FetchError('unsafe GitHub pagination URL')
            return url
    return None


def page_links(headers):
    links = {}
    for url, rel in re.findall(r'<([^>]+)>;\s*rel="([^"]+)"', headers.get('link', '')):
        pages = urllib.parse.parse_qs(urllib.parse.urlsplit(url).query).get('page', [])
        if len(pages) != 1 or not pages[0].isdigit() or int(pages[0]) < 1:
            raise FetchError('invalid pagination Link')
        links[rel] = int(pages[0])
    if headers.get('link') and not links:
        raise FetchError('unparseable pagination Link')
    return links


def validate_page(data, headers, platform, page, seen, start, end):
    if not isinstance(data, list) or len(data) > 100:
        raise FetchError('expected JSON array of <=100 items')
    for item in data:
        if not isinstance(item, dict):
            raise FetchError('expected object in list')
        number = number_of(item)
        if number in seen:
            raise FetchError('duplicate original number across pages: ' + number)
        seen.add(number)
        if start is not None:
            updated = moment(item.get('updated_at'))
            if updated < moment(start) or (platform == 'gitcode' and updated > moment(end)):
                raise FetchError('server ignored requested updated window: ' + number)
    if platform == 'gitcode' or (platform == 'gitee' and 'total_count' in headers):
        if not all(str(headers.get(k, '')).isdigit() for k in ('total_count', 'total_page')):
            raise FetchError('missing/invalid pagination totals')
        total, pages = int(headers['total_count']), int(headers['total_page'])
        if pages != (total + 99) // 100 or (pages and page > pages):
            raise FetchError('inconsistent page count')
        expected = min(100, max(0, total - (page - 1) * 100))
        if len(data) != expected or (page >= pages and len(seen) != total):
            raise FetchError('pagination count/unique ID mismatch')
        return page < pages
    links = page_links(headers)
    if 'next' in links and (links['next'] != page + 1 or len(data) != 100):
        raise FetchError('invalid next page or truncated page')
    if 'last' in links and (links['last'] < page or (links['last'] > page and 'next' not in links)):
        raise FetchError('pagination ended before declared last page')
    return 'next' in links or (platform == 'gitee' and len(data) == 100)


def canonical_url(source, kind, number):
    segment = '/issues/' if kind == 'issue' else {'github': '/pull/', 'gitcode': '/merge_requests/', 'gitee': '/pulls/'}[source['platform']]
    return source['web_url'].rstrip('/') + segment + number


def normalize(source, item, kind, batch):
    number = number_of(item)
    key = ':'.join([source['platform'], source['repo'], kind, number])
    if not all(isinstance(item.get(k), str) for k in ('title', 'state')):
        raise FetchError('missing title/state')
    body = '' if item.get('body') is None else item['body']
    if not isinstance(body, str):
        raise FetchError('non-string body')
    created, updated = stamp(moment(item.get('created_at'))), stamp(moment(item.get('updated_at')))
    content = item['title'] + '\n' + body
    return dict(key=key, platform=source['platform'], repo=source['repo'], number=number,
                type=kind, title=item['title'], url=canonical_url(source, kind, number),
                state=item['state'], created_at=created, updated_at=updated, scopes=source['scopes'],
                content_digest=hashlib.sha256(content.encode()).hexdigest(),
                keywords=[k for k in ('nan', 'precision', 'accuracy', 'mismatch', 'incorrect', 'overflow', 'crash', 'hang', 'deadlock', 'oom', '精度', '溢出', '卡死', '静默', '错误') if k in content.lower()],
                body_path=str(batch / 'bodies' / (hashlib.sha256(key.encode()).hexdigest() + '.md')))


def auth_headers(url):
    headers = {'User-Agent': 'npu-fault-collector/1', 'Accept': 'application/json'}
    if urllib.parse.urlsplit(url).hostname == 'api.github.com':
        token = os.environ.get('GITHUB_TOKEN') or os.environ.get('TOKEN', '')
        if os.environ.get('GITHUB_TOKEN') or token.startswith(('ghp_', 'github_pat_')):
            headers['Authorization'] = 'Bearer ' + token
    return headers


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None  # 不向重定向目标泄露 Authorization，也不隐式换源。


def request(url, raw):
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())
    raw.parent.mkdir(parents=True, exist_ok=True)
    metadata = dict(url=url, fetched_at=stamp(datetime.now(timezone.utc)))
    try:
        with opener.open(urllib.request.Request(url, headers=auth_headers(url)), timeout=45) as response:
            headers = {k.lower(): v for k, v in response.headers.items() if k.lower() in
                       ('total_count', 'total_page', 'link', 'content-type', 'x-ratelimit-remaining', 'x-ratelimit-reset')}
            metadata.update(status=response.status, headers=headers)
            body = response.read()
    except urllib.error.HTTPError as exc:
        raw.with_suffix('.body').write_bytes(exc.read())
        dump(raw.with_suffix('.json'), dict(metadata, status=exc.code, error='HTTPError'))
        raise
    except (urllib.error.URLError, TimeoutError, ConnectionError, IncompleteRead) as exc:
        raw.with_suffix('.body').write_bytes(exc.partial if isinstance(exc, IncompleteRead) else b'')
        dump(raw.with_suffix('.json'), dict(metadata, error=type(exc).__name__, message=str(exc)))
        raise
    raw.with_suffix('.body').write_bytes(body)
    dump(raw.with_suffix('.json'), metadata)
    return json.loads(body), headers


def fetch_list(source, endpoint, start, end, raw_dir, progress):
    platform = source['platform']
    params = dict(state='all', sort='updated', direction='asc', per_page=100)
    params.update(dict(since=start) if platform == 'github' else dict(updated_after=start, updated_before=end))
    seen, first_totals, page, following = set(), None, 1, None
    while True:
        progress.update(endpoint=endpoint, page=page)
        params['page'] = page
        url = following or API[platform] + source['repo'] + '/' + endpoint + '?' + urllib.parse.urlencode(params)
        data, headers = request(url, raw_dir / (endpoint + '-' + str(page)))
        if platform == 'gitcode':
            totals = (headers.get('total_count'), headers.get('total_page'))
            if first_totals is not None and totals != first_totals:
                raise FetchError('pagination totals changed during traversal')
            first_totals = totals
            progress.update(total_count=totals[0], total_page=totals[1])
        more = validate_page(data, headers, platform, page, seen, start, end)
        if platform == 'github':
            declared = page_links(headers).get('last')
            if declared and progress.get('total_page', declared) != declared:
                raise FetchError('last page changed during traversal')
            if declared:
                progress['total_page'] = declared
            if not more and page != progress.get('total_page', page):
                raise FetchError('missing pages before declared last page')
        progress['validated_pages'] = page
        for item in data:
            if moment(item['updated_at']) <= moment(end):  # GitHub 无 before；并发新更新留给下一次 overlap。
                yield item
        if not more:
            return
        if platform == 'github':
            following = github_next(headers)  # 保留官方 Link 的 after/before，不能只重建 page。
        page += 1


def _seed_legacy_cursor(old_manifest):
    """Derive a conservative initial cursor when upgrading an existing state.json without scope_cursors."""
    sinces = [c.get('since') for c in (old_manifest.get('coverage') or []) if c.get('since')]
    if sinces:
        return stamp(min(moment(s) for s in sinces))
    return stamp(moment(old_manifest['generated_at']))


def _write_scope_delta(batch, scope, since_ts):
    scope_file = batch / (scope + '.jsonl')
    since_dt = moment(since_ts)
    delta_lines = []
    if scope_file.exists():
        for line in scope_file.read_text(encoding='utf-8').splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if moment(row['updated_at']) >= since_dt:
                delta_lines.append(line + '\n')
    delta_path = batch / (scope + '_delta.jsonl')
    atomic_write_text(delta_path, ''.join(delta_lines))
    return delta_path, len(delta_lines)


def summary(batch, scope, cached, scope_cursor=None, window_days=7):
    data = json.loads((batch / 'manifest.json').read_text(encoding='utf-8'))
    gen_dt = moment(data['generated_at'])
    if scope_cursor:
        since_ts = stamp(moment(scope_cursor) - timedelta(hours=1))
    else:
        since_ts = stamp((gen_dt - timedelta(days=window_days)).replace(hour=0, minute=0, second=0, microsecond=0))
    delta_path, delta_count = _write_scope_delta(batch, scope, since_ts)
    return dict(
        batch=str(batch),
        candidates=str(batch / (scope + '.jsonl')),
        delta_candidates=str(delta_path),
        delta_count=delta_count,
        scope_window=dict(since=since_ts, until=data['generated_at'], cursor=scope_cursor),
        normalized=str(batch / 'normalized.jsonl'),
        scope=scope,
        cached=cached,
        candidate_count=data['counts'][scope],
        **{k: data[k] for k in (
            'complete', 'coverage', 'failures', 'registry_hash',
            'generated_at', 'registry_sources', 'selected_sources',
        )},
    )


def _save_state(root, state_obj):
    dump(root / 'state.next', state_obj)
    os.replace(root / 'state.next', root / 'state.json')


def mark_consumed(scope, until, output, force=False):
    """Advance a single scope's evaluation cursor after a scan run commits & pushes."""
    if scope not in SCOPES:
        raise FetchError('invalid scope: ' + str(scope))
    valid_until = stamp(moment(until))
    root = output.resolve()
    root.mkdir(parents=True, exist_ok=True)
    with (root / '.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        state = json.loads((root / 'state.json').read_text(encoding='utf-8')) if (root / 'state.json').exists() else dict(watermarks={})
        cursors = dict(state.get('scope_cursors') or {})
        prev = cursors.get(scope)
        if prev and not force and moment(valid_until) < moment(prev):
            new_cursor = prev
        else:
            new_cursor = valid_until
        cursors[scope] = new_cursor
        state['scope_cursors'] = cursors
        _save_state(root, state)
    return dict(complete=True, scope=scope, previous_cursor=prev, cursor=new_cursor)


def run(scope, only, output, force=False, registry=HERE / 'sources.json', now=None):
    registry_bytes = registry.read_bytes()
    config = json.loads(registry_bytes)
    registry_hash = hashlib.sha256(registry_bytes).hexdigest()
    window_days = config.get('window_days', 7)
    sources = [s for s in config['sources'] if not only or s['repo'] == only]
    if not sources:
        raise FetchError('unknown repository: ' + str(only))
    if len({(s['platform'], s['repo']) for s in sources}) != len(sources):
        raise FetchError('duplicate registry source')
    root = output.resolve() / ('only-' + hashlib.sha256(only.encode()).hexdigest()[:12]) if only else output.resolve()
    root.mkdir(parents=True, exist_ok=True)
    # 单输出目录全局锁；22 仓共用一个 batch，按 scope_cursors 隔离两个扫描任务的消费水位。
    with (root / '.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        now = now or datetime.now(timezone.utc)
        state = json.loads((root / 'state.json').read_text(encoding='utf-8')) if (root / 'state.json').exists() else dict(watermarks={})
        previous = Path(state['batch']) if state.get('batch') else None
        scope_cursors = dict(state.get('scope_cursors') or {})
        old = json.loads((previous / 'manifest.json').read_text(encoding='utf-8')) if previous and (previous / 'manifest.json').exists() else None
        if old and not only:
            seed = _seed_legacy_cursor(old)
            seeded = False
            for sc in SCOPES:
                if sc not in scope_cursors:
                    scope_cursors[sc] = seed
                    seeded = True
            if seeded:
                state['scope_cursors'] = scope_cursors
                _save_state(root, state)

        if old and not force:
            age = (now - moment(old['generated_at'])).total_seconds()
            if old['complete'] and old['registry_hash'] == registry_hash and 0 <= age < config['cache_hours'] * 3600:
                return summary(previous, scope, True, scope_cursor=scope_cursors.get(scope), window_days=window_days)

        stage = Path(tempfile.mkdtemp(prefix='.building-', dir=root))
        final = root / ('batch-' + now.strftime('%Y%m%dT%H%M%S%f') + '-' + stage.name.removeprefix('.building-'))
        rows, coverage, failures = {}, [], []
        source_map = {(s['platform'], s['repo']): s for s in sources}
        if previous and (previous / 'normalized.jsonl').exists():
            for line in (previous / 'normalized.jsonl').read_text(encoding='utf-8').splitlines():
                row = json.loads(line)
                source = source_map.get((row['platform'], row['repo']))
                if source:
                    row['scopes'] = source['scopes']
                    row['url'] = canonical_url(source, row['type'], row['number'])
                    rows[row['key']] = row  # 不丢弃尚未由另一 scope 评估的历史候选；正文仍引用不可变旧 batch。
        watermarks = state['watermarks'].copy()
        for source in sources:
            source_key = source['platform'] + ':' + source['repo']
            start = stamp(moment(watermarks[source_key]) - timedelta(hours=1)) if source_key in watermarks else stamp((now.astimezone(timezone.utc) - timedelta(days=window_days)).replace(hour=0, minute=0, second=0, microsecond=0))
            end, progress, pages, fetched = stamp(now), {}, [], 0
            raw_dir = stage / 'raw' / source['platform'] / source['repo']
            complete = False
            try:
                for endpoint in (['issues'] if source['platform'] == 'github' else ['issues', 'pulls']):
                    progress = dict(endpoint=endpoint, page=1, validated_pages=0)
                    for item in fetch_list(source, endpoint, start, end, raw_dir, progress):
                        kind = 'pr' if endpoint == 'pulls' or 'pull_request' in item else 'issue'
                        row = normalize(source, item, kind, final)
                        body_path = stage / Path(row['body_path']).relative_to(final)
                        body_path.parent.mkdir(parents=True, exist_ok=True)
                        body_path.write_text(item.get('body') or '', encoding='utf-8')
                        rows[row['key']] = row
                        fetched += 1
                    pages.append(progress.copy())
                complete = True
                watermarks[source_key] = end
            except ERRORS + (FetchError,) as exc:
                failures.append(dict(platform=source['platform'], repo=source['repo'],
                                     error=type(exc).__name__, message=str(exc), **progress))
            coverage.append(dict(platform=source['platform'], repo=source['repo'], complete=complete,
                                 fetched=fetched, since=start, until=end, pages=pages,
                                 incomplete=None if complete else progress.copy()))
        ordered = sorted(rows.values(), key=lambda r: (r['updated_at'], r['key']), reverse=True)
        counts = {}
        for view in ('normalized', 'inference', 'training'):
            selected = [r for r in ordered if view == 'normalized' or view in r['scopes']]
            (stage / (view + '.jsonl')).write_text(''.join(json.dumps(r, ensure_ascii=False, separators=(',', ':')) + '\n' for r in selected), encoding='utf-8')
            counts[view] = len(selected)
        manifest = dict(complete=not failures, generated_at=stamp(now), registry_hash=registry_hash,
                        coverage=coverage, failures=failures, counts=counts,
                        registry_sources=len(config['sources']), selected_sources=len(sources))
        dump(stage / 'manifest.json', manifest)
        os.replace(stage, final)
        next_state = dict(batch=str(final), watermarks=watermarks)
        if not only and scope_cursors:
            next_state['scope_cursors'] = scope_cursors
        _save_state(root, next_state)
        return summary(final, scope, False, scope_cursor=scope_cursors.get(scope), window_days=window_days)


def parse_url(value, sources):
    parsed = urllib.parse.urlsplit(value)
    hosts = {'github.com': 'github', 'gitcode.com': 'gitcode', 'atomgit.com': 'gitcode', 'gitee.com': 'gitee'}
    match = re.fullmatch(r'/([A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+)/(issues|pull|pulls|merge_requests)/([A-Za-z0-9_-]+)/?', parsed.path)
    if parsed.scheme != 'https' or parsed.netloc not in hosts or parsed.query or parsed.fragment or not match:
        raise FetchError('expected complete canonical issue/PR URL; bare IDs are forbidden')
    repo, segment, number = match.groups()
    platform = hosts[parsed.netloc]
    source = dict(platform=platform, repo=repo, web_url='https://' + parsed.netloc + '/' + repo, scopes=[])
    for known in sources:
        if known['platform'] == platform and known['repo'].casefold() == repo.casefold():
            source = known.copy()
            break
    return source, 'issue' if segment == 'issues' else 'pr', number


def detail(value, output, registry):
    source, kind, number = parse_url(value, json.loads(registry.read_text(encoding='utf-8'))['sources'])
    root = output.resolve() / 'details'
    root.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix='.building-', dir=root))
    final = root / ('detail-' + stage.name.removeprefix('.building-'))
    endpoint = ('issues/' if kind == 'issue' else 'pulls/') + number
    base = API[source['platform']] + source['repo'] + '/'
    result = dict(complete=False, url=canonical_url(source, kind, number), number=number,
                  detail=str(final), comments=str(final / 'comments.jsonl'), body_path=None,
                  comment_count=0, failures=[], pages=[])
    progress = dict(endpoint=endpoint, page=1)
    with (stage / 'comments.jsonl').open('w', encoding='utf-8') as comments:
        try:
            item, _ = request(base + endpoint, stage / 'raw' / 'detail')
            if not isinstance(item, dict) or number_of(item) != number:
                raise FetchError('detail original number mismatch or unexpected JSON object')
            if source['platform'] == 'github' and kind == 'issue' and 'pull_request' in item:
                raise FetchError('GitHub /issues URL returned a PR; use canonical /pull/' + number + ' URL')
            row = normalize(source, item, kind, final)
            body_path = stage / Path(row['body_path']).relative_to(final)
            body_path.parent.mkdir()
            body_path.write_text(item.get('body') or '', encoding='utf-8')
            dump(stage / 'detail.json', row)
            result['body_path'] = row['body_path']
            endpoints = [endpoint + '/comments']
            if source['platform'] == 'github' and kind == 'pr':
                endpoints = ['issues/' + number + '/comments', endpoint + '/comments', endpoint + '/reviews']
            for endpoint in endpoints:
                page, seen, first_totals, expected_pages, following = 1, set(), None, None, None
                while True:
                    progress = dict(endpoint=endpoint, page=page)
                    data, headers = request(following or base + endpoint + '?' + urllib.parse.urlencode(dict(per_page=100, page=page)),
                                            stage / 'raw' / (endpoint.replace('/', '-') + '-' + str(page)))
                    if not isinstance(data, list) or not all(isinstance(c, dict) for c in data):
                        raise FetchError('expected comment JSON array')
                    totals = (headers.get('total_count'), headers.get('total_page'))
                    if first_totals is not None and totals != first_totals:
                        raise FetchError('comment pagination totals changed')
                    first_totals = totals
                    identities = [dict(number=c.get('id')) for c in data]
                    more = validate_page(identities, headers, source['platform'], page, seen, None, None)
                    declared = page_links(headers).get('last')
                    if declared and expected_pages not in (None, declared):
                        raise FetchError('comment last page changed')
                    expected_pages = declared or expected_pages
                    if not more and expected_pages is not None and page != expected_pages:
                        raise FetchError('comment pages missing')
                    for comment in data:
                        comment['id'] = number_of(dict(number=comment.get('id')))
                        comment['endpoint'] = endpoint
                        comments.write(json.dumps(comment, ensure_ascii=False, separators=(',', ':')) + '\n')
                    result['comment_count'] += len(data)
                    if not more:
                        result['pages'].append(dict(endpoint=endpoint, pages=page, unique_ids=len(seen)))
                        break
                    if source['platform'] == 'github':
                        following = github_next(headers)
                    page += 1
            result['complete'] = True
        except ERRORS + (FetchError,) as exc:
            result['failures'].append(dict(error=type(exc).__name__, message=str(exc), **progress))
    dump(stage / 'manifest.json', result)
    os.replace(stage, final)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', nargs='?', choices=['detail', 'mark-consumed'])
    parser.add_argument('url', nargs='?')
    parser.add_argument('--scope', choices=['inference', 'training'])
    parser.add_argument('--until', help='mark-consumed 确认已评估完成的截止时间（ISO8601 UTC）')
    parser.add_argument('--only', help='隔离采集单仓，不修改正式全仓缓存/水位')
    parser.add_argument('--force', action='store_true')
    parser.add_argument('--registry', type=Path, default=HERE / 'sources.json')
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    if args.command == 'detail':
        if not args.url:
            parser.error('detail 需要 <完整URL>')
    elif args.command == 'mark-consumed':
        if not args.scope or not args.until:
            parser.error('mark-consumed 需要 --scope inference|training 与 --until <ISO8601>')
    elif not args.scope:
        parser.error('需要 --scope inference|training，或 detail <完整URL> / mark-consumed')
    try:
        if args.command == 'detail':
            result = detail(args.url, args.output_dir, args.registry)
        elif args.command == 'mark-consumed':
            result = mark_consumed(args.scope, args.until, args.output_dir, force=args.force)
        else:
            result = run(args.scope, args.only, args.output_dir, args.force, args.registry)
    except ERRORS + (FetchError,) as exc:
        result = dict(complete=False, error=type(exc).__name__, message=str(exc))
    print(json.dumps(result, ensure_ascii=False, separators=(',', ':')))
    return 0 if result['complete'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
