#!/usr/bin/env python3
"""NPU 历史存量回扫的标题级预筛。

每日扫描的采集器（profile third 的 fault_scan_sources/collect.py）按 updated_at
增量取数，水位单调前进：pre-cutover 最后更新、之后无人再碰的 issue 永远进不了候选
索引。本脚本独立清点那部分存量，并用校准过的宽口径正则做第一级漏斗。

两个子命令：

    list      按 sources.json 全量拉取 issue 标题（不取正文），缓存成 JSONL
    filter    对缓存应用宽口径正则，按仓输出候选数与候选清单

与 collect.py 的分工：collect.py 负责日常增量并产出正文，本脚本只做一次性存量
清点，不写编目、不碰 collect.py 的缓存与水位。
"""

import argparse
import collections
import json
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

GITCODE = 'https://api.gitcode.com/api/v5/repos/'
GITHUB = 'https://api.github.com/search/issues'

# 宽口径：在首晚 27 个真实入库 issue 上召回 26/27。NPU_SCAN.md 第 2 条那套严格
# 关键词只有 18/27 —— 漏掉"接收偏移错误""数据丢失（0 值）""silent miscompile"
# "偏移值溢出 uint32"这类真实收录项，所以这里必须放宽。
KEEP = re.compile(
    r'精度|静默|错值|错误|乱码|未初始化|脏数据|丢失|漏|越界|溢出|竞态|不一致|不收敛'
    r'|不对|异常|偏差|偏移|nan|\binf\b|wrong|incorrect|corrupt|mismatch|garbl|garbage'
    r'|miscompile|precision|accuracy|logprob|梯度|loss|结果|输出|计算错|数值', re.I)

# 只排除标题层面就能确定无关的。不要排除"需求建议"——hccl#825（TRN-0270 的来源）
# 就是以需求形式提的缺陷。
DROP = re.compile(r'文档|资料|readme|注释|拼写|typo|\bdocs?\b|示例代码|参数名'
                  r'|描述不一致|说明不一致|一致性检视|耗时|性能优化', re.I)

# 首晚产出密度分档。T1 已有 24/27 条入库来源，T3 的 mindspore/mindspore 首晚候选
# 全部被判为非 NPU 域。
TIERS = {
    'Ascend/pytorch': 1, 'Ascend/op-plugin': 1, 'Ascend/torchair': 1,
    'cann/ops-transformer': 1, 'cann/hccl': 1, 'cann/hcomm': 1,
    'Ascend/MindSpeed': 1, 'Ascend/MindSpeed-LLM': 1, 'Ascend/MindSpeed-RL': 1,
    'Ascend/MindIE-LLM': 1, 'Ascend/msmodelslim': 1,
    'cann/ops-nn': 2, 'cann/ops-math': 2, 'cann/ge': 2, 'cann/runtime': 2,
    'Ascend/msprobe': 2, 'Ascend/mssanitizer': 2, 'mindspore/mindformers': 2,
    'mindspore/mindspore': 3,
    'vllm-project/vllm-ascend': 1, 'sgl-project/sgl-kernel-npu': 1,
    'triton-lang/triton-ascend': 1,
}

opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def request(url, token=None):
    """直连取一页 JSON；只向 api.github.com 发 token。"""
    headers = {'User-Agent': 'npu-backfill-prefilter'}
    if token and urllib.parse.urlsplit(url).hostname == 'api.github.com':
        headers['Authorization'] = 'Bearer ' + token
    last = None
    for attempt in range(3):
        try:
            with opener.open(urllib.request.Request(url, headers=headers), timeout=60) as response:
                head = {k.lower(): v for k, v in response.headers.items()}
                return head, json.loads(response.read())
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError,
                ConnectionError, json.JSONDecodeError) as exc:
            last = exc
            time.sleep(3 * (attempt + 1))
    raise SystemExit('fetch failed: %s: %s' % (url, last))


def row(item, repo, platform):
    labels = [x.get('name') for x in (item.get('labels') or []) if isinstance(x, dict)]
    return dict(repo=repo, platform=platform, number=str(item.get('number')),
                title=item.get('title') or '', state=item.get('state'),
                created=item.get('created_at'), updated=item.get('updated_at'),
                labels=labels, url=item.get('html_url') or item.get('web_url') or '')


def list_gitcode(repo, cache):
    """GitCode 列表 API 带 total_page 头，匿名可全量翻页；不取正文。"""
    rows, page = [], 1
    while True:
        query = urllib.parse.urlencode(dict(state='all', per_page=100, page=page,
                                            sort='created', direction='asc'))
        head, data = request(GITCODE + repo + '/issues?' + query)
        rows += [row(item, repo, 'gitcode') for item in data]
        total = int(head.get('total_page') or 0)
        if not data or page >= total:
            return rows, head.get('total_count')
        page += 1
        time.sleep(0.2)


def list_github(repo, token, until):
    """GitHub Search 每查最多 1000 条，按创建年份切片绕开上限。"""
    rows = []
    for year in range(2023, 2027):
        page = 1
        while True:
            q = 'repo:%s is:issue created:%d-01-01..%d-12-31 updated:<%s' % (repo, year, year, until)
            url = GITHUB + '?' + urllib.parse.urlencode(dict(q=q, per_page=100, page=page))
            head, data = request(url, token)
            rows += [row(item, repo, 'github') for item in data.get('items') or []]
            if len(data.get('items') or []) < 100 or page * 100 >= min(1000, data['total_count']):
                break
            page += 1
            time.sleep(2 if token else 7)
        time.sleep(2 if token else 7)
    return rows, len(rows)


def do_list(args):
    sources = json.load(open(args.registry))['sources']
    out = Path(args.cache)
    out.mkdir(parents=True, exist_ok=True)
    token = os.environ.get('GITHUB_TOKEN')
    for source in sources:
        repo, platform = source['repo'], source['platform']
        if args.platform and platform != args.platform:
            continue
        if args.only and repo not in args.only:
            continue
        path = out / (repo.replace('/', '__') + '.jsonl')
        if path.exists() and not args.force:
            print('cached', repo, file=sys.stderr)
            continue
        if platform == 'github' and not token:
            print('skip %s: GITHUB_TOKEN unset (anonymous search is 10/min)' % repo, file=sys.stderr)
            continue
        if platform == 'gitcode':
            rows, declared = list_gitcode(repo, out)
        else:
            rows, declared = list_github(repo, token, args.until)
        with open(str(path) + '.tmp', 'w') as handle:
            for item in rows:
                handle.write(json.dumps(item, ensure_ascii=False) + '\n')
        os.replace(str(path) + '.tmp', path)
        print('%-28s fetched %6d  declared %s' % (repo, len(rows), declared), flush=True)


def do_filter(args):
    rows = []
    for path in sorted(Path(args.cache).glob('*.jsonl')):
        rows += [json.loads(line) for line in open(path)]
    seen = collections.Counter()
    kept = []
    for item in rows:
        tier = TIERS.get(item['repo'], 3)
        if tier > args.tier:
            continue
        # 日常扫描看得见的那段不属于回扫范围。
        if item['updated'] >= args.until:
            continue
        seen[item['repo'], 'hist'] += 1
        title = item['title']
        if KEEP.search(title) and not DROP.search(title):
            seen[item['repo'], 'cand'] += 1
            kept.append(dict(item, tier=tier))
    width = max(len(r) for r, _ in seen) if seen else 10
    print('%-*s %7s %7s' % (width, 'repo', 'hist', 'cand'))
    for repo in sorted({r for r, _ in seen}, key=lambda r: -seen[r, 'cand']):
        print('%-*s %7d %7d' % (width, repo, seen[repo, 'hist'], seen[repo, 'cand']))
    print('%-*s %7d %7d' % (width, 'TOTAL', sum(v for (_, k), v in seen.items() if k == 'hist'),
                            len(kept)))
    if args.out:
        kept.sort(key=lambda r: (r['tier'], r['repo'], r['created']))
        with open(args.out, 'w') as handle:
            for item in kept:
                handle.write(json.dumps(item, ensure_ascii=False) + '\n')
        print('wrote', args.out, file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--registry', default='/workspace/inference_error_review/tools/scan/npu/sources.json')
    parser.add_argument('--cache', default='/tmp/npu_backfill')
    parser.add_argument('--until', default='2026-10-07',
                        help='回扫上界：日常扫描首轮窗口的起点，默认 2026-10-07')
    sub = parser.add_subparsers(dest='cmd', required=True)
    listing = sub.add_parser('list', help='全量拉取 issue 标题')
    listing.add_argument('--platform', choices=['gitcode', 'github'])
    listing.add_argument('--only', action='append', metavar='owner/repo',
                         help='只取 registry 里这个仓，可重复；用于试跑')
    listing.add_argument('--force', action='store_true')
    listing.set_defaults(func=do_list)
    filtering = sub.add_parser('filter', help='应用宽口径正则')
    filtering.add_argument('--tier', type=int, default=1, choices=[1, 2, 3])
    filtering.add_argument('--out', help='候选 JSONL 输出路径')
    filtering.set_defaults(func=do_filter)
    args = parser.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
