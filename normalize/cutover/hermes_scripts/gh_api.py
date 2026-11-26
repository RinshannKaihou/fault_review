#!/usr/bin/python3
"""GitHub REST API helper for Hermes daily fault scans.

Reads GITHUB_TOKEN from environment or ~/.hermes/profiles/third/.env without
ever echoing the secret in command lines, stdout, stderr, or exception strings.
Blocks cross-origin redirects so Authorization headers cannot leak off api.github.com.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import shlex
import urllib.error
import urllib.parse
import urllib.request

API_HOST = "api.github.com"
API_BASE = f"https://{API_HOST}"
DEFAULT_ENV_FILE = Path("/root/.hermes/profiles/third/.env")


class GhApiError(RuntimeError):
    """Safe error message with no credential material."""


def read_env_token(env_file: Path = DEFAULT_ENV_FILE) -> str:
    """Extract GITHUB_TOKEN safely from env or a .env file without shell eval."""
    env_tok = os.environ.get("GITHUB_TOKEN", "").strip()
    if env_tok:
        return env_tok
    if not env_file or not env_file.exists():
        return ""
    val = ""
    for raw in env_file.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        m = re.match(r"^(?:export\s+)?GITHUB_TOKEN=(.+)$", line)
        if m:
            try:
                toks = shlex.split(m.group(1), comments=True)
                val = toks[0].strip() if toks else ""
            except ValueError:
                val = m.group(1).strip().strip("\"'")
    return val


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def _build_url(endpoint: str, params: list[tuple[str, str]]) -> str:
    if endpoint.startswith("https://") or endpoint.startswith("http://"):
        parsed = urllib.parse.urlsplit(endpoint)
        if parsed.scheme != "https" or parsed.netloc != API_HOST or parsed.fragment:
            raise GhApiError(f"refusing non-GitHub API URL host: {parsed.netloc or endpoint}")
        existing = urllib.parse.parse_qsl(parsed.query, keep_blank_values=True)
        merged = urllib.parse.urlencode(existing + params)
        return urllib.parse.urlunsplit(("https", API_HOST, parsed.path, merged, ""))
    path = "/" + endpoint.lstrip("/")
    parsed = urllib.parse.urlsplit(API_BASE + path)
    if parsed.netloc != API_HOST or parsed.fragment:
        raise GhApiError(f"invalid endpoint: {endpoint}")
    existing = urllib.parse.parse_qsl(parsed.query, keep_blank_values=True)
    merged = urllib.parse.urlencode(existing + params)
    return urllib.parse.urlunsplit(("https", API_HOST, parsed.path, merged, ""))


def _next_link(link_header: str) -> str | None:
    for url, rel in re.findall(r'<([^>]+)>;\s*rel="([^"]+)"', link_header or ""):
        if rel == "next":
            parsed = urllib.parse.urlsplit(url)
            if parsed.scheme != "https" or parsed.netloc != API_HOST or parsed.fragment:
                raise GhApiError("unsafe pagination Link URL")
            return url
    return None


def fetch_github(
    endpoint: str,
    *,
    params: list[tuple[str, str]] | None = None,
    max_pages: int = 1,
    timeout: int = 45,
    env_file: Path = DEFAULT_ENV_FILE,
    opener=None,
) -> dict:
    if max_pages < 1:
        raise GhApiError("max_pages must be >= 1")
    token = read_env_token(env_file)
    headers = {
        "User-Agent": "fault-scan-gh-api/1",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"

    opener = opener or urllib.request.build_opener(urllib.request.ProxyHandler({}), _NoRedirect())
    qparams = list(params or [])
    url = _build_url(endpoint, qparams)

    pages_fetched = 0
    combined_items = []
    search_mode = None
    first_payload = None
    last_headers = {}
    status = 0
    next_url: str | None = url

    while next_url and pages_fetched < max_pages:
        req = urllib.request.Request(next_url, headers=headers)
        try:
            with opener.open(req, timeout=timeout) as resp:
                status = resp.status
                last_headers = {k.lower(): v for k, v in resp.headers.items()}
                raw_body = resp.read()
        except urllib.error.HTTPError as exc:
            err_headers = {k.lower(): v for k, v in (exc.headers.items() if exc.headers else [])}
            body_preview = ""
            try:
                body_preview = exc.read().decode("utf-8", errors="replace")[:500]
            except Exception:
                pass
            return {
                "ok": False,
                "authenticated": bool(token),
                "status": exc.code,
                "url": next_url,
                "pages_fetched": pages_fetched,
                "ratelimit_limit": _int_or_none(err_headers.get("x-ratelimit-limit")),
                "ratelimit_remaining": _int_or_none(err_headers.get("x-ratelimit-remaining")),
                "ratelimit_reset": _int_or_none(err_headers.get("x-ratelimit-reset")),
                "error": f"HTTP {exc.code}",
                "message": body_preview,
            }
        except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
            return {
                "ok": False,
                "authenticated": bool(token),
                "status": 0,
                "url": next_url,
                "pages_fetched": pages_fetched,
                "error": type(exc).__name__,
                "message": str(exc)[:300],
            }

        payload = json.loads(raw_body.decode("utf-8"))
        pages_fetched += 1
        if first_payload is None:
            first_payload = payload
            if isinstance(payload, dict) and isinstance(payload.get("items"), list) and "total_count" in payload:
                search_mode = "search"
            elif isinstance(payload, list):
                search_mode = "list"
            else:
                search_mode = "object"

        if search_mode == "search":
            combined_items.extend(payload.get("items") or [])
        elif search_mode == "list":
            combined_items.extend(payload)

        next_url = _next_link(last_headers.get("link", ""))
        if search_mode == "object":
            next_url = None

    if search_mode == "search" and isinstance(first_payload, dict):
        final_data = dict(first_payload, items=combined_items)
        total_count = first_payload.get("total_count")
        fetched_count = len(combined_items)
        incomplete_results = bool(first_payload.get("incomplete_results"))
    elif search_mode == "list":
        final_data = combined_items
        total_count = None
        fetched_count = len(combined_items)
        incomplete_results = False
    else:
        final_data = first_payload
        total_count = None
        fetched_count = 1 if first_payload is not None else 0
        incomplete_results = False

    has_more = next_url is not None
    return {
        "ok": True,
        "authenticated": bool(token),
        "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "status": status,
        "url": url,
        "pages_fetched": pages_fetched,
        "has_more": has_more,
        "next_url": next_url,
        "complete": (not has_more) and (not incomplete_results),
        "total_count": total_count,
        "fetched_count": fetched_count,
        "ratelimit_limit": _int_or_none(last_headers.get("x-ratelimit-limit")),
        "ratelimit_remaining": _int_or_none(last_headers.get("x-ratelimit-remaining")),
        "ratelimit_reset": _int_or_none(last_headers.get("x-ratelimit-reset")),
        "data": final_data,
    }


def _int_or_none(val) -> int | None:
    return int(val) if val is not None and str(val).isdigit() else None


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("endpoint", help="GitHub API path (e.g. /search/issues, /repos/o/r/issues/1, /rate_limit)")
    ap.add_argument("-q", "--query", help="Search query string `q` (for /search/*)")
    ap.add_argument("-p", "--param", action="append", default=[], metavar="KEY=VAL",
                    help="Query parameter KEY=VAL (repeatable)")
    ap.add_argument("--per-page", type=int, default=None, help="Items per page (max 100)")
    ap.add_argument("--max-pages", type=int, default=1, help="Max pagination pages to follow via Link header")
    ap.add_argument("--timeout", type=int, default=45)
    ap.add_argument("--out", type=Path, help="Write full `data` JSON to this file and print compact ledger on stdout")
    ap.add_argument("--env-file", type=Path, default=DEFAULT_ENV_FILE)
    args = ap.parse_args(argv)

    params: list[tuple[str, str]] = []
    if args.query is not None:
        params.append(("q", args.query))
    for item in args.param:
        if "=" not in item:
            ap.error(f"--param expects KEY=VAL, got: {item}")
        k, v = item.split("=", 1)
        params.append((k, v))
    if args.per_page is not None:
        params.append(("per_page", str(args.per_page)))
    elif args.max_pages > 1 or args.endpoint.lstrip("/").startswith("search/"):
        if not any(k == "per_page" for k, _ in params):
            params.append(("per_page", "100"))

    try:
        res = fetch_github(
            args.endpoint,
            params=params,
            max_pages=args.max_pages,
            timeout=args.timeout,
            env_file=args.env_file,
        )
    except GhApiError as exc:
        print(json.dumps({"ok": False, "error": "GhApiError", "message": str(exc)}, ensure_ascii=False))
        return 2

    if res.get("ok") and args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(res["data"], ensure_ascii=False, indent=1), encoding="utf-8")
        ledger = {k: v for k, v in res.items() if k != "data"}
        ledger["out"] = str(args.out)
        print(json.dumps(ledger, ensure_ascii=False))
    else:
        print(json.dumps(res, ensure_ascii=False))
    return 0 if res.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
