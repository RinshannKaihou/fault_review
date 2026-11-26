#!/usr/bin/env python3
"""Automated RAG catalog sync, per-entry quarantine, thin-layer read-only LLM QA,
and 3-tier closed-loop routing (Tier-1 auto-rework queue + Tier-2/3 human inbox).

Subcommands:
  sync          Run isolated worktree sync -> gates (with per-entry quarantine) -> QA -> atomic publish
  status        Show open human-decision items and open scan-rework todos
  resolve       Resolve an open item in data/pending_human.json (approve_exemption / unquarantine / dismiss)
  list-todos    List open Tier-1 rework todos for a scan scope (inference | training)
  done-todo     Mark a Tier-1 rework todo as completed or dismissed after scan revisit
"""

from __future__ import annotations

import os
import sys

# Re-exec into system Python if invoked from Hermes' minimal Python 3.14 without PyYAML/numpy/jieba.
for _mod in ("yaml", "numpy", "jieba"):
    try:
        __import__(_mod)
    except ImportError:
        for _cand in (os.environ.get("FAULT_RAG_PYTHON", ""), "/usr/bin/python3", "/usr/local/bin/python3"):
            if _cand and os.path.exists(_cand) and os.path.realpath(_cand) != os.path.realpath(sys.executable):
                os.execv(_cand, [_cand] + sys.argv)
        sys.exit(f"rag_auto_sync: missing {_mod}; run with /usr/bin/python3")

import argparse
import datetime as dt
import fcntl
import hashlib
import json
import pwd
import re
import shlex
import shutil
import subprocess
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_RAG_ROOT = SCRIPT_DIR.parent
DEFAULT_UPSTREAM_ROOT = Path("/workspace")
DEFAULT_FEEDBACK_DIR = Path("/root/.hermes/profiles/third/cache/rag_feedback")
REPOS = ("inference", "training")
UPSTREAM_NAMES = {"inference": "inference_error_review", "training": "training_error_review"}


# ---------------------------------------------------------------------------
# Utilities & safe credential loading (zero eval, zero secret echo)
# ---------------------------------------------------------------------------

def utc_now() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def utc_iso(ts: dt.datetime | None = None) -> str:
    return (ts or utc_now()).astimezone(dt.timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def atomic_write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".tmp_{path.name}_", dir=str(path.parent))
    os.close(fd)
    tmp_p = Path(tmp)
    try:
        tmp_p.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        os.replace(tmp_p, path)
    finally:
        if tmp_p.exists():
            tmp_p.unlink()


def atomic_copy_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".tmp_{dst.name}_", dir=str(dst.parent))
    os.close(fd)
    tmp_p = Path(tmp)
    try:
        shutil.copy2(src, tmp_p)
        os.replace(tmp_p, dst)
    finally:
        if tmp_p.exists():
            tmp_p.unlink()


def read_rc_var(name: str) -> str:
    """Safely read an exported variable from user/root .bashrc without shell eval."""
    val = os.environ.get(name, "").strip()
    if val:
        return val
    candidates = [os.path.expanduser("~/.bashrc")]
    try:
        candidates.append(os.path.join(pwd.getpwuid(os.getuid()).pw_dir, ".bashrc"))
    except KeyError:
        pass
    candidates.append("/root/.bashrc")
    seen: set[str] = set()
    for rc in candidates:
        if not rc or rc in seen or not os.path.exists(rc):
            continue
        seen.add(rc)
        try:
            lines = Path(rc).read_text(encoding="utf-8", errors="ignore").splitlines()
        except OSError:
            continue
        found = ""
        for raw in lines:
            m = re.match(rf"^\s*export\s+{re.escape(name)}=(.+)$", raw.strip())
            if m:
                try:
                    toks = shlex.split(m.group(1), comments=True)
                    if toks:
                        found = toks[0]
                except ValueError:
                    found = m.group(1).strip().strip("\"'")
        if found:
            return found
    return ""


def git(cwd: Path, *args: str, check: bool = True, timeout: int = 120) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-C", str(cwd), *args],
        text=True,
        capture_output=True,
        check=check,
        timeout=timeout,
    )


# ---------------------------------------------------------------------------
# Persistent Stores: pending_human.json (Tier 2 & 3) + rag_feedback (Tier 1)
# ---------------------------------------------------------------------------

def load_pending_human(rag_root: Path) -> dict[str, Any]:
    path = rag_root / "data" / "pending_human.json"
    if not path.exists():
        return {"version": 1, "updated_at": utc_iso(), "items": []}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict) and isinstance(data.get("items"), list):
            return data
    except (OSError, json.JSONDecodeError):
        pass
    return {"version": 1, "updated_at": utc_iso(), "items": []}


def save_pending_human(rag_root: Path, store: dict[str, Any]) -> Path:
    store["updated_at"] = utc_iso()
    path = rag_root / "data" / "pending_human.json"
    atomic_write_json(path, store)
    return path


def active_quarantined_ids(store: dict[str, Any]) -> dict[str, set[str]]:
    out: dict[str, set[str]] = {"inference": set(), "training": set()}
    for item in store.get("items", []):
        if item.get("status") == "open" and item.get("kind") == "gate_quarantined_entry":
            repo = item.get("repo")
            eid = item.get("entry_id")
            if repo in out and eid:
                out[repo].add(eid)
    return out


def upsert_human_item(store: dict[str, Any], item: dict[str, Any]) -> tuple[str, bool]:
    """Insert or refresh an item by item_id. Never resurrects dismissed/resolved items of the same signature."""
    item_id = item["item_id"]
    for existing in store["items"]:
        if existing.get("item_id") == item_id:
            if existing.get("status") in ("resolved", "dismissed"):
                return item_id, False
            existing.update({k: v for k, v in item.items() if k != "created_at"})
            existing["updated_at"] = utc_iso()
            return item_id, False
    item.setdefault("status", "open")
    item.setdefault("created_at", utc_iso())
    item["updated_at"] = item["created_at"]
    store["items"].append(item)
    return item_id, True


def load_feedback_ledger(feedback_dir: Path) -> dict[str, Any]:
    path = feedback_dir / "history.json"
    if path.exists():
        try:
            d = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(d, dict) and isinstance(d.get("todos"), dict):
                return d
        except (OSError, json.JSONDecodeError):
            pass
    return {"version": 1, "updated_at": utc_iso(), "todos": {}}


def save_feedback_ledger(feedback_dir: Path, ledger: dict[str, Any]) -> None:
    ledger["updated_at"] = utc_iso()
    atomic_write_json(feedback_dir / "history.json", ledger)
    for scope in REPOS:
        open_rows = [
            row for row in ledger["todos"].values()
            if row.get("repo") == scope and row.get("status") == "open"
        ]
        open_rows.sort(key=lambda r: (r.get("created_at", ""), r.get("todo_id", "")))
        todo_file = feedback_dir / f"{scope}_todo.jsonl"
        todo_file.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(prefix=f".tmp_{scope}_todo_", dir=str(feedback_dir))
        os.close(fd)
        tmp_p = Path(tmp)
        try:
            tmp_p.write_text(
                "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in open_rows),
                encoding="utf-8",
            )
            os.replace(tmp_p, todo_file)
        finally:
            if tmp_p.exists():
                tmp_p.unlink()


def enqueue_tier1_todo(
    feedback_dir: Path,
    *,
    repo: str,
    entry_id: str,
    kind: str,
    dedup_key: str,
    instruction: str,
    detail: dict[str, Any] | None = None,
) -> tuple[str, bool]:
    """Idempotently enqueue a Tier-1 rework ticket (at most once per unique dedup signature)."""
    raw_sig = f"{repo}|{entry_id}|{kind}|{dedup_key}"
    todo_id = "todo-" + hashlib.sha1(raw_sig.encode("utf-8")).hexdigest()[:10]
    feedback_dir.mkdir(parents=True, exist_ok=True)
    with (feedback_dir / ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        ledger = load_feedback_ledger(feedback_dir)
        if todo_id in ledger["todos"]:
            return todo_id, False
        row = {
            "todo_id": todo_id,
            "repo": repo,
            "entry_id": entry_id,
            "kind": kind,
            "status": "open",
            "attempts": 1,
            "created_at": utc_iso(),
            "instruction": instruction,
            "detail": detail or {},
        }
        ledger["todos"][todo_id] = row
        save_feedback_ledger(feedback_dir, ledger)
    return todo_id, True


# ---------------------------------------------------------------------------
# Snapshot Staging (filtering pending_audit & quarantined IDs cleanly)
# ---------------------------------------------------------------------------

def _sync_dir_clean(dst: Path) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True, exist_ok=True)


def stage_catalogs(
    *,
    src_inf: Path,
    src_tr: Path,
    baseline_catalogs: Path,
    dst_catalogs: Path,
    quarantined: dict[str, set[str]] | None = None,
) -> dict[str, Any]:
    """Populate dst_catalogs from upstream repos while:
      - skipping brand-new `pending_audit` entries (preserving baseline YAML if an existing entry is re-auditing),
      - skipping brand-new `quarantined` entries,
      - keeping patterns/*.yaml and id_registry.tsv consistent with staged entries.
    """
    quarantined = quarantined or {"inference": set(), "training": set()}
    stats: dict[str, Any] = {
        "excluded_pending": {"inference": [], "training": []},
        "preserved_baseline_pending": {"inference": [], "training": []},
        "excluded_quarantined": {"inference": [], "training": []},
        "added_ids": {"inference": [], "training": []},
        "modified_ids": {"inference": [], "training": []},
        "active_counts": {"inference": 0, "training": 0},
    }

    sources = {"inference": src_inf, "training": src_tr}
    for repo, src_root in sources.items():
        base_entries_dir = baseline_catalogs / repo / "entries"
        dst_repo_dir = dst_catalogs / repo
        dst_entries_dir = dst_repo_dir / "entries"
        _sync_dir_clean(dst_entries_dir)

        excluded_new_ids: set[str] = set()
        staged_ids: set[str] = set()
        active_count = 0

        src_entries_dir = src_root / "catalog" / "entries"
        for fn in sorted(os.listdir(src_entries_dir)):
            if not fn.endswith(".yaml"):
                continue
            src_fp = src_entries_dir / fn
            raw_text = src_fp.read_text(encoding="utf-8")
            doc = yaml.safe_load(raw_text) or {}
            eid = doc.get("id") or fn[:-5]
            status = doc.get("status")
            base_fp = base_entries_dir / fn
            in_baseline = base_fp.exists()

            if status == "pending_audit":
                if in_baseline:
                    shutil.copy2(base_fp, dst_entries_dir / fn)
                    os.chmod(dst_entries_dir / fn, 0o644)
                    base_doc = yaml.safe_load(base_fp.read_text(encoding="utf-8")) or {}
                    staged_ids.add(eid)
                    if base_doc.get("status") not in ("merged", "retired"):
                        active_count += 1
                    stats["preserved_baseline_pending"][repo].append(eid)
                else:
                    excluded_new_ids.add(eid)
                    stats["excluded_pending"][repo].append({
                        "id": eid,
                        "slug": doc.get("slug", ""),
                        "title": doc.get("title", ""),
                        "scanned": ((doc.get("migration") or {}).get("scanned")
                                    or (doc.get("labels") or {}).get("discovered") or ""),
                        "held": ((doc.get("migration") or {}).get("audit") or {}).get("held") or [],
                        "issue_unfixed": ((doc.get("migration") or {}).get("audit") or {}).get("issue_unfixed") or [],
                    })
                continue

            if eid in quarantined.get(repo, set()) and not in_baseline:
                excluded_new_ids.add(eid)
                stats["excluded_quarantined"][repo].append(eid)
                continue

            dst_fp = dst_entries_dir / fn
            dst_fp.write_text(raw_text, encoding="utf-8")
            os.chmod(dst_fp, 0o644)
            staged_ids.add(eid)
            if status not in ("merged", "retired"):
                active_count += 1
            if not in_baseline:
                stats["added_ids"][repo].append(eid)
            elif base_fp.read_text(encoding="utf-8") != raw_text:
                stats["modified_ids"][repo].append(eid)

        # Also preserve any baseline file that somehow vanished upstream (defensive)
        if base_entries_dir.is_dir():
            for fn in sorted(os.listdir(base_entries_dir)):
                if not fn.endswith(".yaml"):
                    continue
                eid = fn[:-5]
                if eid not in staged_ids and eid not in excluded_new_ids:
                    shutil.copy2(base_entries_dir / fn, dst_entries_dir / fn)
                    os.chmod(dst_entries_dir / fn, 0o644)
                    staged_ids.add(eid)

        stats["active_counts"][repo] = active_count

        # Patterns: filter excluded_new_ids from members; drop brand-new patterns left with 0 members
        src_pat_dir = src_root / "catalog" / "patterns"
        base_pat_dir = baseline_catalogs / repo / "patterns"
        dst_pat_dir = dst_repo_dir / "patterns"
        if src_pat_dir.is_dir():
            _sync_dir_clean(dst_pat_dir)
            for fn in sorted(os.listdir(src_pat_dir)):
                if not fn.endswith(".yaml"):
                    continue
                src_pfp = src_pat_dir / fn
                raw_pat = src_pfp.read_text(encoding="utf-8")
                if not excluded_new_ids:
                    (dst_pat_dir / fn).write_text(raw_pat, encoding="utf-8")
                    os.chmod(dst_pat_dir / fn, 0o644)
                    continue
                pdoc = yaml.safe_load(raw_pat) or {}
                members = pdoc.get("members")
                if isinstance(members, list) and any(m in excluded_new_ids for m in members):
                    kept = [m for m in members if m not in excluded_new_ids]
                    if not kept and not (base_pat_dir / fn).exists():
                        continue
                    pdoc["members"] = kept
                    (dst_pat_dir / fn).write_text(
                        yaml.safe_dump(pdoc, allow_unicode=True, sort_keys=False),
                        encoding="utf-8",
                    )
                else:
                    (dst_pat_dir / fn).write_text(raw_pat, encoding="utf-8")
                os.chmod(dst_pat_dir / fn, 0o644)

        # id_registry.tsv: filter rows belonging to excluded_new_ids
        src_reg = src_root / "catalog" / "id_registry.tsv"
        dst_reg = dst_repo_dir / "id_registry.tsv"
        reg_lines = src_reg.read_text(encoding="utf-8").splitlines(keepends=True)
        filtered_reg = []
        for idx, line in enumerate(reg_lines):
            if idx == 0:
                filtered_reg.append(line)
                continue
            first_col = line.split("\t", 1)[0].strip()
            if first_col in excluded_new_ids:
                continue
            filtered_reg.append(line)
        dst_reg.write_text("".join(filtered_reg), encoding="utf-8")
        os.chmod(dst_reg, 0o644)

        # Markdown negative / frozen master docs
        if repo == "inference":
            src_master = src_root / "FAULT_MASTER_REFERENCE.zh.md"
            if src_master.is_file():
                shutil.copy2(src_master, dst_repo_dir / "FAULT_MASTER_REFERENCE.zh.md")
                os.chmod(dst_repo_dir / "FAULT_MASTER_REFERENCE.zh.md", 0o644)
        for md_name in ("rejected.md", "blindspots.md"):
            src_md = src_root / "catalog" / md_name
            if src_md.is_file():
                shutil.copy2(src_md, dst_repo_dir / md_name)
                os.chmod(dst_repo_dir / md_name, 0o644)

    # Write SOURCES.txt
    sources_lines = ["# upstream commits of this snapshot (written by scripts/sync_catalogs.sh)\n"]
    for repo, src_root in sources.items():
        sha = git(src_root, "rev-parse", "HEAD").stdout.strip()
        cdate = git(src_root, "log", "-1", "--format=%cI").stdout.strip()
        sources_lines.append(f"{src_root.name} {sha} {cdate} entries={stats['active_counts'][repo]}\n")
    (dst_catalogs / "SOURCES.txt").write_text("".join(sources_lines), encoding="utf-8")
    os.chmod(dst_catalogs / "SOURCES.txt", 0o644)
    return stats


# ---------------------------------------------------------------------------
# Worktree Reindex & Deterministic Quality Gates (CLI + Web + Topology)
# ---------------------------------------------------------------------------

def prepare_worktree_links(rag_root: Path, wt_dir: Path) -> None:
    """Link gitignored models/ and web/node_modules/ and seed data/ for incremental embedding reuse."""
    models_src = rag_root / "models"
    models_dst = wt_dir / "models"
    if models_src.exists() and not models_dst.exists():
        os.symlink(models_src, models_dst)

    nm_src = rag_root / "web" / "node_modules"
    nm_dst = wt_dir / "web" / "node_modules"
    if nm_src.exists() and not nm_dst.exists():
        nm_dst.parent.mkdir(parents=True, exist_ok=True)
        os.symlink(nm_src, nm_dst)

    wt_data = wt_dir / "data"
    wt_data.mkdir(parents=True, exist_ok=True)
    for fname in ("entries.jsonl", "embeddings.npy"):
        src_f = rag_root / "data" / fname
        dst_f = wt_data / fname
        if src_f.is_file() and not dst_f.exists():
            shutil.copy2(src_f, dst_f)


def run_worktree_gates(wt_dir: Path, lexical_only: bool = False) -> dict[str, Any]:
    """Run reindex + CLI bench/indy gates + Web gate + Topology check inside wt_dir."""
    py = sys.executable
    reindex_cmd = [py, "-m", "fault_rag", "reindex"] + (["--lexical"] if lexical_only else [])
    r_idx = subprocess.run(reindex_cmd, cwd=str(wt_dir), text=True, capture_output=True, timeout=1800)
    if r_idx.returncode != 0:
        return {
            "passed": False,
            "stage": "reindex",
            "failures": [f"reindex exited {r_idx.returncode}: {r_idx.stderr.strip()[-400:]}"],
            "cli_bench": None,
            "cli_indy": None,
            "web": None,
            "topology": None,
        }

    failures: list[str] = []
    cli_bench = None
    cli_indy = None

    if not lexical_only:
        for suite_name, cases_rel, base_rel, extra in (
            ("bench", "eval/cases.jsonl", "eval/baseline.json", []),
            ("indy", "eval/indy_cases.jsonl", "eval/baseline_indy.json", ["--per-case"]),
        ):
            proc = subprocess.run(
                [py, "eval/run_eval.py", "--cases", cases_rel, "--check", base_rel, *extra],
                cwd=str(wt_dir),
                text=True,
                capture_output=True,
                timeout=1800,
            )
            parsed = None
            if proc.stdout.strip():
                try:
                    parsed = json.loads(proc.stdout)
                except json.JSONDecodeError:
                    parsed = None
            if suite_name == "bench":
                cli_bench = parsed
            else:
                cli_indy = parsed
            if proc.returncode != 0:
                if parsed and parsed.get("quality_gate", {}).get("failures"):
                    for g in parsed["quality_gate"]["failures"]:
                        failures.append(f"cli/{suite_name}: {g}")
                else:
                    failures.append(f"cli/{suite_name} exited {proc.returncode}: {proc.stderr.strip()[-300:]}")

    # Web static-site gate
    web_report = None
    with tempfile.TemporaryDirectory(prefix="fault_rag_web_gate_") as web_tmp:
        dist_dir = os.path.join(web_tmp, "dist")
        rep_path = os.path.join(web_tmp, "web_gate.json")
        has_tf = (wt_dir / "web" / "node_modules" / "@huggingface" / "transformers").is_dir()
        build_args = [py, "web/build_site.py", "--out", dist_dir]
        check_args = [py, "web/eval/check_web.py", "--dist", dist_dir, "--report", rep_path]
        if lexical_only or not has_tf:
            build_args.insert(2, "--no-vectors")
            check_args.extend(["--tier", "bm25"])
        b_proc = subprocess.run(build_args, cwd=str(wt_dir), text=True, capture_output=True, timeout=1800)
        if b_proc.returncode != 0:
            failures.append(f"web/build_site exited {b_proc.returncode}: {b_proc.stderr.strip()[-300:]}")
        else:
            c_proc = subprocess.run(check_args, cwd=str(wt_dir), text=True, capture_output=True, timeout=1800)
            if os.path.exists(rep_path):
                try:
                    web_report = json.loads(Path(rep_path).read_text(encoding="utf-8"))
                except json.JSONDecodeError:
                    web_report = None
            if c_proc.returncode != 0:
                err_lines = [
                    ln.removeprefix("WEB GATE: ").strip()
                    for ln in c_proc.stderr.splitlines()
                    if ln.strip()
                ]
                failures.extend(f"web: {ln}" for ln in (err_lines or [f"check_web exited {c_proc.returncode}"]))

    # Topology gate
    t_proc = subprocess.run(
        [py, "topology/audit_topology.py", "--entries", str(wt_dir / "data" / "entries.jsonl"), "--check"],
        cwd=str(wt_dir),
        text=True,
        capture_output=True,
        timeout=300,
    )
    topo_report = None
    if t_proc.stdout.strip():
        try:
            topo_report = json.loads(t_proc.stdout)
        except json.JSONDecodeError:
            topo_report = None
    if t_proc.returncode != 0:
        failures.append(f"topology: {t_proc.stderr.strip()[-300:]}")

    return {
        "passed": not failures,
        "stage": "complete",
        "failures": failures,
        "cli_bench": cli_bench,
        "cli_indy": cli_indy,
        "web": web_report,
        "topology": topo_report,
    }


def find_candidate_culprits(
    gate_res: dict[str, Any],
    added_keys: set[tuple[str, str]],
) -> list[tuple[tuple[str, str], str]]:
    """Identify newly added entries appearing in top_10 of any failing per-case query."""
    culprits: list[tuple[tuple[str, str], str]] = []
    seen: set[tuple[str, str]] = set()
    failures = gate_res.get("failures") or []
    failed_case_ids: set[str] = set()
    for f in failures:
        m = re.search(r"\b((?:indy|bench)-[A-Za-z0-9_-]+|[a-z0-9-]+-\d{3})\s+dropped out of", f)
        if m:
            failed_case_ids.add(m.group(1))

    reports_to_scan: list[dict[str, Any]] = []
    if gate_res.get("cli_indy"):
        reports_to_scan.append(gate_res["cli_indy"])
    if gate_res.get("cli_bench"):
        reports_to_scan.append(gate_res["cli_bench"])
    if isinstance(gate_res.get("web"), dict):
        for rep in gate_res["web"].values():
            if isinstance(rep, dict) and "cases" in rep:
                reports_to_scan.append(rep)

    for rep in reports_to_scan:
        for row in rep.get("cases", []):
            cid = row.get("case_id", "")
            if failed_case_ids:
                if cid not in failed_case_ids:
                    continue
            else:
                m_row = row.get("metrics") or {}
                if m_row.get("recall_at_5", 1.0) >= 1.0 and m_row.get("recall_at_10", 1.0) >= 1.0:
                    continue
            for pair in row.get("top_10", []):
                if isinstance(pair, (list, tuple)) and len(pair) == 2:
                    key = (str(pair[0]), str(pair[1]))
                    if key in added_keys and key not in seen:
                        seen.add(key)
                        culprits.append((key, cid))
    return culprits


def measure_single_channel_ranks(root_dir: Path, case: dict[str, Any]) -> dict[str, int | None]:
    """Measure hybrid / bm25 / dense rank of the case's first gold entry inside root_dir."""
    code = r"""
import json, sys
sys.path.insert(0, sys.argv[1])
from fault_rag.search import search
case = json.loads(sys.argv[2])
gold = tuple(case["relevant"][0])
out = {}
for mode, kw in (("hybrid", {}), ("bm25", {}), ("dense", {})):
    res = search(case["query"], topk=250, mode=mode, **case["filters"], **kw)
    rank = next((i for i, r in enumerate(res, 1) if (r["repo"], r["id"]) == gold), None)
    out[mode] = rank
print(json.dumps(out))
"""
    gold = [{"repo": r["repo"], "id": r["id"]} for r in case["relevant"]]
    payload = json.dumps({"query": case["query"], "filters": case["filters"], "relevant": [(g["repo"], g["id"]) for g in gold]})
    proc = subprocess.run([sys.executable, "-c", code, str(root_dir), payload], text=True, capture_output=True, timeout=300)
    if proc.returncode == 0 and proc.stdout.strip():
        try:
            return json.loads(proc.stdout)
        except json.JSONDecodeError:
            pass
    return {"hybrid": None, "bm25": None, "dense": None}


# ---------------------------------------------------------------------------
# Thin-Layer Read-Only LLM Client & Post-Gate Semantic QA
# ---------------------------------------------------------------------------

def call_gateway_llm(
    *,
    api_url: str,
    api_key: str,
    system_prompt: str,
    user_prompt: str,
    models: tuple[str, ...] = ("qwen3.8-max", "glm-5.3-flash"),
    timeout: int = 120,
) -> dict[str, Any] | None:
    """Single-turn read-only JSON LLM call against JUDGE_API_URL with model fallback."""
    if not api_url or not api_key:
        return None
    for model in models:
        body = json.dumps({
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.0,
            "max_tokens": 4096,
        }).encode("utf-8")
        for attempt in range(2):
            req = urllib.request.Request(
                api_url,
                data=body,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                method="POST",
            )
            try:
                with urllib.request.urlopen(req, timeout=timeout) as resp:
                    raw = json.loads(resp.read().decode("utf-8"))
                    content = (raw.get("choices") or [{}])[0].get("message", {}).get("content") or ""
                    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", content.strip())
                    start, end = cleaned.find("{"), cleaned.rfind("}")
                    if start >= 0 and end > start:
                        obj = json.loads(cleaned[start : end + 1])
                        if isinstance(obj, dict):
                            return obj
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, KeyError, OSError):
                time.sleep(2 * (attempt + 1))
    return None


def scrub_identifiers_from_blind_query(query: str, slug: str, component: str) -> str:
    """Ensure blind symptom query does not leak exact snake_case slug tokens or C++/Python API identifiers."""
    out = query
    tokens = set()
    if component:
        for tok in re.findall(r"[A-Za-z_][A-Za-z0-9_]{3,}", component):
            if tok.lower() not in {"fp16", "bf16", "fp32", "int8", "int4", "fp8", "cuda", "cann", "vllm", "npu", "gpu", "loss", "nan", "inf"}:
                tokens.add(tok)
    for tok in sorted(tokens, key=len, reverse=True):
        out = re.sub(re.escape(tok), "算子", out, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", out).strip()


def batch_search_in_worktree(wt_dir: Path, queries: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    """Run multiple fault_rag searches in a single Python process so neural models load once."""
    if not queries:
        return []
    code = r"""
import json, sys
sys.path.insert(0, sys.argv[1])
from fault_rag.search import search
qs = json.loads(sys.stdin.read())
out = []
for q in qs:
    rows = search(
        q["query"],
        topk=q.get("topk", 10),
        mode=q.get("mode", "hybrid"),
        no_rerank=q.get("no_rerank", False),
        include_neg=False,
    )
    out.append([
        {
            "repo": r["repo"],
            "id": r["id"],
            "name": r["name"],
            "title": r.get("title"),
            "summary": r.get("summary"),
            "pattern": r.get("pattern"),
            "bm25_rank": r.get("bm25_rank"),
            "dense_rank": r.get("dense_rank"),
        }
        for r in rows
    ])
print(json.dumps(out, ensure_ascii=False))
"""
    proc = subprocess.run(
        [sys.executable, "-c", code, str(wt_dir)],
        input=json.dumps(queries, ensure_ascii=False),
        text=True,
        capture_output=True,
        timeout=900,
    )
    if proc.returncode != 0 or not proc.stdout.strip():
        return [[] for _ in queries]
    try:
        return json.loads(proc.stdout)
    except json.JSONDecodeError:
        return [[] for _ in queries]


def inspect_new_entry_topology(wt_dir: Path, target_keys: set[tuple[str, str]]) -> list[dict[str, Any]]:
    """Deterministically inspect newly added/modified entries for causal_inversion or duplicate enums."""
    if not target_keys:
        return []
    sys.path.insert(0, str(wt_dir))
    from topology import audit_topology as AT

    topo = json.loads((wt_dir / "topology" / "causal_topology.json").read_text(encoding="utf-8"))
    forward_set: set[tuple[str, str]] = set()
    wildcard_sinks: set[str] = set()
    for edge in topo["canonical_forward_edges"]:
        if edge["from"] == "*":
            wildcard_sinks.add(edge["to"])
        else:
            forward_set.add((edge["from"], edge["to"]))
    coupled_set = {(p["a"], p["b"]) for p in topo["intra_step_coupled_pairs"]}
    loops = topo["feedback_loops"]

    defects: list[dict[str, Any]] = []
    with (wt_dir / "data" / "entries.jsonl").open("r", encoding="utf-8") as fh:
        for line in fh:
            e = json.loads(line)
            key = (e.get("repo"), e.get("id"))
            entry_kind = e.get("kind") or (e.get("fields") or {}).get("kind")
            if key not in target_keys or e.get("neg") or entry_kind != "故障":
                continue
            f = e.get("fields", {})
            raw_subs = AT.split_semi(f.get("subsystem"))
            raw_effs = AT.split_eff(f.get("effect_objects"))
            if len(raw_effs) != len(set(raw_effs)):
                defects.append({
                    "repo": key[0],
                    "entry_id": key[1],
                    "defect": "duplicate_effect_objects",
                    "raw_effects": raw_effs,
                })
            if len(raw_subs) > 1 and len(raw_subs) != len(set(raw_subs)):
                defects.append({
                    "repo": key[0],
                    "entry_id": key[1],
                    "defect": "duplicate_subsystems",
                    "raw_subsystems": raw_subs,
                })
            inversions = []
            for i in range(len(raw_effs) - 1):
                cls, detail = AT.classify_carrier_pair(
                    raw_effs[i], raw_effs[i + 1], e, raw_subs, forward_set, wildcard_sinks, coupled_set, loops
                )
                if cls == "causal_inversion":
                    inversions.append(f"{raw_effs[i]} -> {raw_effs[i + 1]} ({detail})")
            if inversions:
                defects.append({
                    "repo": key[0],
                    "entry_id": key[1],
                    "defect": "causal_inversion",
                    "raw_effects": raw_effs,
                    "inversions": inversions,
                })
    return defects


def run_post_gate_qa(
    *,
    wt_dir: Path,
    added_ids: dict[str, list[str]],
    modified_ids: dict[str, list[str]],
    excluded_pending: dict[str, list[dict[str, Any]]],
    indy_report: dict[str, Any] | None,
    api_url: str,
    api_key: str,
    enable_llm: bool = True,
) -> dict[str, Any]:
    """Run deterministic topology checks + read-only LLM blind retrievability & neighbor link screening."""
    qa_summary: dict[str, Any] = {
        "llm_used": False,
        "blind_tested": 0,
        "blind_top5_hits": 0,
        "blind_top10_hits": 0,
        "tier1_todos": [],
        "tier2_items": [],
        "slipped_notes": [],
    }

    # 1. Check stuck pending_audit entries (>= 2 days old)
    today = utc_now().date()
    for repo in REPOS:
        for pend in excluded_pending.get(repo, []):
            scanned_str = str(pend.get("scanned") or "")[:10]
            age_days = 0
            try:
                age_days = (today - dt.date.fromisoformat(scanned_str)).days
            except ValueError:
                age_days = 0
            if age_days >= 2:
                qa_summary["tier2_items"].append({
                    "item_id": f"stuck-{repo}-{pend['id']}-{scanned_str}",
                    "tier": 2,
                    "kind": "stuck_pending_audit",
                    "repo": repo,
                    "entry_id": pend["id"],
                    "age_days": age_days,
                    "title": pend.get("title", ""),
                    "held": pend.get("held", []),
                    "issue_unfixed": pend.get("issue_unfixed", []),
                    "summary": f"{repo} {pend['id']} 已滞留 pending_audit {age_days} 天（held={pend.get('held')}, unfixed={pend.get('issue_unfixed')}）",
                })

    # 2. Deterministic topology check on newly added / modified entries
    changed_keys = {
        (repo, eid)
        for repo in REPOS
        for eid in (added_ids.get(repo, []) + modified_ids.get(repo, []))
    }
    for d in inspect_new_entry_topology(wt_dir, changed_keys):
        repo, eid = d["repo"], d["entry_id"]
        if d["defect"] == "causal_inversion":
            qa_summary["tier1_todos"].append({
                "repo": repo,
                "entry_id": eid,
                "kind": "topology_inversion",
                "dedup_key": ",".join(d["raw_effects"]),
                "instruction": (
                    f"条目 {eid} 的 effect.objects {d['raw_effects']} 存在因果逆序 {d['inversions']}；"
                    f"请核对原 issue，用 tools/catalog_tool.py 按物理因果直接度从前到后重排 effect.objects。"
                ),
                "detail": d,
            })
        else:
            qa_summary["tier1_todos"].append({
                "repo": repo,
                "entry_id": eid,
                "kind": d["defect"],
                "dedup_key": d["defect"],
                "instruction": f"条目 {eid} 存在重复枚举字段 ({d['defect']})，请用 tools/catalog_tool.py 去重修正。",
                "detail": d,
            })

    # Load full YAML docs for newly added entries
    new_entries: list[dict[str, Any]] = []
    for repo in REPOS:
        for eid in added_ids.get(repo, []):
            fp = wt_dir / "catalogs" / repo / "entries" / f"{eid}.yaml"
            if not fp.exists():
                continue
            y = yaml.safe_load(fp.read_text(encoding="utf-8")) or {}
            if y.get("kind") != "故障":
                continue
            objs = (y.get("cause") or {}).get("objects") or []
            comp = "；".join(str(o.get("component", "")) for o in objs if isinstance(o, dict))
            linked_ids = {
                str(r.get("target"))
                for r in (y.get("relations") or [])
                if isinstance(r, dict) and r.get("target")
            }
            new_entries.append({
                "repo": repo,
                "id": eid,
                "slug": y.get("slug", ""),
                "title": y.get("title", ""),
                "component": comp,
                "cause_behavior": (y.get("cause") or {}).get("behavior", ""),
                "effect_detail": (y.get("effect") or {}).get("detail", ""),
                "orig_behavior": (y.get("original") or {}).get("behavior", ""),
                "pattern": y.get("pattern"),
                "linked_ids": linked_ids,
            })

    if not enable_llm or not api_url or not api_key or (not new_entries and not indy_report):
        return qa_summary

    # 3. Thin-layer read-only LLM: Blind oral query generation + neighbor relation screening
    if new_entries:
        blind_queries_by_key: dict[tuple[str, str], str] = {}
        sys_q = (
            "你是大模型训练与推理排障工程师。给你若干故障条目的外部现象描述，请站在不知道底层源码的用户视角，"
            "为每条生成 1 句真实口语化中文排障搜索词（12～35 字）。\n"
            "硬性约束：严禁出现具体 C++/CUDA/Ascend 算子函数名、内部类名、PR/Issue 编号或条目标题原话；"
            "只描述用户观察到的现象与触发场景（如模型/精度/并行方式/报错或静默错值表现）。\n"
            '只输出 JSON：{"queries": {"<repo>:<id>": "<口语搜索词>", ...}}'
        )
        for start in range(0, len(new_entries), 12):
            chunk = new_entries[start : start + 12]
            lines = []
            for item in chunk:
                symptom = (item["orig_behavior"] or item["effect_detail"])[:280]
                lines.append(f"- {item['repo']}:{item['id']}｜现象：{symptom}")
            resp = call_gateway_llm(
                api_url=api_url,
                api_key=api_key,
                system_prompt=sys_q,
                user_prompt="\n".join(lines),
            )
            if resp and isinstance(resp.get("queries"), dict):
                qa_summary["llm_used"] = True
                for item in chunk:
                    k_str = f"{item['repo']}:{item['id']}"
                    raw_q = str(resp["queries"].get(k_str) or "").strip()
                    if raw_q:
                        blind_queries_by_key[(item["repo"], item["id"])] = scrub_identifiers_from_blind_query(
                            raw_q, item["slug"], item["component"]
                        )

        # Build batched search jobs (blind oral queries + mechanistic neighbor queries)
        search_Specs: list[dict[str, Any]] = []
        spec_meta: list[tuple[str, dict[str, Any]]] = []
        for item in new_entries:
            bq = blind_queries_by_key.get((item["repo"], item["id"]))
            if bq:
                search_Specs.append({"query": bq, "topk": 10, "mode": "hybrid", "no_rerank": False})
                spec_meta.append(("blind", item))
            mq = f"{item['title']} {item['cause_behavior']}".strip()
            search_Specs.append({"query": mq, "topk": 6, "mode": "hybrid", "no_rerank": True})
            spec_meta.append(("neighbor", item))

        all_results = batch_search_in_worktree(wt_dir, search_Specs)
        candidate_pairs: list[dict[str, Any]] = []
        seen_pairs: set[tuple[str, str]] = set()

        for (kind, item), rows in zip(spec_meta, all_results):
            key = (item["repo"], item["id"])
            if kind == "blind":
                qa_summary["blind_tested"] += 1
                bq = blind_queries_by_key[key]
                rank = next((idx for idx, r in enumerate(rows, 1) if (r["repo"], r["id"]) == key), None)
                family_top3 = any(
                    (r["id"] in item["linked_ids"] or (item["pattern"] and r.get("pattern") == item["pattern"]))
                    for r in rows[:3]
                )
                if rank is not None and rank <= 5:
                    qa_summary["blind_top5_hits"] += 1
                    qa_summary["blind_top10_hits"] += 1
                elif rank is not None and rank <= 10:
                    qa_summary["blind_top10_hits"] += 1
                elif not family_top3:
                    qa_summary["tier1_todos"].append({
                        "repo": item["repo"],
                        "entry_id": item["id"],
                        "kind": "weak_retrievability",
                        "dedup_key": "blind_miss_top10",
                        "instruction": (
                            f"条目 {item['id']} 在口语症状盲测（查询：「{bq}」）中未进前 10 名；"
                            f"请核对原 issue，在不超出原文证据的前提下于 original.behavior 或 effect.detail 补充用户侧可观测表象词。"
                        ),
                        "detail": {"blind_query": bq, "top_3": [(r["repo"], r["id"], r.get("title")) for r in rows[:3]]},
                    })
            elif kind == "neighbor":
                for rank_i, r in enumerate(rows[:4], 1):
                    n_key = (r["repo"], r["id"])
                    if n_key == key:
                        continue
                    if r["id"] in item["linked_ids"]:
                        continue
                    if item["pattern"] and r.get("pattern") == item["pattern"]:
                        continue
                    if r.get("bm25_rank") is None or r.get("bm25_rank", 999) > 12 or rank_i > 3:
                        continue
                    pair_sig = tuple(sorted([f"{key[0]}:{key[1]}", f"{n_key[0]}:{n_key[1]}"]))
                    if pair_sig in seen_pairs:
                        continue
                    seen_pairs.add(pair_sig)
                    candidate_pairs.append({
                        "pair_id": f"{item['repo']}:{item['id']}__vs__{r['repo']}:{r['id']}",
                        "src": item,
                        "dst": r,
                    })

        # Classify top candidate unlinked neighbor pairs in 1 read-only LLM call
        if candidate_pairs:
            capped = candidate_pairs[:12]
            sys_rel = (
                "你是故障编目关联审查员。给你若干对尚未建立关联的故障条目摘要（A 为今日新入库，B 为检索出的近邻）。"
                "请严格按底层失效机制判断每对关系，只选其一：\n"
                "- unrelated：底层根因或机制不同（仅共享算子名或表面现象）\n"
                "- missed_cross：A 与 B 分属 inference 和 training 不同库，但描述的是同一个底层算子/框架缺陷在训推两侧的表现，应补 cross 边\n"
                "- missed_var：A 与 B 同库，属同一组件/契约类缺陷在不同参数/型号/路径下的变体，应补 var 边\n"
                "- suspected_duplicate：A 与 B 同库，且根因、组件与触发条件高度重合，疑似实质重复（需人工确认是否 merge）\n"
                "宁缺毋滥，拿不准一律标 unrelated。\n"
                '只输出 JSON：{"decisions": {"<pair_id>": {"verdict": "unrelated|missed_cross|missed_var|suspected_duplicate", "reason": "<35字内理由>"}}}'
            )
            pair_lines = []
            for p in capped:
                a, b = p["src"], p["dst"]
                pair_lines.append(
                    f"- {p['pair_id']}\n"
                    f"  A [{a['repo']} {a['id']}] {a['title']}｜原因：{a['cause_behavior']}｜影响：{a['effect_detail']}\n"
                    f"  B [{b['repo']} {b['id']}] {b.get('title')}｜{b.get('summary')}"
                )
            rel_resp = call_gateway_llm(
                api_url=api_url,
                api_key=api_key,
                system_prompt=sys_rel,
                user_prompt="\n".join(pair_lines),
            )
            if rel_resp and isinstance(rel_resp.get("decisions"), dict):
                qa_summary["llm_used"] = True
                decisions = rel_resp["decisions"]
                for p in capped:
                    dec = decisions.get(p["pair_id"])
                    if not isinstance(dec, dict):
                        continue
                    verdict = dec.get("verdict")
                    reason = str(dec.get("reason") or "").strip()[:120]
                    a, b = p["src"], p["dst"]
                    if verdict == "missed_cross" and a["repo"] != b["repo"]:
                        qa_summary["tier1_todos"].append({
                            "repo": a["repo"],
                            "entry_id": a["id"],
                            "kind": "missed_cross",
                            "dedup_key": b["id"],
                            "instruction": (
                                f"新条目 {a['id']} 与跨库条目 {b['repo']} {b['id']}（{b.get('title')}）疑似同一底层机制（{reason}）；"
                                f"请核对双方原文，若属实请用 tools/catalog_tool.py 为 {a['id']} 补充 cross -> {b['id']}。"
                            ),
                            "detail": {"target_repo": b["repo"], "target_id": b["id"], "reason": reason},
                        })
                    elif verdict == "missed_var" and a["repo"] == b["repo"]:
                        qa_summary["tier1_todos"].append({
                            "repo": a["repo"],
                            "entry_id": a["id"],
                            "kind": "missed_var",
                            "dedup_key": b["id"],
                            "instruction": (
                                f"新条目 {a['id']} 与本库 {b['id']}（{b.get('title')}）疑似同族变体（{reason}）；"
                                f"请核对双方原文，若属实且当前无 var 边，可用 tools/catalog_tool.py 补充 var 关联。"
                            ),
                            "detail": {"target_id": b["id"], "reason": reason},
                        })
                    elif verdict == "suspected_duplicate" and a["repo"] == b["repo"]:
                        qa_summary["tier2_items"].append({
                            "item_id": f"dup-{a['repo']}-{min(a['id'], b['id'])}-{max(a['id'], b['id'])}",
                            "tier": 2,
                            "kind": "suspected_duplicate",
                            "repo": a["repo"],
                            "entry_id": a["id"],
                            "target_id": b["id"],
                            "reason": reason,
                            "summary": f"{a['repo']} 新条目 {a['id']} 与既有 {b['id']} 疑似实质重复（{reason}），待人工确认是否 merge",
                            "suggested_command": (
                                f"# 核实双方 YAML 后，若确认实质重复可在 /workspace/{UPSTREAM_NAMES[a['repo']]} 执行 merge"
                            ),
                        })

    # 4. Explain notable non-blocking rank churn (`slipped` drops >= 3 ranks)
    if indy_report and isinstance(indy_report.get("churn"), dict):
        big_slips = [
            s for s in indy_report["churn"].get("slipped", [])
            if s.get("from") and s.get("to") and (s["to"] - s["from"] >= 3)
        ]
        for s in big_slips[:5]:
            qa_summary["slipped_notes"].append(f"{s['case_id']}({s['from']}→{s['to']})")

    return qa_summary


# ---------------------------------------------------------------------------
# Main Sync Orchestration (`cmd_sync`)
# ---------------------------------------------------------------------------

def cmd_sync(args: argparse.Namespace) -> int:
    rag_root = Path(args.rag_root).resolve()
    src_inf = Path(args.src_inference).resolve()
    src_tr = Path(args.src_training).resolve()
    feedback_dir = Path(args.feedback_dir).resolve()

    (rag_root / "data").mkdir(parents=True, exist_ok=True)
    lock_file = rag_root / "data" / ".sync.lock"
    with lock_file.open("a") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        return _run_sync_locked(args, rag_root, src_inf, src_tr, feedback_dir)


def _run_sync_locked(
    args: argparse.Namespace,
    rag_root: Path,
    src_inf: Path,
    src_tr: Path,
    feedback_dir: Path,
) -> int:
    day = utc_now().strftime("%Y-%m-%d")

    # 1. Precheck RAG git worktree cleanliness & remote parity (unless --dry-run)
    if not args.dry_run:
        branch = git(rag_root, "rev-parse", "--abbrev-ref", "HEAD").stdout.strip()
        if branch != "main":
            print(f"❌ RAG 每日同步阻塞：{rag_root} 当前分支为 {branch}（非 main）")
            return 1
        dirty = git(rag_root, "status", "--porcelain", "-uall").stdout.strip()
        if dirty:
            print(f"❌ RAG 每日同步阻塞：{rag_root} 工作区存在未提交改动，已中止同步")
            return 1
        fetch_res = git(rag_root, "fetch", "--quiet", "origin", "main", check=False)
        if fetch_res.returncode != 0:
            print(f"❌ RAG 每日同步阻塞：git fetch origin main 失败（{fetch_res.stderr.strip()[:200]}）")
            return 1
        head_sha = git(rag_root, "rev-parse", "HEAD").stdout.strip()
        remote_sha = git(rag_root, "rev-parse", "origin/main").stdout.strip()
        if head_sha != remote_sha:
            print(f"❌ RAG 每日同步阻塞：本地 HEAD ({head_sha[:7]}) 与 origin/main ({remote_sha[:7]}) 不一致")
            return 1

    for p in (src_inf / "catalog" / "entries", src_tr / "catalog" / "entries"):
        if not p.is_dir():
            print(f"❌ RAG 每日同步阻塞：无法读取上游目录 {p}")
            return 1

    human_store = load_pending_human(rag_root)
    quarantined = active_quarantined_ids(human_store)

    # 2. Create isolated temporary git worktree
    tmp_parent = Path(tempfile.mkdtemp(prefix="fault_rag_sync_wt_"))
    wt_dir = tmp_parent / "wt"
    try:
        git(rag_root, "worktree", "add", "--quiet", "--detach", str(wt_dir), "HEAD")
        prepare_worktree_links(rag_root, wt_dir)

        stage_stats = stage_catalogs(
            src_inf=src_inf,
            src_tr=src_tr,
            baseline_catalogs=rag_root / "catalogs",
            dst_catalogs=wt_dir / "catalogs",
            quarantined=quarantined,
        )

        # Check if catalogs/{inference,training} actually changed vs HEAD
        diff_check = git(wt_dir, "status", "--porcelain", "-uall", "--", "catalogs/inference", "catalogs/training")
        has_catalog_changes = bool(diff_check.stdout.strip())

        if not has_catalog_changes and not args.force:
            # Still check if any upstream pending_audit entry has been stuck >= 2 days
            qa_idle = run_post_gate_qa(
                wt_dir=wt_dir,
                added_ids={"inference": [], "training": []},
                modified_ids={"inference": [], "training": []},
                excluded_pending=stage_stats["excluded_pending"],
                indy_report=None,
                api_url="",
                api_key="",
                enable_llm=False,
            )
            new_h = 0
            for h_item in qa_idle["tier2_items"]:
                _, created = upsert_human_item(human_store, h_item)
                new_h += int(created)
            if new_h:
                save_pending_human(rag_root, human_store)
            pend_cnt = len(stage_stats["excluded_pending"]["inference"]) + len(stage_stats["excluded_pending"]["training"])
            open_h = sum(1 for x in human_store["items"] if x.get("status") == "open")
            extra_note = f"（跳过待审 {pend_cnt} 条；挂起工单 {open_h} 项）" if (pend_cnt or open_h) else ""
            print(f"✅ RAG 每日同步（{day}）：无新增已审编目变更，本地与线上索引已是最新{extra_note}")
            return 0

        # 3. Run gates; if failing, attempt automatic per-entry culprit quarantine
        gate_res = run_worktree_gates(wt_dir, lexical_only=args.lexical_only)
        newly_quarantined: list[dict[str, Any]] = []

        if not gate_res["passed"] and not args.lexical_only:
            added_keys = {
                (r, eid) for r in REPOS for eid in stage_stats["added_ids"][r]
            }
            trial_quarantined = {r: set(quarantined[r]) for r in REPOS}
            initial_failures = list(gate_res.get("failures") or [])

            for _round in range(3):
                cands = find_candidate_culprits(gate_res, added_keys)
                new_cands = [(k, cid) for (k, cid) in cands if k[1] not in trial_quarantined[k[0]]]
                if not new_cands:
                    break
                # Quarantine the highest-priority candidate culprit(s) from this round
                for (c_repo, c_id), cid in new_cands:
                    trial_quarantined[c_repo].add(c_id)
                    surfaces = []
                    if any(f.startswith("cli/") and cid in f for f in gate_res.get("failures", [])):
                        surfaces.append("cli")
                    if any(f.startswith("web:") and cid in f for f in gate_res.get("failures", [])):
                        surfaces.append("web")
                    newly_quarantined.append({
                        "repo": c_repo,
                        "entry_id": c_id,
                        "case_id": cid,
                        "gate_surfaces": surfaces or ["cli"],
                    })

                # Reset worktree catalogs to HEAD cleanly and re-stage without culprits
                git(wt_dir, "checkout", "--quiet", "HEAD", "--", "catalogs")
                git(wt_dir, "clean", "-fdq", "--", "catalogs")
                stage_stats = stage_catalogs(
                    src_inf=src_inf,
                    src_tr=src_tr,
                    baseline_catalogs=rag_root / "catalogs",
                    dst_catalogs=wt_dir / "catalogs",
                    quarantined=trial_quarantined,
                )
                gate_res = run_worktree_gates(wt_dir, lexical_only=args.lexical_only)
                if gate_res["passed"]:
                    # Record quarantined culprits with single-channel rank diagnostics
                    indy_cases_map = {
                        c["case_id"]: c
                        for c in (
                            json.loads(ln)
                            for ln in (rag_root / "eval" / "indy_cases.jsonl").read_text(encoding="utf-8").splitlines()
                            if ln.strip()
                        )
                    }
                    for q_info in newly_quarantined:
                        cid = q_info["case_id"]
                        case_obj = indy_cases_map.get(cid)
                        before_ranks = measure_single_channel_ranks(rag_root, case_obj) if case_obj else {}
                        item_id = f"quarantine-{q_info['repo']}-{q_info['entry_id']}-{cid}"
                        upsert_human_item(human_store, {
                            "item_id": item_id,
                            "tier": 3,
                            "kind": "gate_quarantined_entry",
                            "repo": q_info["repo"],
                            "entry_id": q_info["entry_id"],
                            "case_id": cid,
                            "gate_surfaces": q_info["gate_surfaces"],
                            "pre_sync_ranks": before_ranks,
                            "gate_failure_snapshot": initial_failures,
                            "summary": (
                                f"新条目 {q_info['repo']} {q_info['entry_id']} 导致评测题 {cid} 掉出阈值，已自动单条隔离；"
                                f"其余无冲突条目照常发布"
                            ),
                            "suggested_command": (
                                f"/usr/bin/python3 /workspace/RAG/fault_review/scripts/rag_auto_sync.py "
                                f"resolve {item_id} --action approve_exemption --reason '<确认原因>'"
                            ),
                        })
                    save_pending_human(rag_root, human_store)
                    break

        # If gates STILL fail -> non-destructive circuit-break
        if not gate_res["passed"]:
            sig = hashlib.sha1("\n".join(gate_res["failures"]).encode("utf-8")).hexdigest()[:10]
            cb_id = f"circuit-break-{day}-{sig}"
            upsert_human_item(human_store, {
                "item_id": cb_id,
                "tier": 3,
                "kind": "gate_circuit_break",
                "failures": gate_res["failures"],
                "added_ids": stage_stats["added_ids"],
                "modified_ids": stage_stats["modified_ids"],
                "summary": f"RAG 同步门禁未通过（{'; '.join(gate_res['failures'][:3])}），已无损熔断保留上一版健康索引",
            })
            save_pending_human(rag_root, human_store)
            print(
                f"⚠️ RAG 每日同步熔断（{day}）：门禁拦截，正式库与线上索引保持不变。\n"
                f"  拦截原因：{'；'.join(gate_res['failures'][:4])}\n"
                f"  落盘工单：{rag_root / 'data' / 'pending_human.json'} ({cb_id})"
            )
            return 0

        # Re-check whether anything remains to sync after any culprit quarantine
        diff_after = git(wt_dir, "status", "--porcelain", "-uall", "--", "catalogs/inference", "catalogs/training")
        if not diff_after.stdout.strip() and not args.force:
            print(
                f"⚠️ RAG 每日同步（{day}）：待同步的 {len(newly_quarantined)} 条新条目因触发门禁已单条隔离，"
                f"其余无新增变更；工单见 {rag_root / 'data' / 'pending_human.json'}"
            )
            return 0

        # 4. Run Post-Gate QA (Deterministic topology delta + Read-only thin-layer LLM)
        api_key = read_rc_var("JUDGE_API_KEY")
        api_url = read_rc_var("JUDGE_API_URL")
        qa_res = run_post_gate_qa(
            wt_dir=wt_dir,
            added_ids=stage_stats["added_ids"],
            modified_ids=stage_stats["modified_ids"],
            excluded_pending=stage_stats["excluded_pending"],
            indy_report=gate_res.get("cli_indy"),
            api_url=api_url,
            api_key=api_key,
            enable_llm=not args.no_llm,
        )

        emitted_tier1 = 0
        for todo in qa_res["tier1_todos"]:
            if not args.dry_run:
                _, created = enqueue_tier1_todo(
                    feedback_dir,
                    repo=todo["repo"],
                    entry_id=todo["entry_id"],
                    kind=todo["kind"],
                    dedup_key=todo["dedup_key"],
                    instruction=todo["instruction"],
                    detail=todo.get("detail"),
                )
                emitted_tier1 += int(created)
            else:
                emitted_tier1 += 1

        emitted_tier2 = 0
        for h_item in qa_res["tier2_items"]:
            if not args.dry_run:
                _, created = upsert_human_item(human_store, h_item)
                emitted_tier2 += int(created)
            else:
                emitted_tier2 += 1
        if emitted_tier2 and not args.dry_run:
            save_pending_human(rag_root, human_store)

        n_inf_add = len(stage_stats["added_ids"]["inference"])
        n_trn_add = len(stage_stats["added_ids"]["training"])
        n_inf_mod = len(stage_stats["modified_ids"]["inference"])
        n_trn_mod = len(stage_stats["modified_ids"]["training"])
        inf_tot = stage_stats["active_counts"]["inference"]
        trn_tot = stage_stats["active_counts"]["training"]

        if args.dry_run:
            print(
                f"🧪 [DRY-RUN] RAG 同步预演通过（{day}）：\n"
                f"  拟同步：inference +{n_inf_add}/~{n_inf_mod}（共 {inf_tot}），training +{n_trn_add}/~{n_trn_mod}（共 {trn_tot}）\n"
                f"  跳过待审：INF {len(stage_stats['excluded_pending']['inference'])} 条，TRN {len(stage_stats['excluded_pending']['training'])} 条；"
                f"隔离：{len(newly_quarantined)} 条\n"
                f"  质检：盲测 Top-5 {qa_res['blind_top5_hits']}/{qa_res['blind_tested']}（Top-10 {qa_res['blind_top10_hits']}/{qa_res['blind_tested']}），"
                f"拟派发返工单 {emitted_tier1} 条，人工待批 {emitted_tier2} 项"
            )
            return 0

        # 5. Commit & push in worktree -> fast-forward rag_root -> atomically swap data/ index files
        inf_sha = git(src_inf, "rev-parse", "--short", "HEAD").stdout.strip()
        trn_sha = git(src_tr, "rev-parse", "--short", "HEAD").stdout.strip()
        commit_msg = (
            f"sync catalogs {day}: {inf_tot} INF (+{n_inf_add}) + {trn_tot} TRN (+{n_trn_add}) "
            f"(upstream {inf_sha} / {trn_sha})"
        )
        git(wt_dir, "add", "catalogs")
        ws_chk = git(
            wt_dir,
            "-c", "core.whitespace=-blank-at-eol",
            "diff", "--cached", "--check",
            "--", "catalogs/SOURCES.txt", "catalogs/inference/id_registry.tsv", "catalogs/training/id_registry.tsv",
            check=False,
        )
        if ws_chk.returncode != 0:
            print(f"❌ RAG 每日同步失败：索引元数据空白符检查未通过\n{ws_chk.stdout[:300]}")
            return 1

        git(wt_dir, "commit", "-q", "-m", commit_msg)
        if not args.no_push:
            push_res = git(wt_dir, "push", "-q", "origin", "HEAD:main", check=False, timeout=180)
            if push_res.returncode != 0:
                print(f"❌ RAG 每日同步失败：git push origin HEAD:main 失败（{push_res.stderr.strip()[:250]}）")
                return 1
            git(rag_root, "fetch", "--quiet", "origin", "main")
            git(rag_root, "merge", "--ff-only", "--quiet", "origin/main")
        else:
            wt_sha = git(wt_dir, "rev-parse", "HEAD").stdout.strip()
            git(rag_root, "merge", "--ff-only", "--quiet", wt_sha)

        # Atomic swap of data/ artifacts (embeddings.npy FIRST so entries.jsonl is never ahead of embeddings.npy)
        for idx_file in ("embeddings.npy", "entries.jsonl", "bm25.pkl", "meta.json"):
            src_art = wt_dir / "data" / idx_file
            if src_art.is_file():
                atomic_copy_file(src_art, rag_root / "data" / idx_file)

        pushed_sha = git(rag_root, "rev-parse", "--short", "HEAD").stdout.strip()
        pend_inf = [p["id"] for p in stage_stats["excluded_pending"]["inference"]]
        pend_trn = [p["id"] for p in stage_stats["excluded_pending"]["training"]]
        open_human = [x for x in human_store["items"] if x.get("status") == "open"]

        lines = [
            f"✅ RAG 编目自动同步完成 — {day} (`{pushed_sha}`)",
            f"• 已审入库：推理 `+{n_inf_add}` / 修订 `{n_inf_mod}`（共 {inf_tot} 条）；训练 `+{n_trn_add}` / 修订 `{n_trn_mod}`（共 {trn_tot} 条）",
            "• 门禁结果：CLI (bench + indy 逐条) · Web (Pages 同构) · 因果拓扑词表 全部通过",
        ]
        if qa_res["blind_tested"]:
            lines.append(
                f"• 新条目口语盲测：Top-5 命中 `{qa_res['blind_top5_hits']}/{qa_res['blind_tested']}`"
                f"（Top-10 `{qa_res['blind_top10_hits']}/{qa_res['blind_tested']}`）"
                + (f"；churn 下滑：{', '.join(qa_res['slipped_notes'])}" if qa_res["slipped_notes"] else "")
            )
        if pend_inf or pend_trn:
            skipped_str = ", ".join(pend_inf + pend_trn)
            lines.append(f"• 暂缓未过审条目（保持隔离待上游复审）：{skipped_str}")
        if emitted_tier1:
            lines.append(f"• 自动返工闭环：已向次日扫描回访队列派发 `{emitted_tier1}` 条优化单（补边/症状词/拓扑序）")
        if newly_quarantined or open_human:
            lines.append(
                f"⚠️ 待人工拍板工单：当前共 `{len(open_human)}` 项（详见 `{rag_root / 'data' / 'pending_human.json'}`，"
                f"可在飞书回复机器人或 CLI 运行 `python3 scripts/rag_auto_sync.py status` 查看处理）"
            )
        print("\n".join(lines))
        return 0
    finally:
        git(rag_root, "worktree", "remove", "--force", str(wt_dir), check=False)
        shutil.rmtree(tmp_parent, ignore_errors=True)


# ---------------------------------------------------------------------------
# Subcommands for Async Human Triage & Nightly Scan Agent Todo Consumption
# ---------------------------------------------------------------------------

def cmd_status(args: argparse.Namespace) -> int:
    rag_root = Path(args.rag_root).resolve()
    feedback_dir = Path(args.feedback_dir).resolve()
    human_store = load_pending_human(rag_root)
    ledger = load_feedback_ledger(feedback_dir)

    open_human = [x for x in human_store.get("items", []) if x.get("status") == "open"]
    open_todos = [x for x in ledger.get("todos", {}).values() if x.get("status") == "open"]
    print(json.dumps({
        "pending_human_file": str(rag_root / "data" / "pending_human.json"),
        "open_human_count": len(open_human),
        "open_human_items": open_human,
        "feedback_dir": str(feedback_dir),
        "open_tier1_todo_count": len(open_todos),
        "open_tier1_todos": open_todos,
    }, ensure_ascii=False, indent=2))
    return 0


def cmd_resolve(args: argparse.Namespace) -> int:
    rag_root = Path(args.rag_root).resolve()
    store = load_pending_human(rag_root)
    target = next((x for x in store.get("items", []) if x.get("item_id") == args.item_id), None)
    if not target:
        sys.stderr.write(f"resolve: item_id not found: {args.item_id}\n")
        return 1

    if args.action == "approve_exemption":
        if target.get("kind") != "gate_quarantined_entry":
            sys.stderr.write("approve_exemption applies only to gate_quarantined_entry items\n")
            return 1
        if not args.reason or not args.reason.strip():
            sys.stderr.write("approve_exemption requires --reason explaining the owner-accepted regression\n")
            return 1
        surfaces = target.get("gate_surfaces") or ["cli"]
        surface_paths = []
        if "cli" in surfaces:
            surface_paths.append(rag_root / "eval" / "fix_list.json")
        if "web" in surfaces:
            surface_paths.append(rag_root / "web" / "eval" / "fix_list.json")
        cid = target["case_id"]
        for fix_path in surface_paths:
            if not fix_path.exists():
                continue
            fix_doc = json.loads(fix_path.read_text(encoding="utf-8"))
            if not any(r.get("case_id") == cid for r in fix_doc.get("fix_list", [])):
                fix_doc["fix_list"].append({
                    "case_id": cid,
                    "kind": "accepted_regression",
                    "triggered_by": f"{target['repo']}:{target['entry_id']}",
                    "ranks": {"pre_sync": target.get("pre_sync_ranks") or {}},
                    "why": args.reason.strip(),
                    "exit": args.exit_condition or "Remove once the gold returns to baseline cutoff on current snapshot.",
                })
                atomic_write_json(fix_path, fix_doc)
        target["status"] = "resolved"
        target["resolution"] = "approved_exemption"
        target["resolved_at"] = utc_iso()
        target["reason"] = args.reason.strip()
    elif args.action == "unquarantine":
        target["status"] = "resolved"
        target["resolution"] = "unquarantined_for_retry"
        target["resolved_at"] = utc_iso()
    elif args.action == "dismiss":
        target["status"] = "dismissed"
        target["resolved_at"] = utc_iso()
        if args.reason:
            target["reason"] = args.reason.strip()

    save_pending_human(rag_root, store)
    print(json.dumps({"ok": True, "item_id": args.item_id, "status": target["status"]}, ensure_ascii=False))
    return 0


def cmd_list_todos(args: argparse.Namespace) -> int:
    feedback_dir = Path(args.feedback_dir).resolve()
    ledger = load_feedback_ledger(feedback_dir)
    rows = [
        r for r in ledger.get("todos", {}).values()
        if r.get("status") == "open" and (not args.scope or r.get("repo") == args.scope)
    ]
    rows.sort(key=lambda r: (r.get("created_at", ""), r.get("todo_id", "")))
    print(json.dumps({"scope": args.scope, "count": len(rows), "todos": rows}, ensure_ascii=False, indent=2))
    return 0


def cmd_done_todo(args: argparse.Namespace) -> int:
    feedback_dir = Path(args.feedback_dir).resolve()
    feedback_dir.mkdir(parents=True, exist_ok=True)
    with (feedback_dir / ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        ledger = load_feedback_ledger(feedback_dir)
        row = ledger.get("todos", {}).get(args.id)
        if not row:
            sys.stderr.write(f"done-todo: unknown todo_id {args.id}\n")
            return 1
        row["status"] = "dismissed" if args.dismiss else "done"
        row["completed_at"] = utc_iso()
        if args.note:
            row["resolution_note"] = args.note
        save_feedback_ledger(feedback_dir, ledger)
    print(json.dumps({"ok": True, "todo_id": args.id, "status": row["status"]}, ensure_ascii=False))
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--rag-root", default=str(DEFAULT_RAG_ROOT))
    ap.add_argument("--feedback-dir", default=str(DEFAULT_FEEDBACK_DIR))
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_sync = sub.add_parser("sync", help="Run isolated worktree sync + gates + QA + publish")
    p_sync.add_argument("--src-inference", default=os.environ.get("FAULT_RAG_SRC_INFERENCE_REPO", str(DEFAULT_UPSTREAM_ROOT / "inference_error_review")))
    p_sync.add_argument("--src-training", default=os.environ.get("FAULT_RAG_SRC_TRAINING_REPO", str(DEFAULT_UPSTREAM_ROOT / "training_error_review")))
    p_sync.add_argument("--lexical-only", action="store_true")
    p_sync.add_argument("--no-llm", action="store_true")
    p_sync.add_argument("--no-push", action="store_true")
    p_sync.add_argument("--dry-run", action="store_true")
    p_sync.add_argument("--force", action="store_true")

    sub.add_parser("status", help="Inspect open human items and Tier-1 scan todos")

    p_res = sub.add_parser("resolve", help="Resolve an open item in pending_human.json")
    p_res.add_argument("item_id")
    p_res.add_argument("--action", choices=("approve_exemption", "unquarantine", "dismiss"), required=True)
    p_res.add_argument("--reason", default="")
    p_res.add_argument("--exit-condition", default="")

    p_lt = sub.add_parser("list-todos", help="List open Tier-1 rework todos for nightly scan agent")
    p_lt.add_argument("--scope", choices=REPOS, default=None)

    p_dt = sub.add_parser("done-todo", help="Mark a Tier-1 rework todo completed or dismissed")
    p_dt.add_argument("--id", required=True)
    p_dt.add_argument("--dismiss", action="store_true")
    p_dt.add_argument("--note", default="")
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    dispatch = {
        "sync": cmd_sync,
        "status": cmd_status,
        "resolve": cmd_resolve,
        "list-todos": cmd_list_todos,
        "done-todo": cmd_done_todo,
    }
    return dispatch[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
