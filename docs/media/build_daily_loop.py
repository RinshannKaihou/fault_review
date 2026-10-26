#!/usr/bin/env python3
"""Build the live-panel config for the fault-catalog daily loop shown in README.

  python3 build_daily_loop.py terminal-dark daily-loop-dark.json
  python3 build_daily_loop.py light-pastel  daily-loop-light.json
Render and encode with render_daily_loop.py (2x pixel density; the skill's render.py alone gives a blurry README image).
Numbers on screen: entry counts from catalogs/SOURCES.txt, daily growth from the two catalog repos git log (2026-10-21..26).
"""
import json, sys

theme = sys.argv[1] if len(sys.argv) > 1 else "terminal-dark"
out = sys.argv[2] if len(sys.argv) > 2 else "config.json"

N = lambda id, row, col, **k: dict(id=id, row=row, col=col, **k)
nodes = [
    # row 0: trigger, legend, status
    N("cron", 0, 0, type="trigger", hue="cyan", icon="timer", label="hermes cron", en="Trigger",
      desc=["推理 23:00 · 训练 00:30", "审计 02:00 UTC"]),
    N("legend", 0, 1, span=2, type="legend",
      items=[{"style": "solid", "text": "前进"}, {"style": "dashed", "text": "回环 · 重试/复议/次日"},
             {"style": "dot", "text": "数据流过"}]),
    N("status", 0, 3, span=2, type="note", label="", desc=[], icon="refresh-cw"),
    # rows 1-2: scan agent
    N("precheck", 1, 0, group="scan", type="diamond", label="Git 干净？", en="precheck"),
    N("search", 1, 1, group="scan", icon="eye", label="搜索", en="Search",
      desc=["GitHub · arXiv · NPU", "近 7 日 + 回访旧源"]),
    N("triage", 1, 2, group="scan", icon="bug", label="查重判定", en="Triage",
      desc=["规范 URL 查重", "沿用 14 日内判定"]),
    N("cross", 1, 3, group="scan", icon="refresh-cw", label="跨库关联", en="Cross",
      desc=["查另一编目", "同机制写 cross 链接"]),
    N("record", 1, 4, group="scan", icon="ticket", label="记录", en="Record",
      desc=["catalog_tool new", "pending_audit"]),
    N("blocked", 2, 0, group="scan", icon="bell", label="停扫报告", en="Blocked",
      desc=["不 pull/reset/强推", "等人工整理"]),
    N("scannote", 2, 1, span=2, group="scan", type="note", icon="info", label="",
      desc=["三条回路：校验不过 → 重交 ≤3 次", "评委不一致 → 六票复议", "上线后 → 次日 cron 再跑一轮"]),
    N("report", 2, 3, group="scan", icon="git-pull-request", label="日报 + 推送", en="Push",
      desc=["commit · push", "远程 SHA 核验"]),
    N("valid", 2, 4, group="scan", type="diamond", label="校验通过？", en="validate"),
    # rows 3-4: audit script
    N("judges", 3, 3, group="audit", icon="bot", label="三评委盲审", en="Judge×3",
      desc=["分类 + 忠实度", "空票重试一次"]),
    N("decided", 3, 2, group="audit", type="diamond", label="全票一致？", en="unanimous?"),
    N("fixer", 3, 1, group="audit", icon="shield-check", label="改写 + 双验", en="Fix",
      desc=["失实文本才改写", "两验证都过才留"]),
    N("apply", 3, 0, group="audit", icon="git-merge", label="定稿合入", en="Merge",
      desc=["active · 模式归族", "validate · push"]),
    N("panel", 4, 2, group="audit", icon="message-square", label="六票复议", en="Panel×6",
      desc=["≥4 有效票 · ≥2/3", "不足则保留原值"]),
    N("growth", 4, 3, span=2, group="audit", type="note", icon="info", label="编目 2187 + 451 条 · 模式 243",
      desc=["日增（10-21~26）：推理 +10~24 · 训练 +7~25"]),
    # row 5: merge into retrieval, human gate
    N("review", 5, 3, group="merge", icon="hand", label="人工审查", en="Review",
      desc=["读审计日报", "确认后才同步"]),
    N("rag", 5, 2, group="merge", icon="refresh-cw", label="同步快照", en="Sync",
      desc=["sync_catalogs.sh", "reindex"]),
    N("gate", 5, 1, group="merge", icon="shield-check", label="检索门禁", en="Eval",
      desc=["36 + 63 条回归", "退化即不发布"]),
    N("release", 5, 0, group="merge", type="output", icon="bot", label="上线检索", en="Release",
      desc=["CLI + Pages", "供 agent 引用"]),
    N("rules", 5, 4, group="merge", type="note", icon="hand", label="人只改规则文件",
      desc=["不改产出"]),
]
E = lambda a, b, **k: dict({"from": a, "to": b}, **k)
edges = [
    E("cron", "precheck", fromSide="bottom", toSide="top"),
    E("precheck", "search", kind="yes", label="Yes"),
    E("precheck", "blocked", kind="no", dashed=False, fromSide="bottom", toSide="top"),
    E("search", "triage"), E("triage", "cross"), E("cross", "record"),
    E("record", "valid", fromSide="bottom", toSide="top"),
    E("valid", "record", kind="no", label="No · 重交", lane="right", fromSide="right", toSide="right"),
    E("valid", "report", kind="yes", label="Yes"),
    E("report", "judges", fromSide="bottom", toSide="top"),
    E("judges", "decided"),
    E("decided", "fixer", kind="yes", label="Yes"),
    E("decided", "panel", kind="no", dashed=True, fromSide="bottom", toSide="top"),
    E("panel", "fixer", fromSide="left", toSide="bottom"),
    E("fixer", "apply"),
    E("apply", "review", fromSide="bottom", toSide="top"),
    E("review", "rag"), E("rag", "gate"), E("gate", "release"),
    E("release", "cron", kind="back", label="次日再来 · next day", lane="left", fromSide="left", toSide="left"),
]
walk = ["cron", "precheck", "search", "triage", "cross", "record", "valid", "record", "valid", "report",
        "judges", "decided", "panel", "fixer", "apply", "review", "rag", "gate", "release", "cron"]

view = {
    "type": "view", "kind": "loop",
    "title": "故障模式库 · 每日自主增长 Loop",
    "subtitle": "推理 / 训练两库 · 每夜扫描 → 记录 → 审计 → 合入 · 人只在发布前拍板",
    "period": 27, "cols": 5, "font": "sans",
    "status": {"node": "status", "round": "每日 · 无人值守", "step": "当前 · {label}"},
    "groups": [
        {"id": "scan", "label": "① 每日扫描 · 搜索 + 记录", "sub": "扫描 agent", "hue": "blue"},
        {"id": "audit", "label": "② 每日审计 · 02:00", "sub": "脚本 · 无 agent", "hue": "amber"},
        {"id": "merge", "label": "③ 合入检索库", "sub": "human gate", "hue": "green"},
    ],
    "nodes": nodes, "edges": edges, "walks": [walk],
    "motion": "ambient+walk", "ambientPeriod": 3.0,
}
cfg = {
    "meta": {"title": "fault-rag daily loop", "lang": "zh"},
    "canvas": {"width": 1280, "height": 960, "duration": 27, "fps": 30},
    "theme": {"preset": theme, "titleFont": "smiley"},
    "titlebar": {"text": "~/fault_review/daily-loop — live-panel"},
    "credit": {"text": "流程：daily_scan.prompt.md · daily_audit.sh · catalog_audit.py · 数字：两编目 git log 与 SOURCES.txt（2026-10-26）· 节奏为示意",
               "c": "muted", "x": 16, "w": 1248, "align": "left"},
    "elements": [view],
}
json.dump(cfg, open(out, "w"), ensure_ascii=False, indent=1)
print(out)
