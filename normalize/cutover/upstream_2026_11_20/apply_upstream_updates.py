#!/usr/bin/env python3
"""Idempotently preview (--dry-run) or apply (--apply) the 2026-11-20 scan/audit
improvements to `/workspace/inference_error_review` and `/workspace/training_error_review`.

Run by the maintainer AFTER `git pull --ff-only` in both upstream repos.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import re
import shutil
import sys

HERE = Path(__file__).resolve().parent
HERMES_SCRIPTS = HERE.parent / "hermes_scripts"
INF_REPO = Path("/workspace/inference_error_review")
TRN_REPO = Path("/workspace/training_error_review")

NEW_NPU_STACKS = [
    "torchair",
    "msprobe",
    "mssanitizer",
    "msmodelslim",
    "mindie-llm",
    "vllm-ascend",
    "sgl-kernel-npu",
    "triton-ascend",
]

TOPOLOGY_HELPER_BLOCK = '''# Causal topology carrier ranks & legal feedback/coupled transitions (synced from fault_review/topology/causal_topology.json)
EFFECT_CARRIER_RANK = {
    "训练样本流": 10, "输入编码": 15, "数值结果": 20, "隐状态": 25, "KV状态": 30,
    "跨请求/跨样本数据": 35, "输出分布": 40, "输出内容": 50, "训练信号": 60, "梯度": 70,
    "优化器状态": 80, "模型权重": 90, "训练轨迹": 95, "可用性/进度": 100, "性能/资源": 100,
    "指标/日志": 110, "测试判定": 115,
}
COUPLED_CARRIER_PAIRS = frozenset({
    ("隐状态", "数值结果"), ("梯度", "数值结果"), ("输入编码", "训练样本流"),
    ("跨请求/跨样本数据", "KV状态"), ("跨请求/跨样本数据", "隐状态"),
    ("跨请求/跨样本数据", "数值结果"), ("可用性/进度", "性能/资源"),
})
LEGAL_FEEDBACK_PAIRS = frozenset({
    *{("模型权重", dst) for dst in ("数值结果", "隐状态", "KV状态", "输出分布", "输出内容", "训练信号", "梯度", "优化器状态")},
    ("KV状态", "数值结果"), ("KV状态", "隐状态"),
    *{("KV状态", dst) for dst in ("训练信号", "梯度", "训练样本流", "训练轨迹")},
    *{("输出分布", dst) for dst in ("训练信号", "梯度", "训练样本流", "训练轨迹", "KV状态", "跨请求/跨样本数据")},
    *{("输出内容", dst) for dst in ("训练信号", "梯度", "训练样本流", "训练轨迹")},
    ("数值结果", "训练样本流"), ("数值结果", "输入编码"),
})
ABORT_RECOVERY_SUBSYSTEMS = frozenset({"调度/服务", "运行时/执行", "集合通信", "重计算/卸载", "KV缓存", "权重/检查点"})


def _is_causal_inversion(e1, e2, subsystems):
    if e1 not in EFFECT_CARRIER_RANK or e2 not in EFFECT_CARRIER_RANK or e1 == e2:
        return False
    if (e1, e2) in COUPLED_CARRIER_PAIRS or (e2, e1) in COUPLED_CARRIER_PAIRS or (e1, e2) in LEGAL_FEEDBACK_PAIRS:
        return False
    if e1 == "可用性/进度" and any(s in ABORT_RECOVERY_SUBSYSTEMS for s in subsystems):
        return False
    return EFFECT_CARRIER_RANK[e1] >= EFFECT_CARRIER_RANK[e2]
'''


def patch_vocab(text: str) -> str:
    if "vllm-ascend" in text:
        return text
    anchor = "  - mindspeed-rl\n"
    if anchor not in text:
        raise RuntimeError("vocab.yaml: expected `  - mindspeed-rl` anchor not found")
    addition = "  # v1.4 新增：NPU 推理/评测/跨平台扫描高频专仓栈\n" + "".join(
        f"  - {s}\n" for s in NEW_NPU_STACKS
    )
    text = text.replace("# 故障模式库受控词表 v1.3", "# 故障模式库受控词表 v1.4", 1)
    return text.replace(anchor, anchor + addition, 1)


def patch_catalog_audit(text: str) -> str:
    text = re.sub(
        r'^API\s*=\s*["\']https?://[^"\']+["\']\s*$',
        'API = os.environ.get("JUDGE_API_URL") or sys.exit("JUDGE_API_URL not set")',
        text,
        count=1,
        flags=re.M,
    )
    text = text.replace(
        'BASE = ("glm-5.3-flash", "deepseek-v4.1-flash", "qwen3.8-flash")',
        'BASE = ("glm-5.3-flash", "doubao-seed-2.1-pro", "qwen3.8-flash")',
    )
    text = text.replace(
        'MODEL_OPTS = {"kimi-k3.1-pro": {"temperature": 1.0}}',
        'MODEL_OPTS = {"kimi-k3.1-pro": {"temperature": 1.0}, "doubao-seed-2.1-pro": {"reasoning_effort": "low"}}',
    )
    return text


def patch_catalog_tool(text: str) -> str:
    if "EFFECT_CARRIER_RANK" not in text:
        anchor = 'NARRATIVE_ORDER = ("mechanism", "source", "behavior", "trigger_surface", "signature", "repro", "discovery", "notes")\n'
        if anchor not in text:
            raise RuntimeError("catalog_tool.py: NARRATIVE_ORDER anchor not found")
        text = text.replace(anchor, anchor + "\n" + TOPOLOGY_HELPER_BLOCK, 1)

    cause_anchor = "        else:\n            for o in objs:\n                enum(o.get(\"subsystem\"), voc[\"subsystem\"], \"cause.objects.subsystem\")"
    cause_repl = (
        "        else:\n"
        "            subs = [o.get(\"subsystem\") for o in objs if isinstance(o, dict) and o.get(\"subsystem\") not in ph]\n"
        "            if len(subs) != len(set(subs)):\n"
        "                (errors if strict_lengths else warnings).append(\n"
        "                    f\"cause.objects 含重复 subsystem {subs}：同一子系统多组件应合写进一个 component\")\n"
        "            for o in objs:\n"
        "                enum(o.get(\"subsystem\"), voc[\"subsystem\"], \"cause.objects.subsystem\")"
    )
    if "cause.objects 含重复 subsystem" not in text:
        if cause_anchor not in text:
            raise RuntimeError("catalog_tool.py: cause.objects anchor not found")
        text = text.replace(cause_anchor, cause_repl, 1)

    eff_anchor = "        else:\n            for o in eobjs:\n                enum(o, voc[\"effect_object\"], \"effect.objects\")"
    eff_repl = (
        "        else:\n"
        "            for o in eobjs:\n"
        "                enum(o, voc[\"effect_object\"], \"effect.objects\")\n"
        "            non_ph_eobjs = [o for o in eobjs if o not in ph]\n"
        "            if len(non_ph_eobjs) != len(set(non_ph_eobjs)):\n"
        "                (errors if strict_lengths else warnings).append(f\"effect.objects 含重复枚举: {eobjs}\")\n"
        "            subs_all = [o.get(\"subsystem\") for o in (objs if isinstance(objs, list) else []) if isinstance(o, dict)]\n"
        "            for a_obj, b_obj in zip(eobjs, eobjs[1:]):\n"
        "                if _is_causal_inversion(a_obj, b_obj, subs_all):\n"
        "                    canonical = sorted(eobjs, key=lambda x: EFFECT_CARRIER_RANK.get(x, 999))\n"
        "                    (errors if strict_lengths else warnings).append(\n"
        "                        f\"effect.objects 存在因果逆序 {a_obj!r}->{b_obj!r}（第一项应写最直接变错的状态载体，建议顺序：{canonical}）\"\n"
        "                    )"
    )
    if "effect.objects 存在因果逆序" not in text:
        if eff_anchor not in text:
            raise RuntimeError("catalog_tool.py: effect.objects anchor not found")
        text = text.replace(eff_anchor, eff_repl, 1)
    return text


def apply_one_file(path: Path, new_text: str, apply: bool) -> bool:
    old_text = path.read_text(encoding="utf-8") if path.exists() else None
    if old_text == new_text:
        print(f"  [unchanged] {path}")
        return False
    print(f"  [{'WRITE' if apply else 'WOULD WRITE'}] {path}")
    if apply:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(new_text, encoding="utf-8")
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    grp = ap.add_mutually_exclusive_group(required=True)
    grp.add_argument("--dry-run", action="store_true", help="Check anchors and preview changes without modifying files")
    grp.add_argument("--apply", action="store_true", help="Write changes to both upstream catalog repos")
    args = ap.parse_args()

    changed = 0
    for repo in (INF_REPO, TRN_REPO):
        if not repo.is_dir():
            sys.exit(f"missing upstream repo: {repo}")
        print(f"=== {repo} ===")
        vocab_p = repo / "catalog" / "vocab.yaml"
        changed += apply_one_file(vocab_p, patch_vocab(vocab_p.read_text(encoding="utf-8")), args.apply)

        audit_p = repo / "tools" / "catalog_audit.py"
        changed += apply_one_file(audit_p, patch_catalog_audit(audit_p.read_text(encoding="utf-8")), args.apply)

        tool_p = repo / "tools" / "catalog_tool.py"
        changed += apply_one_file(tool_p, patch_catalog_tool(tool_p.read_text(encoding="utf-8")), args.apply)

    # Sync prompt docs
    changed += apply_one_file(
        INF_REPO / "daily_scan.prompt.md",
        (HERE / "inference_daily_scan.prompt.md").read_text(encoding="utf-8"),
        args.apply,
    )
    changed += apply_one_file(
        TRN_REPO / "hermes" / "daily_scan.prompt.md",
        (HERE / "training_daily_scan.prompt.md").read_text(encoding="utf-8"),
        args.apply,
    )

    # Sync NPU collector versioned source in inference_error_review/tools/scan/npu/
    npu_dir = INF_REPO / "tools" / "scan" / "npu"
    if npu_dir.is_dir():
        for src_rel, dst_name in (
            ("fault_scan_npu_entry.py", "fault_scan_npu_entry.py"),
            ("test_fault_scan_npu_entry.py", "test_fault_scan_npu_entry.py"),
            ("fault_scan_sources/collect.py", "collect.py"),
        ):
            changed += apply_one_file(
                npu_dir / dst_name,
                (HERMES_SCRIPTS / src_rel).read_text(encoding="utf-8"),
                args.apply,
            )

    print(f"Done ({'applied' if args.apply else 'dry-run'}): {changed} file(s) {'updated' if args.apply else 'to update'}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
