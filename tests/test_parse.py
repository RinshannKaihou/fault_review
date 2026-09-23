import os
import tempfile
import unittest

from fault_rag import parse


class CategoryParsingTests(unittest.TestCase):
    def test_inference_subsection_does_not_override_id_category(self):
        content = """## 第 9 类 · 位置编码故障
## §3.8 输入层故障（tokenizer / 模板 / RoPE）
### 9.26 `fault_rope`

`model_on` · Cov: `NEW` · 触发 `config` · 置信度 `verified`

- **机制**：RoPE scale 配置错误。
"""
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False) as f:
            f.write(content)
            path = f.name
        try:
            entries = parse.parse_master(path, "inference")
        finally:
            os.unlink(path)

        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0]["category"], "cat9")
        self.assertEqual(
            entries[0]["section"],
            "§3.8 输入层故障（tokenizer / 模板 / RoPE）",
        )

    def test_category_is_derived_from_entry_id_grammar(self):
        cases = {
            ("inference", "9.26"): "cat9",
            ("inference", "K9.14"): "KV9",
            ("training", "RL-ADV.02"): "RL-ADV",
            ("training", "HW.01"): "HW",
        }
        for (repo, entry_id), expected in cases.items():
            with self.subTest(repo=repo, entry_id=entry_id):
                self.assertEqual(
                    parse._category_from_id(entry_id, repo), expected
                )

    def test_unsupported_entry_id_fails_loudly(self):
        with self.assertRaisesRegex(ValueError, "unsupported entry id"):
            parse._category_from_id("bad-id", "inference")


class EntryTypeTests(unittest.TestCase):
    def _parse_one(self, body):
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False) as f:
            f.write(body)
            path = f.name
        try:
            entries = parse.parse_master(path, "inference")
        finally:
            os.unlink(path)
        self.assertEqual(len(entries), 1)
        return entries[0]

    def test_explicit_existing_pointer_is_classified_as_pointer(self):
        entry = self._parse_one("""## 第 10 类 · 模板故障
### 10.4 `chat_template_faults`

`源文档未给出` · Cov: `existing` · 触发 `n/a` · 置信度 `源文档未给出`

- **机制**：这是一条指针行，不是新故障。Cited for completeness; not a new entry.
""")
        self.assertEqual(entry["entry_type"], "pointer")
        self.assertFalse(entry["neg"])
        self.assertIsNone(entry["confidence"])

    def test_regular_catalog_entry_is_classified_as_fault(self):
        entry = self._parse_one("""## 第 9 类 · 位置编码故障
### 9.26 `fault_rope`

`model_on` · Cov: `NEW` · 触发 `config` · 置信度 `verified`

- **机制**：RoPE scale 配置错误。
""")
        self.assertEqual(entry["entry_type"], "fault")

    def test_negative_chunk_is_classified_as_negative(self):
        content = """# Blindspots

## B.1 盲区

没有可分析的前向。
"""
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False) as f:
            f.write(content)
            path = f.name
        try:
            entries = parse.parse_negative_doc(path, "training", "blindspot")
        finally:
            os.unlink(path)
        self.assertTrue(entries)
        self.assertTrue(all(e["entry_type"] == "negative" for e in entries))
        self.assertTrue(all(e["neg"] for e in entries))


class EmbeddingHashTests(unittest.TestCase):
    def test_embedding_hash_changes_when_embedding_text_changes(self):
        old = parse.embedding_hash("fault rope\ncategory §3.8")
        new = parse.embedding_hash("fault rope\ncategory cat9")
        self.assertNotEqual(old, new)

    def test_embedding_hash_is_stable(self):
        text = "fault rope\ncategory cat9"
        self.assertEqual(parse.embedding_hash(text), parse.embedding_hash(text))


class TrainingBoundaryTests(unittest.TestCase):
    def test_fenced_rules_and_headings_stay_in_multiline_field(self):
        for fence in ("```", "~~~"):
            with self.subTest(fence=fence):
                content = f"""# DATA · 数据管线
## DATA
### DATA.01 `data_example`
stage: `pretrain` · Cov: `NEW` · 置信度 `documented`
- **机制**：起因。
{fence}markdown
---
# example heading
{fence}
- **发现来源**：每日扫描
### DATA.02 `next_example`
stage: `pretrain` · Cov: `NEW` · 置信度 `documented`
- **机制**：下一条。
"""
                with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False) as f:
                    f.write(content)
                    path = f.name
                try:
                    entries = parse.parse_master(path, "training")
                finally:
                    os.unlink(path)
                self.assertEqual([e["id"] for e in entries], ["DATA.01", "DATA.02"])
                self.assertEqual(entries[0]["fields"]["mechanism"], f"起因。\n{fence}markdown\n---\n# example heading\n{fence}")
                self.assertEqual(entries[0]["fields"]["discovery"], "每日扫描")
                self.assertIn(f"{fence}markdown\n---\n# example heading\n{fence}", entries[0]["raw"])
                self.assertEqual(entries[0]["line_end"], 10)

    def test_horizontal_rules_inside_entry_keep_continuation_and_next_field(self):
        content = """# DATA · 数据管线
## DATA
### DATA.01 `data_example`
stage: `pretrain` · Cov: `NEW` · 置信度 `documented`
- **机制**：起因。
---
后续正文。
  ---
# 正文中的标题
- **发现来源**：每日扫描
### DATA.02 `next_example`
stage: `pretrain` · Cov: `NEW` · 置信度 `documented`
- **机制**：下一条。
"""
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False) as f:
            f.write(content)
            path = f.name
        try:
            entries = parse.parse_master(path, "training")
        finally:
            os.unlink(path)
        self.assertEqual([e["id"] for e in entries], ["DATA.01", "DATA.02"])
        self.assertEqual(entries[0]["fields"]["mechanism"], "起因。\n---\n后续正文。\n  ---\n# 正文中的标题")
        self.assertEqual(entries[0]["fields"]["discovery"], "每日扫描")
        self.assertIn("---\n后续正文。\n  ---\n# 正文中的标题", entries[0]["raw"])
        self.assertEqual(entries[0]["line_end"], 10)

    def test_unmatched_heading_after_rule_stays_in_entry(self):
        content = """# DATA · 数据管线
## DATA
### DATA.01 `data_example`
stage: `pretrain` · Cov: `NEW` · 置信度 `documented`
- **机制**：起因。
---

# 使用注意
继续说明。
- **发现来源**：每日扫描
### DATA.02 `next_example`
stage: `pretrain` · Cov: `NEW` · 置信度 `documented`
- **机制**：下一条。
"""
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False) as f:
            f.write(content)
            path = f.name
        try:
            entries = parse.parse_master(path, "training")
        finally:
            os.unlink(path)
        self.assertEqual([e["id"] for e in entries], ["DATA.01", "DATA.02"])
        self.assertEqual(entries[0]["fields"]["mechanism"], "起因。\n---\n# 使用注意\n继续说明。")
        self.assertEqual(entries[0]["fields"]["discovery"], "每日扫描")
        self.assertIn("---\n\n# 使用注意\n继续说明。", entries[0]["raw"])
        self.assertEqual(entries[0]["line_end"], 10)

    def test_entry_stops_before_next_chapter_separator(self):
        content = """# CKPT · 检查点
## CKPT
### CKPT.28 `checkpoint_example`
stage: `pretrain/sft` · Cov: `NEW` · 置信度 `documented`
- **机制**：上一章故障。
- **发现来源**：2026-09-23 每日扫描

---

# DATA · 数据管线
## DATA
### DATA.01 `data_example`
stage: `pretrain` · Cov: `NEW` · 置信度 `documented`
- **机制**：下一章故障。
- **发现来源**：2026-09-24 每日扫描
"""
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False) as f:
            f.write(content)
            path = f.name
        try:
            entries = parse.parse_master(path, "training")
        finally:
            os.unlink(path)
        self.assertEqual([e["id"] for e in entries], ["CKPT.28", "DATA.01"])
        self.assertEqual(entries[0]["fields"]["discovery"], "2026-09-23 每日扫描")
        self.assertNotIn("# DATA", entries[0]["raw"])
        self.assertEqual(entries[1]["category"], "DATA")


if __name__ == "__main__":
    unittest.main()
