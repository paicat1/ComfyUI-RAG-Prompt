#!/usr/bin/env python3
"""Fix hardcoded task prompts in nodes.py to not conflict with user system prompts."""
import re
from pathlib import Path

nodes_file = Path(__file__).resolve().parent.parent / "nodes.py"
content = nodes_file.read_text(encoding="utf-8")

# === Fix 1: stage1_task ===
# Replace the hardcoded structural instructions with just the user theme
old_stage1 = (
    '            # \u7b2c\u4e00\u9636\u6bb5\u4ec5\u7528\u4e8e\u201c\u7ed3\u6784\u5316\u8349\u6848\u201d\u751f\u6210\n'
    '            stage1_context = stage1["context"] if stage1["rag_hit"] else ""\n'
    '            stage1_task = (\n'
    '                "\u8bf7\u6839\u636e\u7528\u6237\u9700\u6c42\u8f93\u51fa\u201c\u7ed3\u6784\u5316\u8349\u6848\u201d\uff0c\u7528\u4e8e\u540e\u7eed\u586b\u8bcd\u3002\\n"\n'
    '                "\u8981\u6c42\uff1a\\n"\n'
    '                "1) \u53ea\u8f93\u51fa\u7ed3\u6784\u4e0e\u69fd\u4f4d\uff0c\u4e0d\u5199\u5b8c\u6574\u6587\u6848\u53e5\u5b50\uff1b\\n"\n'
    '                "2) \u4f7f\u7528\u5206\u6bb5\u6807\u9898\u548c\u8981\u70b9\uff08\u53ef\u7559\u5360\u4f4d\u7b26\uff09\uff1b\\n"\n'
    '                "3) \u7ed3\u6784\u9700\u8986\u76d6\uff1a\u4e3b\u4f53\u3001\u955c\u5934\u3001\u5149\u7ebf\u3001\u6750\u8d28\u3001\u670d\u88c5\u3001\u80cc\u666f\u3001\u98ce\u683c\u3001\u8d1f\u9762\u7ea6\u675f\uff1b\\n"\n'
    '                "4) \u4e25\u7981\u8f93\u51fa\u201c\u6700\u7ec8\u7b54\u6848\u201d\u201c\u6210\u54c1\u63d0\u793a\u8bcd\u201d\u4e4b\u7c7b\u5185\u5bb9\u3002\\n\\n"\n'
    '                f"\u7528\u6237\u9700\u6c42\uff1a\\n{question}"\n'
    '            )'
)

new_stage1 = (
    '            # \u7b2c\u4e00\u9636\u6bb5\u9aa8\u67b6\u751f\u6210\uff0c\u5177\u4f53\u6b65\u9aa4\u548c\u8f93\u51fa\u683c\u5f0f\u7531 stage1_system_prompt \u63a7\u5236\n'
    '            stage1_context = stage1["context"] if stage1["rag_hit"] else ""\n'
    '            stage1_task = f"\u7528\u6237\u4e3b\u9898\uff1a\\n{question}"'
)

if old_stage1 in content:
    content = content.replace(old_stage1, new_stage1)
    print("Fix 1 (stage1_task): APPLIED")
else:
    print("Fix 1 (stage1_task): NOT FOUND - please check manually")

# === Fix 2: final_task ===
# Replace the hardcoded fill instructions with just user theme + skeleton
old_stage2 = (
    '        if stage1_outline:\n'
    '            final_task = (\n'
    '                "\u4f60\u5c06\u6536\u5230\u4e00\u4e2a\u201c\u7ed3\u6784\u5316\u8349\u6848\u201d\uff0c\u8bf7\u5728\u4e0d\u6539\u53d8\u8be5\u7ed3\u6784\u5c42\u7ea7\u7684\u524d\u63d0\u4e0b\uff0c\u4f7f\u7528\u68c0\u7d22\u5230\u7684\u8bcd\u6c47\u548c\u793a\u4f8b\u8fdb\u884c\u586b\u5145\u3002\\n"\n'
    '                "\u8981\u6c42\uff1a\\n"\n'
    '                "1) \u4fdd\u7559\u7ed3\u6784\u6807\u9898\u548c\u987a\u5e8f\uff1b\\n"\n'
    '                "2) \u7528\u66f4\u5177\u4f53\u3001\u53ef\u6267\u884c\u7684\u8bcd\u6c47\u586b\u5145\u5360\u4f4d\uff1b\\n"\n'
    '                "3) \u4e0d\u8981\u8f93\u51fa\u201c\u7ed3\u6784\u5316\u8349\u6848\u201d\u201c\u6700\u7ec8\u7ed3\u679c\u201d\u7b49\u6807\u7b7e\u8bcd\uff1b\\n"\n'
    '                "4) \u4e0d\u8981\u8f93\u51fa markdown \u6807\u9898\uff08\u5982 #\uff09\uff1b\\n"\n'
    '                "5) \u4e0d\u8981\u8f93\u51fa\u89e3\u91ca\u3001\u5206\u6790\u3001\u524d\u540e\u7f00\uff1b\\n"\n'
    '                "6) \u4ec5\u8f93\u51fa\u6700\u7ec8\u53ef\u76f4\u63a5\u4f7f\u7528\u7684\u7ed3\u679c\u6b63\u6587\u3002\\n\\n"\n'
    '                f"\u7528\u6237\u9700\u6c42\uff1a\\n{question}\\n\\n"\n'
    '                f"\u7ed3\u6784\u5316\u8349\u6848\uff1a\\n{stage1_outline}"\n'
    '            )'
)

new_stage2 = (
    '        if stage1_outline:\n'
    '            # \u7b2c\u4e8c\u9636\u6bb5\u586b\u8bcd\u751f\u6210\uff0c\u5177\u4f53\u903b\u8f91\u548c\u8f93\u51fa\u683c\u5f0f\u7531 stage2_system_prompt \u63a7\u5236\n'
    '            final_task = (\n'
    '                f"\u7528\u6237\u4e3b\u9898\uff1a\\n{question}\\n\\n"\n'
    '                f"\u7b2c\u4e00\u9636\u6bb5\u8f93\u51fa\u7684\u9aa8\u67b6\uff1a\\n{stage1_outline}"\n'
    '            )'
)

if old_stage2 in content:
    content = content.replace(old_stage2, new_stage2)
    print("Fix 2 (final_task): APPLIED")
else:
    print("Fix 2 (final_task): NOT FOUND - please check manually")

nodes_file.write_text(content, encoding="utf-8")
print("Done - nodes.py updated")
