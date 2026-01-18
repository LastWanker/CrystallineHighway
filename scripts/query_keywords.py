"""根据关键词查询并输出结果。"""

from __future__ import annotations

import argparse

from crystalline_highway.core.memory_system import MemorySystem


def main() -> None:
    parser = argparse.ArgumentParser(description="关键词查询")
    parser.add_argument("keywords", nargs="*", help="关键词（可空格分隔）")
    args = parser.parse_args()

    if args.keywords:
        query_text = "".join(args.keywords)
    else:
        query_text = input("请输入关键词：").strip()

    if not query_text:
        raise SystemExit("关键词不能为空。")

    system = MemorySystem()
    results = system.retrieve_text(query_text)

    if not results:
        print("没有找到相关结果。")
        return

    print("查询结果：")
    for text, sources in results.items():
        source_text = ", ".join(sources) if sources else "无"
        print(f"- {text} (来源: {source_text})")


if __name__ == "__main__":
    main()
