"""手动粘贴文本进行背诵。"""

from __future__ import annotations

from src.crystalline_highway.core.memory_system import MemorySystem

# TODO: 把你要背诵的文本粘贴到这里。
TEXT_TO_RECITE = """
和赵小姐清晨爬紫金山，雾气缠着松林，她递来温水和巧克力，山顶风大，我心却发热。
""".strip()


def main() -> None:
    if not TEXT_TO_RECITE:
        raise SystemExit("请先在 recite_manual.py 中粘贴要背诵的文本。")
    system = MemorySystem()
    system.recite_text(TEXT_TO_RECITE)
    system.store.save()
    print("背诵完成，数据已写入数据库。")


if __name__ == "__main__":
    main()
