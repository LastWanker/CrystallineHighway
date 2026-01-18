"""根据 SQLite 数据库生成层级视图。"""

from __future__ import annotations

import argparse
import random
import sqlite3
from dataclasses import dataclass
from pathlib import Path

from src.crystalline_highway.config import MemoryConfig


@dataclass(frozen=True)
class MetaInfo:
    meta_id: str
    text: str
    crystallized_count: int


@dataclass(frozen=True)
class InstanceInfo:
    node_id: str
    meta_id: str


@dataclass(frozen=True)
class EdgeInfo:
    src_id: str
    dst_id: str
    edge_type: str


def _meta_sort_key(meta_id: str) -> int:
    if meta_id.startswith("meta-"):
        suffix = meta_id.replace("meta-", "", 1)
        if suffix.isdigit():
            return int(suffix)
    return 10**9


def load_database(db_path: Path) -> tuple[dict[str, MetaInfo], list[InstanceInfo], list[EdgeInfo]]:
    if not db_path.exists():
        raise FileNotFoundError(f"数据库不存在：{db_path}")

    meta_entries: dict[str, MetaInfo] = {}
    instance_nodes: list[InstanceInfo] = []
    edges: list[EdgeInfo] = []

    with sqlite3.connect(db_path) as conn:
        for row in conn.execute(
            """
            SELECT meta_id, text, crystallized_count
            FROM meta_entries
            """
        ):
            meta_id, text, crystallized_count = row
            meta_entries[meta_id] = MetaInfo(
                meta_id=meta_id,
                text=text,
                crystallized_count=int(crystallized_count),
            )

        for row in conn.execute(
            """
            SELECT node_id, meta_id
            FROM instance_nodes
            """
        ):
            node_id, meta_id = row
            instance_nodes.append(InstanceInfo(node_id=node_id, meta_id=meta_id))

        for row in conn.execute(
            """
            SELECT src_id, dst_id, edge_type
            FROM graph_edges
            """
        ):
            src_id, dst_id, edge_type = row
            edges.append(EdgeInfo(src_id=src_id, dst_id=dst_id, edge_type=edge_type))

    return meta_entries, instance_nodes, edges


def build_layout(
    meta_entries: dict[str, MetaInfo],
    instance_nodes: list[InstanceInfo],
    *,
    layer_spacing: float = 1.8,
    x_spacing: float = 1.6,
    jitter: float = 0.15,
) -> dict[str, tuple[float, float]]:
    randomizer = random.Random(42)
    layer_map: dict[int, list[InstanceInfo]] = {}
    for node in instance_nodes:
        meta = meta_entries.get(node.meta_id)
        if meta is None:
            continue
        layer_map.setdefault(meta.crystallized_count, []).append(node)

    positions: dict[str, tuple[float, float]] = {}
    for count in sorted(layer_map.keys()):
        nodes = sorted(
            layer_map[count],
            key=lambda item: (_meta_sort_key(item.meta_id), item.node_id),
        )
        if not nodes:
            continue
        total_width = (len(nodes) - 1) * x_spacing
        for index, node in enumerate(nodes):
            x = index * x_spacing - total_width / 2
            x += randomizer.uniform(-jitter, jitter)
            y = count * layer_spacing + randomizer.uniform(-jitter, jitter)
            positions[node.node_id] = (x, y)
    return positions


def render_plot(
    output_path: Path,
    meta_entries: dict[str, MetaInfo],
    instance_nodes: list[InstanceInfo],
    edges: list[EdgeInfo],
    positions: dict[str, tuple[float, float]],
) -> None:
    try:
        import matplotlib.pyplot as plt
        from matplotlib import font_manager, rcParams
    except ImportError as exc:  # pragma: no cover - 可选依赖
        raise SystemExit("缺少 matplotlib，请先安装后再运行可视化脚本。") from exc

    def configure_chinese_font() -> None:
        candidates = [
            "Noto Sans CJK SC",
            "Noto Sans CJK",
            "Source Han Sans SC",
            "SimHei",
            "Microsoft YaHei",
            "WenQuanYi Zen Hei",
            "PingFang SC",
            "Arial Unicode MS",
        ]
        for name in candidates:
            try:
                font_path = font_manager.findfont(name, fallback_to_default=False)
            except ValueError:
                continue
            if Path(font_path).exists():
                rcParams["font.family"] = name
                rcParams["font.sans-serif"] = [name]
                rcParams["axes.unicode_minus"] = False
                return

    configure_chinese_font()

    max_nodes = max((len(instance_nodes), 1))
    max_layer = max(
        (meta_entries[node.meta_id].crystallized_count for node in instance_nodes if node.meta_id in meta_entries),
        default=1,
    )
    fig_width = max(8.0, min(20.0, max_nodes * 0.6))
    fig_height = max(6.0, min(20.0, max_layer * 1.2))

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    for edge in edges:
        if edge.src_id not in positions or edge.dst_id not in positions:
            continue
        src = positions[edge.src_id]
        dst = positions[edge.dst_id]
        if edge.edge_type == "vertical":
            color = "red"
            zorder = 3
        else:
            color = "lightskyblue"
            zorder = 2
        ax.annotate(
            "",
            xy=dst,
            xytext=src,
            arrowprops={"arrowstyle": "->", "color": color, "lw": 1.5, "alpha": 0.9},
            zorder=zorder,
        )

    for node in instance_nodes:
        if node.node_id not in positions:
            continue
        meta = meta_entries.get(node.meta_id)
        if meta is None:
            continue
        x, y = positions[node.node_id]
        ax.scatter([x], [y], color="#444444", s=50, zorder=4)
        ax.text(
            x,
            y,
            meta.text,
            ha="center",
            va="center",
            fontsize=8,
            color="white",
            zorder=5,
            bbox={"facecolor": "#222222", "alpha": 0.85, "boxstyle": "round,pad=0.25"},
        )

    ax.set_title("Crystalline Highway 层级视图")
    ax.axis("off")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="绘制固化层级视图")
    parser.add_argument(
        "--db",
        type=Path,
        default=Path(MemoryConfig().storage_path),
        help="SQLite 数据库路径",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data") / "hierarchy_view.png",
        help="输出图片路径",
    )
    args = parser.parse_args()

    meta_entries, instance_nodes, edges = load_database(args.db)
    if not instance_nodes:
        raise SystemExit("数据库中没有实例节点，请先背诵写入数据。")
    positions = build_layout(meta_entries, instance_nodes)
    render_plot(args.output, meta_entries, instance_nodes, edges, positions)
    print(f"层级视图已生成：{args.output}")


if __name__ == "__main__":
    main()
