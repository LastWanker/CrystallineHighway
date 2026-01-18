"""SQLite 持久化存储实现。"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Dict

from ..models.graph import EdgeType, Graph
from ..models.instance import InstanceNode, InstanceStats
from ..models.meta import MetaEntry


class SQLiteStore:
    """使用 SQLite 持久化：词典、实例册、图三套结构分表保存。"""

    def __init__(self, db_path: str) -> None:
        self.db_path = db_path
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self.meta_table: Dict[str, MetaEntry] = {}
        self.instance_table: Dict[str, InstanceNode] = {}
        self.graph = Graph()
        self._init_schema()

    def _init_schema(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS meta_entries (
                    normalized_text TEXT PRIMARY KEY,
                    meta_id TEXT NOT NULL,
                    text TEXT NOT NULL,
                    global_freq REAL NOT NULL,
                    private_freq REAL NOT NULL,
                    level INTEGER NOT NULL,
                    crystallized_count INTEGER NOT NULL,
                    labels TEXT NOT NULL,
                    category_vector TEXT NOT NULL,
                    instances TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS instance_nodes (
                    node_id TEXT PRIMARY KEY,
                    meta_id TEXT NOT NULL,
                    vector_pos TEXT NOT NULL,
                    stats TEXT NOT NULL,
                    hub_penalty REAL NOT NULL,
                    payload TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS graph_edges (
                    src_id TEXT NOT NULL,
                    dst_id TEXT NOT NULL,
                    edge_type TEXT NOT NULL,
                    walk_count INTEGER NOT NULL,
                    PRIMARY KEY (src_id, dst_id)
                )
                """
            )
            conn.commit()

    def load(self) -> None:
        """从 SQLite 加载三套结构。"""

        self.meta_table = {}
        self.instance_table = {}
        self.graph = Graph()
        with sqlite3.connect(self.db_path) as conn:
            for row in conn.execute(
                """
                SELECT normalized_text, meta_id, text, global_freq, private_freq, level,
                       crystallized_count, labels, category_vector, instances
                FROM meta_entries
                """
            ):
                (
                    normalized_text,
                    meta_id,
                    text,
                    global_freq,
                    private_freq,
                    level,
                    crystallized_count,
                    labels,
                    category_vector,
                    instances,
                ) = row
                entry = MetaEntry(
                    meta_id=meta_id,
                    text=text,
                    normalized_text=normalized_text,
                    global_freq=global_freq,
                    private_freq=private_freq,
                    level=level,
                    crystallized_count=crystallized_count,
                    labels=set(json.loads(labels)),
                    category_vector=list(json.loads(category_vector)),
                    instances=set(json.loads(instances)),
                )
                self.meta_table[normalized_text] = entry
            for row in conn.execute(
                """
                SELECT node_id, meta_id, vector_pos, stats, hub_penalty, payload
                FROM instance_nodes
                """
            ):
                node_id, meta_id, vector_pos, stats, hub_penalty, payload = row
                stats_data = json.loads(stats)
                node = InstanceNode(
                    node_id=node_id,
                    meta_id=meta_id,
                    vector_pos=list(json.loads(vector_pos)),
                    stats=InstanceStats(**stats_data),
                    hub_penalty=hub_penalty,
                    payload=dict(json.loads(payload)),
                )
                self.instance_table[node_id] = node
            for row in conn.execute(
                """
                SELECT src_id, dst_id, edge_type, walk_count
                FROM graph_edges
                """
            ):
                src_id, dst_id, edge_type, walk_count = row
                self.graph.set_edge(src_id, dst_id, EdgeType(edge_type), int(walk_count))

    def save(self) -> None:
        """保存当前内存态到 SQLite。"""

        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM meta_entries")
            conn.execute("DELETE FROM instance_nodes")
            conn.execute("DELETE FROM graph_edges")
            conn.executemany(
                """
                INSERT INTO meta_entries (
                    normalized_text, meta_id, text, global_freq, private_freq, level,
                    crystallized_count, labels, category_vector, instances
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        key,
                        meta.meta_id,
                        meta.text,
                        meta.global_freq,
                        meta.private_freq,
                        meta.level,
                        meta.crystallized_count,
                        json.dumps(sorted(meta.labels), ensure_ascii=False),
                        json.dumps(meta.category_vector, ensure_ascii=False),
                        json.dumps(sorted(meta.instances), ensure_ascii=False),
                    )
                    for key, meta in self.meta_table.items()
                ],
            )
            conn.executemany(
                """
                INSERT INTO instance_nodes (
                    node_id, meta_id, vector_pos, stats, hub_penalty, payload
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        node.node_id,
                        node.meta_id,
                        json.dumps(node.vector_pos, ensure_ascii=False),
                        json.dumps(node.stats.__dict__, ensure_ascii=False),
                        node.hub_penalty,
                        json.dumps(node.payload, ensure_ascii=False),
                    )
                    for node in self.instance_table.values()
                ],
            )
            edges = []
            for src_id, targets in self.graph.out_edges.items():
                for dst_id, edge in targets.items():
                    edges.append((src_id, dst_id, edge.edge_type.value, edge.walk_count))
            conn.executemany(
                """
                INSERT INTO graph_edges (src_id, dst_id, edge_type, walk_count)
                VALUES (?, ?, ?, ?)
                """,
                edges,
            )
            conn.commit()
