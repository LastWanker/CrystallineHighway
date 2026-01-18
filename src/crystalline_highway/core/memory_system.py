"""记忆系统核心流程实现。"""

from __future__ import annotations

import math
from typing import Dict, Iterable, List, Tuple

from ..config import MemoryConfig
from ..frequency.calibration import (
    FrequencyCalibrator,
    FrequencyCalibration,
    fallback_frequency_calibration,
    load_frequency_calibration,
    save_frequency_calibration,
)
from ..frequency.global_frequency import GlobalFrequencyProvider
from ..frequency.tolerance import (
    effective_frequency,
    private_typical,
    radius_from_frequency,
)
from . import vector
from .recitation import RecitationPlanner
from .registry import Registry
from .segmentation import ChineseSegmenter
from .text_utils import normalize_text
from ..models.graph import EdgeType
from ..models.instance import InstanceNode
from ..models.session import SessionState
from ..storage import create_store


class MemorySystem:
    """寻找驱动建构的记忆系统原型。"""

    def __init__(self, config: MemoryConfig | None = None) -> None:
        self.config = config or MemoryConfig()
        self.store = create_store(self.config)
        self.store.load()
        self.global_frequency = GlobalFrequencyProvider(
            language=self.config.frequency_language,
            sample_size=self.config.frequency_word_sample_size,
            fallback_frequency=self.config.frequency_fallback_avg_freq,
        )
        self.registry = Registry(self.store, self.config, self.global_frequency)
        self.frequency_calibration = self._load_frequency_calibration()
        # 分词器与背诵计划器是“文本进入系统的入口”，对应指导文件里的
        # “先拆分、再倒序背诵、直到收敛”的流程。
        self.segmenter = ChineseSegmenter()
        self.recitation_planner = RecitationPlanner(self.segmenter)

    def _load_frequency_calibration(self) -> FrequencyCalibration:
        calibration = load_frequency_calibration(self.config.frequency_calibration_path)
        if calibration is not None:
            return calibration
        calibration = fallback_frequency_calibration(
            self.config,
            avg_freq=self.global_frequency.typical_frequency(),
        )
        if self.config.frequency_auto_calibrate:
            calibrator = FrequencyCalibrator(self.config, self.global_frequency)
            calibration = calibrator.build(self.registry.vector_provider)
            save_frequency_calibration(self.config.frequency_calibration_path, calibration)
        return calibration

    def _dynamic_radius(self, meta_text: str) -> float:
        """根据频率计算动态容忍度。

        设计依据：
        - 高频元 → 容忍度极小（去中心化）；
        - 低频元 → 容忍度极大（稳定锚点）。
        这正是“寻找驱动建构”的核心直觉之一。
        """

        meta = self.store.meta_table.get(normalize_text(meta_text))
        if meta is None:
            return self.config.radius_ceiling
        private_scale = self._private_typical_frequency()
        effective_freq = effective_frequency(
            meta.global_freq,
            meta.private_freq,
            private_scale,
            self.config.frequency_private_beta,
            self.config.frequency_private_cap,
            self.config.frequency_eps,
        )
        radius = radius_from_frequency(
            self.frequency_calibration,
            effective_freq,
            self.config.frequency_hit_probability,
            self.frequency_calibration.avg_freq,
            self.config.radius_floor,
            self.config.radius_ceiling,
            self.config.frequency_eps,
        )
        # 固化元的容忍度更大，但用对数缓和扩张速度
        if meta.level > 0:
            count = max(meta.crystallized_count, 1)
            radius *= self._crystallized_radius_multiplier(count)
        return min(max(radius, self.config.radius_floor), self.config.radius_ceiling)

    def _find_candidates(
        self, meta_text: str, center: List[float], radius: float
    ) -> List[Tuple[InstanceNode, float]]:
        """在半径内寻找匹配实例。"""

        meta = self.store.meta_table.get(normalize_text(meta_text))
        if meta is None:
            return []
        candidates = []
        for node_id in meta.instances:
            node = self.store.instance_table[node_id]
            dist = vector.distance(center, node.vector_pos)
            if dist <= radius:
                candidates.append((node, dist))
        return candidates

    def _select_best(self, candidates: List[Tuple[InstanceNode, float]]) -> InstanceNode | None:
        """择优规则：先近，再常走。"""

        if not candidates:
            return None
        candidates.sort(key=lambda item: (item[1], -item[0].stats.use_count))
        return candidates[0][0]

    def _create_instance_near(
        self, meta_text: str, center: List[float], radius: float
    ) -> InstanceNode:
        """未命中时在附近新建实例。

        这里严格遵循“寻找即建构”的规则：
        - 没命中 → 在当前上下文附近新建；
        - 新建位置受到范畴向量轻微牵引（但频率越高牵引越弱）。
        """

        meta = self.registry.ensure_meta(meta_text)
        # 高频元的范畴偏移更弱，按“有效频率”做衰减
        private_scale = self._private_typical_frequency()
        effective_freq = effective_frequency(
            meta.global_freq,
            meta.private_freq,
            private_scale,
            self.config.frequency_private_beta,
            self.config.frequency_private_cap,
            self.config.frequency_eps,
        )
        bias = vector.scale(meta.category_vector, 1.0 / max(effective_freq, 1.0))
        jitter_scale = min(radius, self.config.jitter_scale)
        return self.registry.create_instance(meta, center, bias, jitter_scale)

    def _ensure_instance(self, meta_text: str, center: List[float]) -> InstanceNode:
        """确保某元在当前位置附近有实例可用。"""

        radius = self._dynamic_radius(meta_text)
        candidates = self._find_candidates(meta_text, center, radius)
        actual_distance = min((dist for _node, dist in candidates), default=None)
        success = "成功" if candidates else "失败"
        distance_text = f"{actual_distance:.4f}" if actual_distance is not None else "无"
        print(f"寻找{meta_text}，容忍半径{radius:.4f}，实际距离{distance_text}，{success}")
        chosen = self._select_best(candidates)
        if chosen is None:
            chosen = self._create_instance_near(meta_text, center, radius)
        chosen.stats.use_count += 1
        return chosen

    def write_sequence(self, tokens: Iterable[str]) -> List[InstanceNode]:
        """写入/构筑流程：将输入文本序列化为寻找事件。

        关键点：
        - 输入是一串“寻找事件”而不是 token 化文本；
        - 每一步都留下边计数（固化燃料）；
        - 优先选择更长词条，避免已有词被拆散。
        """

        tokens = [token for token in tokens if normalize_text(token)]
        tokens = self._prefer_longer_tokens(list(tokens))
        path_nodes: List[InstanceNode] = []
        center = vector.zero_vector(self.config.vector_dim)
        prev_node: InstanceNode | None = None
        for token in tokens:
            node = self._ensure_instance(token, center)
            if prev_node is not None:
                self.store.graph.add_edge(prev_node.node_id, node.node_id, EdgeType.horizontal)
                prev_node.stats.pass_count += 1
                node.stats.pass_count += 1
                self._maybe_crystallize(prev_node, node)
            path_nodes.append(node)
            prev_node = node
            center = node.vector_pos
        self._save_if_needed()
        return path_nodes

    def recite_sequence(self, tokens: Iterable[str]) -> List[InstanceNode]:
        """背诵流程：反复走同一路径，强化固化。"""

        return self.write_sequence(tokens)

    def recite_text(self, text: str) -> None:
        """背诵调度：按拆分层级倒序输入，直到收敛或达到上限。

        新增逻辑：
        - 先用最小词素完成“实例层面的预注册”（确保实例册有落点）；
        - 再按短语/短句/长句/段落/全文倒序背诵，逐层收敛后才进入下一轮。
        """

        self._pre_register_morphemes(text)
        plan = self.recitation_planner.build_plan(text)
        self._recite_until_converged(plan)

    def _pre_register_morphemes(self, text: str) -> None:
        """用最小词素提前注册元条目与实例落点。"""

        for token in self.segmenter.segment_morphemes(text):
            if normalize_text(token):
                # 这里直接触发一次寻找，保证实例册里确实有落点。
                self._ensure_instance(token, vector.zero_vector(self.config.vector_dim))

    def _recite_until_converged(self, plan) -> None:
        """循环背诵直到收敛：对应单元被注册或固化。"""

        rounds = 0
        while rounds < self.config.recitation_max_rounds:
            rounds += 1
            for unit in plan:
                self._register_unit(unit)
                # 每个层级单元都必须收敛：也就是它们作为“寻找目标”已被注册到词典中。
                tokens = self.segmenter.segment_morphemes(unit.display_text)
                self.recite_sequence(tokens)
            if self._check_converged(plan):
                self._tag_converged(plan)
                return
        self._force_register_unresolved(plan)
        self._tag_converged(plan)

    def _check_converged(self, plan) -> bool:
        for unit in plan:
            if unit.normalized_text not in self.store.meta_table:
                return False
        return True

    def _tag_converged(self, plan) -> None:
        for unit in plan:
            meta = self.store.meta_table.get(unit.normalized_text)
            if meta is None:
                continue
            # 固化后的展示文本尽量使用带标点的原文本，便于输出阅读。
            if unit.display_text and len(unit.display_text) > len(meta.text):
                meta.text = unit.display_text
            meta.labels.add(unit.label)

    def _register_unit(self, unit) -> None:
        if not unit.normalized_text:
            return
        if unit.label == "morpheme":
            meta = self.registry.ensure_meta(
                unit.display_text,
                normalized_text=unit.normalized_text,
            )
        else:
            meta = self.store.meta_table.get(unit.normalized_text)
            if meta is None:
                return
        if not meta.instances:
            radius = self._dynamic_radius(unit.display_text)
            self._create_instance_near(
                unit.display_text,
                vector.zero_vector(self.config.vector_dim),
                radius,
            )

    def _force_register_unresolved(self, plan) -> None:
        """背诵轮次耗尽时才补注册，避免大块文本一上来就落词典。"""

        for unit in plan:
            if not unit.normalized_text:
                continue
            if unit.normalized_text in self.store.meta_table:
                continue
            meta = self.registry.ensure_meta(
                unit.display_text,
                normalized_text=unit.normalized_text,
            )
            if not meta.instances:
                radius = self._dynamic_radius(unit.display_text)
                self._create_instance_near(
                    unit.display_text,
                    vector.zero_vector(self.config.vector_dim),
                    radius,
                )

    def _private_typical_frequency(self) -> float:
        return private_typical(
            (meta.private_freq for meta in self.store.meta_table.values()),
            self.config.frequency_private_typical_default,
        )

    def _maybe_crystallize(self, left: InstanceNode, right: InstanceNode) -> None:
        """检测两元路径是否达到固化阈值。"""

        edge = self.store.graph.get_edge(left.node_id, right.node_id)
        if edge is None:
            return
        # 固化不应期：刚参与固化的实例，下一次固化直接跳过。
        # 这对应“对一些问题的回答.txt”中的“不应期 flag”规则，
        # 用于避免 A-B 刚固化又立即参与 A-B-C 的连锁固化。
        if left.stats.refractory > 0 or right.stats.refractory > 0:
            left.stats.refractory = max(0, left.stats.refractory - 1)
            right.stats.refractory = max(0, right.stats.refractory - 1)
            return
        total_count = self._crystallized_count(left) + self._crystallized_count(right)
        threshold = self._crystallize_threshold(total_count)
        if edge.walk_count < threshold:
            return
        # 生成固化元文本：直接拼接，符合“固化时输出可读文本”的要求。
        left_meta = self._meta_text_from_id(left.meta_id)
        right_meta = self._meta_text_from_id(right.meta_id)
        crystallized_text = f"{left_meta}{right_meta}"
        crystallized_meta = self.registry.ensure_meta(crystallized_text)
        self._assign_crystallized_count(crystallized_meta, total_count)
        # 层级标签提升
        crystallized_meta.level = max(
            crystallized_meta.level,
            self._meta_level(left.meta_id) + 1,
            self._meta_level(right.meta_id) + 1,
        )
        # 固化元位置：子元向量平均并向离心方向偏移。
        # 这体现“固化元应更稳定、更稀疏”的设定。
        mean_pos = vector.mean([left.vector_pos, right.vector_pos])
        offset = vector.scale(vector.normalize(mean_pos), self.config.crystallize_offset_scale)
        new_pos = vector.add(mean_pos, offset)
        new_node = self.registry.create_instance(
            crystallized_meta,
            new_pos,
            bias_vector=vector.zero_vector(self.config.vector_dim),
            jitter_scale=0.0,
        )
        new_node.payload["source"] = crystallized_text
        new_node.payload["crystallized_count"] = str(total_count)
        # 固化元容忍度极高、触发频率极低：在这里体现为高层级标签
        # 旧路计数抽走迁移
        decrement = max(1, threshold // 2)
        self.store.graph.downgrade_edge(left.node_id, right.node_id, decrement)
        self.store.graph.reset_edge(left.node_id, right.node_id, 1)
        # 建立纵向路径（索引路）
        self.store.graph.add_edge(left.node_id, new_node.node_id, EdgeType.vertical)
        self.store.graph.add_edge(right.node_id, new_node.node_id, EdgeType.vertical)
        left.stats.refractory = 1
        right.stats.refractory = 1

    def retrieve(self, query_tokens: Iterable[str]) -> Dict[str, List[str]]:
        """检索/回忆流程，返回候选节点及其来源。"""

        session = SessionState(ttl_budget=self.config.retrieval_ttl)
        tokens = [token for token in query_tokens if normalize_text(token)]
        tokens = self._prefer_longer_tokens(list(tokens))
        seed_nodes = self._resolve_query_seed_nodes(tokens)
        if not seed_nodes:
            return {}
        # 多源扩散
        frontier = [(node.node_id, session.ttl_budget) for node in seed_nodes]
        for node in seed_nodes:
            session.touch(node.node_id, source=node.node_id)
        while frontier:
            current_id, ttl = frontier.pop(0)
            if ttl <= 0:
                continue
            current_node = self.store.instance_table[current_id]
            neighbors = self.store.graph.neighbors(current_id, include_reverse_horizontal=True)
            for neighbor_id in neighbors:
                neighbor_node = self.store.instance_table[neighbor_id]
                penalty = min(neighbor_node.hub_penalty, self.config.hub_penalty_cap)
                next_ttl = ttl - 1 - penalty
                if next_ttl < 0:
                    continue
                session.touch(neighbor_id, source=current_id)
                frontier.append((neighbor_id, next_ttl))
        ranked = sorted(session.light_count.items(), key=lambda item: -item[1])
        return self._collect_retrieval_results(ranked, session)

    def _resolve_query_seed_nodes(self, tokens: List[str]) -> List[InstanceNode]:
        """查询态仅用已有实例，不在词典中落新点。"""

        seed_nodes: List[InstanceNode] = []
        for token in tokens:
            meta = self.store.meta_table.get(normalize_text(token))
            if meta is None:
                continue
            candidates = [self.store.instance_table[node_id] for node_id in meta.instances]
            if not candidates:
                continue
            candidates.sort(key=lambda node: -node.stats.use_count)
            seed_nodes.append(candidates[0])
        return seed_nodes

    def _save_if_needed(self) -> None:
        if self.config.storage_auto_save:
            self.store.save()

    def _collect_retrieval_results(
        self,
        ranked: List[Tuple[str, int]],
        session: SessionState,
    ) -> Dict[str, List[str]]:
        """按配额输出短句/长句/段落/记忆结果。"""

        results: Dict[str, List[str]] = {}
        selected: set[str] = set()

        def add_by_label(label: str, quota: int) -> None:
            for node_id, _count in ranked:
                if node_id in selected:
                    continue
                node = self.store.instance_table[node_id]
                meta = self._meta_from_id(node.meta_id)
                if meta is None or label not in meta.labels:
                    continue
                results[meta.text] = session.hit_sources.get(node_id, [])
                selected.add(node_id)
                if len([k for k in results if label in self._meta_labels(k)]) >= quota:
                    break

        add_by_label("short_sentence", self.config.retrieval_quota_short)
        add_by_label("long_sentence", self.config.retrieval_quota_long)
        add_by_label("paragraph", self.config.retrieval_quota_paragraph)
        add_by_label("full_text", self.config.retrieval_quota_memory)

        possible_quota = self.config.retrieval_quota_possible
        for node_id, _count in ranked:
            if node_id in selected:
                continue
            node = self.store.instance_table[node_id]
            meta = self._meta_from_id(node.meta_id)
            if meta is None:
                continue
            results[meta.text] = session.hit_sources.get(node_id, [])
            selected.add(node_id)
            possible_quota -= 1
            if possible_quota <= 0:
                break
        return results

    def _prefer_longer_tokens(self, tokens: List[str]) -> List[str]:
        """优先选择更长的词条（例如字典中已有 CD 时选 CD）。"""

        if not tokens:
            return []
        merged: List[str] = []
        index = 0
        while index < len(tokens):
            if index + 1 < len(tokens):
                candidate = f"{tokens[index]}{tokens[index + 1]}"
                if normalize_text(candidate) in self.store.meta_table:
                    merged.append(candidate)
                    index += 2
                    continue
            merged.append(tokens[index])
            index += 1
        return merged

    def _crystallize_threshold(self, total_count: int) -> int:
        """根据固化元数量标签计算阈值。"""

        if total_count <= 1:
            return self.config.crystallize_threshold
        # 指导文件要求：阈值 = ceil(log2(固化元数量)) + 1。
        # 这能保证固化层级递增（2 → 3 → 4 ...）并避免过快固化。
        return math.ceil(math.log2(total_count)) + 1

    def _crystallized_radius_multiplier(self, total_count: int) -> float:
        """固化元容忍度倍率（对数缓和）。"""

        return self.config.crystallize_radius_multiplier * (math.log2(total_count + 1) + 1.0)

    def _crystallized_count(self, node: InstanceNode) -> int:
        meta = self._meta_from_id(node.meta_id)
        if meta is None:
            return 1
        if "crystallized_count" in node.payload:
            try:
                return int(node.payload["crystallized_count"])
            except ValueError:
                return meta.crystallized_count
        return meta.crystallized_count

    def _assign_crystallized_count(self, meta, total_count: int) -> None:
        if meta.crystallized_count == 1:
            meta.crystallized_count = total_count

    def _meta_text_from_id(self, meta_id: str) -> str:
        meta = self._meta_from_id(meta_id)
        if meta is None:
            return meta_id
        return meta.text

    def _meta_from_id(self, meta_id: str):
        for meta in self.store.meta_table.values():
            if meta.meta_id == meta_id:
                return meta
        return None

    def _meta_level(self, meta_id: str) -> int:
        meta = self._meta_from_id(meta_id)
        if meta is None:
            return 0
        return meta.level

    def _meta_labels(self, text: str) -> List[str]:
        meta = self.store.meta_table.get(normalize_text(text))
        if meta is None:
            return []
        return list(meta.labels)

    def write_text(self, text: str) -> List[InstanceNode]:
        """直接写入原始文本：先切分成短语级词条再写入。"""

        tokens = self.segmenter.segment_words(text)
        return self.write_sequence(tokens)

    def retrieve_text(self, text: str) -> Dict[str, List[str]]:
        """直接检索原始文本：先切分成短语级词条作为起点。"""

        tokens = self.segmenter.segment_words(text)
        return self.retrieve(tokens)
