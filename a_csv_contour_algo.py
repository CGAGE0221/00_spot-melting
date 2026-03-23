import numpy as np
import networkx as nx
from shapely.geometry import Polygon, MultiPolygon, Point
from shapely.ops import nearest_points


class ContourGenerator:
    def __init__(self, geometry, contour_dwell_time, contour_beam_current, inner_projection_offset):
        """
        :param geometry: Shapely Polygon/MultiPolygon (完美支持带孔的环形)
        :param contour_dwell_time: 轮廓停留时间 (us)
        :param contour_beam_current: 轮廓束流 (mA)
        :param inner_projection_offset: 内部投影偏移量 (mm) - 投影目标线
        """
        self.geometry = geometry
        self.dwell_time = contour_dwell_time
        self.beam_current = contour_beam_current
        self.inner_projection = inner_projection_offset

        # 遵循文献 B6 策略硬编码的常量
        self.contour_offsets = [0.09, 0.18, 0.27]
        self.num_lines = len(self.contour_offsets)
        self.line_spacing = 0.09
        self.spot_spacing = 0.10
        self.num_melt_groups = 8
        self.outer_projection = 0.05  # 剔除过冲危险区的外部界限

    # ──────────────────────────────────────────────
    #  1. 环提取工具
    # ──────────────────────────────────────────────
    def _extract_rings(self, geom):
        """从 Polygon / MultiPolygon 中提取所有 LinearRing（外环+内孔环）"""
        rings = []
        if isinstance(geom, Polygon):
            rings.append(geom.exterior)
            rings.extend(geom.interiors)
        elif isinstance(geom, MultiPolygon):
            for poly in geom.geoms:
                rings.append(poly.exterior)
                rings.extend(poly.interiors)
        return rings

    # ──────────────────────────────────────────────
    #  2. 投影过滤 Infill（与原实现一致）
    # ──────────────────────────────────────────────
    def project_and_filter_infill(self, spots, melting_indices):
        """
        将靠近边缘的填充点投影到 inner_bound 上，剔除太靠近边缘的点。
        ⚠ 会原地修改 spots[idx] 的坐标。
        """
        print(f"\n  [轮廓搭接] 执行 Infill 投影 | 剔除区: <{self.outer_projection}mm, "
              f"投影目标: {self.inner_projection}mm")

        inner_bound = self.geometry.buffer(-self.inner_projection)
        outer_bound = self.geometry.buffer(-self.outer_projection)

        if inner_bound.is_empty:
            print("  [警告] 几何体太小或孔壁太薄，无法生成内部投影安全线！")
            return melting_indices

        target_lines = self._extract_rings(inner_bound)
        new_indices = []
        projected_count, discarded_count = 0, 0

        for idx in melting_indices:
            x, y = spots[idx]
            pt = Point(x, y)

            if inner_bound.contains(pt):
                new_indices.append(idx)
            elif outer_bound.contains(pt):
                min_dist = float('inf')
                best_proj = pt
                for ring in target_lines:
                    p1, p2 = nearest_points(pt, ring)
                    dist = pt.distance(p2)
                    if dist < min_dist:
                        min_dist = dist
                        best_proj = p2
                spots[idx] = (best_proj.x, best_proj.y)
                new_indices.append(idx)
                projected_count += 1
            else:
                discarded_count += 1

        print(f"  -> 原填充点: {len(melting_indices)} | 投影点: {projected_count} | "
              f"剔除点: {discarded_count} | 最终保留: {len(new_indices)}")
        return new_indices

    # ──────────────────────────────────────────────
    #  3. 轮廓预处理：生成 contour groups
    # ──────────────────────────────────────────────
    def generate_contour_groups(self):
        """
        按文献 Section 2.1.1:
        - 从 inner_projection 开始向外，每隔 line_spacing 生成一条轮廓线
        - 同一距离的所有 ring 收集为一个 contour group
        - 从最靠近 infill 的 contour group 开始排列（距离降序 → 越大的offset越靠近infill）
        """
        groups = []
        for offset in self.contour_offsets:
            # Negative buffer means "toward material interior":
            # outer boundary moves inward, while hole boundaries move outward.
            buffered_geom = self.geometry.buffer(-offset)
            if not buffered_geom.is_empty:
                rings = self._extract_rings(buffered_geom)
                if rings:
                    groups.append({'distance': offset, 'rings': rings})

        # 文献: 从最靠近 infill 的开始 → offset 值最大的先扫
        # 先走最靠近填充的一圈，再逐步向外到真实边界附近。
        groups.sort(key=lambda x: x['distance'], reverse=True)
        groups.sort(key=lambda x: x['distance'], reverse=True)
        return groups

    # ──────────────────────────────────────────────
    #  4. 在单条 ring 上生成等距点（点数为 num_melt_groups 的倍数）
    # ──────────────────────────────────────────────
    def _sample_spots_on_ring(self, ring):
        """
        文献 Section 2.1.1:
        - 点间距 ≈ spot_spacing, 但点数必须是 num_melt_groups 的倍数
        - 若不满足则略微减小点间距
        返回: list of (x, y)
        """
        length = ring.length
        if length < 1e-6:
            return []

        raw_count = length / self.spot_spacing
        num_spots = max(self.num_melt_groups,
                        int(np.ceil(raw_count / self.num_melt_groups)) * self.num_melt_groups)
        actual_spacing = length / num_spots

        points = []
        for i in range(num_spots):
            pt = ring.interpolate(i * actual_spacing)
            points.append((pt.x, pt.y))
        return points

    # ──────────────────────────────────────────────
    #  5. MST 路径规划：多条轮廓线的排序与分段
    # ──────────────────────────────────────────────
    def _compute_min_distance_between_rings(self, points_a, points_b):
        """
        计算两条轮廓线之间的最短距离及对应的点索引。
        返回: (min_dist, idx_a, idx_b)
        """
        coords_a = np.array(points_a)
        coords_b = np.array(points_b)

        # 使用广播计算所有点对的距离
        diff = coords_a[:, np.newaxis, :] - coords_b[np.newaxis, :, :]
        dist_matrix = np.linalg.norm(diff, axis=2)

        flat_idx = np.argmin(dist_matrix)
        idx_a, idx_b = np.unravel_index(flat_idx, dist_matrix.shape)
        min_dist = dist_matrix[idx_a, idx_b]
        return min_dist, int(idx_a), int(idx_b)

    def _build_contour_segment_sequence(self, ring_points_list):
        """
        文献 Section 2.1.2 的完整实现:

        Step 1: 构建完全距离图 (每个 ring 是一个 vertex)
        Step 2: 用 Kruskal 算法找最小生成树 (MST)
        Step 3: 在 MST 边的最短距离处分割轮廓线为 contour segments,
                然后按 (counter-)clockwise 沿 segments 行走, 在 MST 边处跳转

        参数:
            ring_points_list: list of list of (x,y) — 每条轮廓线上的采样点

        返回:
            contour_segments: list of list of (x,y) — 按扫描顺序排列的分段
        """
        n_rings = len(ring_points_list)

        # ── 只有一条轮廓线：无需 MST ──
        if n_rings == 1:
            return [ring_points_list[0]]

        # ── Step 1: 完全距离图 ──
        G_dist = nx.Graph()
        G_dist.add_nodes_from(range(n_rings))

        # 缓存: edge_info[(i,j)] = (min_dist, idx_on_ring_i, idx_on_ring_j)
        edge_info = {}
        for i in range(n_rings):
            for j in range(i + 1, n_rings):
                min_d, idx_i, idx_j = self._compute_min_distance_between_rings(
                    ring_points_list[i], ring_points_list[j])
                G_dist.add_edge(i, j, weight=min_d)
                edge_info[(i, j)] = (min_d, idx_i, idx_j)
                edge_info[(j, i)] = (min_d, idx_j, idx_i)

        # ── Step 2: 最小生成树 (Kruskal) ──
        mst = nx.minimum_spanning_tree(G_dist, algorithm='kruskal')

        # ── Step 3: 根据 MST 创建 contour segment sequence ──
        #
        # 对每条 ring，收集与之相连的 MST 边在该 ring 上的分割点索引
        # 每条 ring 被这些分割点切分为若干 contour segment
        # 然后通过 DFS 遍历 MST，按 (counter-)clockwise 顺序串联所有 segment

        # 3a. 收集每条 ring 上的分割点
        ring_split_points = {i: set() for i in range(n_rings)}
        for u, v in mst.edges():
            key = (u, v) if (u, v) in edge_info else (v, u)
            _, idx_u, idx_v = edge_info[(u, v)]
            ring_split_points[u].add(idx_u)
            ring_split_points[v].add(idx_v)

        # 3b. 对每条 ring，将点按分割点切分为 segments
        #     每个 segment 是一段连续的弧，用 (ring_id, start_idx, [points...]) 表示
        ring_segments = {}  # ring_id -> list of segments
        # 同时记录每个分割点属于哪个 segment 的起点，方便跳转
        split_to_segment = {}  # (ring_id, split_idx) -> segment_index_in_ring_segments[ring_id]

        for ring_id in range(n_rings):
            points = ring_points_list[ring_id]
            n_pts = len(points)
            splits = sorted(ring_split_points[ring_id])

            if len(splits) == 0:
                # 无分割点，整条 ring 就是一个 segment
                ring_segments[ring_id] = [list(range(n_pts))]
                continue

            # 从每个分割点开始，顺时针到下一个分割点，构成一个 segment
            segments = []
            for s_idx in range(len(splits)):
                start = splits[s_idx]
                end = splits[(s_idx + 1) % len(splits)]
                seg_indices = []
                idx = start
                while True:
                    seg_indices.append(idx)
                    if idx == end and len(seg_indices) > 1:
                        break
                    idx = (idx + 1) % n_pts
                    if idx == start and len(seg_indices) > 1:
                        # 完整绕了一圈（只有一个分割点时）
                        break
                segments.append(seg_indices)
                split_to_segment[(ring_id, start)] = len(segments) - 1

            ring_segments[ring_id] = segments

        # 3c. DFS 遍历 MST，产生 contour segment sequence
        contour_segments = []
        visited_rings = set()

        def dfs_traverse(ring_id, entry_split_idx=None):
            """
            从 ring_id 进入（通过 entry_split_idx 分割点），
            按顺时针遍历该 ring 的所有 segments，
            在遇到通往未访问 ring 的 MST 边时递归跳转。
            """
            visited_rings.add(ring_id)
            points = ring_points_list[ring_id]
            segments = ring_segments[ring_id]
            n_segs = len(segments)

            if n_segs == 0:
                return

            # 确定从哪个 segment 开始
            if entry_split_idx is not None and (ring_id, entry_split_idx) in split_to_segment:
                start_seg_idx = split_to_segment[(ring_id, entry_split_idx)]
            else:
                start_seg_idx = 0

            # 按顺序遍历所有 segment
            for offset in range(n_segs):
                seg_idx = (start_seg_idx + offset) % n_segs
                seg_indices = segments[seg_idx]

                # 输出这个 segment 的点
                seg_points = [points[i] for i in seg_indices]
                contour_segments.append(seg_points)

                # 检查这个 segment 的终点是否是通往其他 ring 的 MST 边
                end_idx = seg_indices[-1]
                for neighbor in mst.neighbors(ring_id):
                    if neighbor in visited_rings:
                        continue
                    key = (ring_id, neighbor)
                    if key not in edge_info:
                        key = (neighbor, ring_id)
                    _, my_split, neighbor_split = edge_info[(ring_id, neighbor)]
                    if my_split == end_idx:
                        # 跳转到 neighbor ring
                        dfs_traverse(neighbor, neighbor_split)

        # 从 ring 0 开始 DFS
        dfs_traverse(0, entry_split_idx=None)

        # 处理可能因图不连通而遗漏的 ring（安全兜底）
        for ring_id in range(n_rings):
            if ring_id not in visited_rings:
                pts = ring_points_list[ring_id]
                contour_segments.append(pts)

        return contour_segments

    # ──────────────────────────────────────────────
    #  6. 对一个 contour group 生成最终扫描序列（含 point skip 分组交错）
    # ──────────────────────────────────────────────
    def _route_contour_group_with_melt_groups(self, group):
        """
        文献 Algorithm 1 的核心实现:

        foreach melt group in melt group sequence do
            foreach contour segment in contour segment sequence do
                Append melt spots to spot melting sequence

        每个 contour segment 的点通过 point skip 被分为 num_melt_groups 组，
        先输出 melt_group 0 的所有 segment，再输出 melt_group 1 的所有 segment，依此类推。
        """
        # 6a. 在每条 ring 上采样点
        ring_points_list = []
        for ring in group['rings']:
            pts = self._sample_spots_on_ring(ring)
            if pts:
                ring_points_list.append(pts)

        if not ring_points_list:
            return []

        # 6b. 用 MST 确定 contour segment sequence
        contour_segments = self._build_contour_segment_sequence(ring_points_list)

        if not contour_segments:
            return []

        # 6c. 对每个 segment，按 point skip 分配 melt group
        #     segment 中第 i 个点的 melt_group = i % num_melt_groups
        #
        # 然后按 melt group 顺序输出:
        #   for mg in 0..7:
        #       for seg in segments:
        #           输出 seg 中属于 mg 的点

        sequence = []
        for melt_group_id in range(self.num_melt_groups):
            for seg_points in contour_segments:
                for i in range(melt_group_id, len(seg_points), self.num_melt_groups):
                    x, y = seg_points[i]
                    sequence.append({
                        'x': x, 'y': y,
                        'dwell_time': self.dwell_time,
                        'beam_current': self.beam_current,
                        'is_virtual': False,
                        'type': 'contour'
                    })

        return sequence

    # ──────────────────────────────────────────────
    #  7. 生成完整轮廓扫描序列（公开接口）
    # ──────────────────────────────────────────────
    def generate_full_contour_sequence(self):
        """
        文献 Algorithm 1 的完整流程:
        1. 生成 contour groups（按距离从内到外排列）
        2. 对每个 contour group:
           a. 在每条 ring 上采样点（点数为8的倍数）
           b. 用 MST 确定多条轮廓线的 contour segment sequence
           c. 按 8 个 melt group 交错输出（point skip）
        3. 依次拼接所有 contour group 的序列
        """
        print(f"  [轮廓生成] 文献策略: {self.num_lines}圈, "
              f"线距{self.line_spacing}mm, 点距≈{self.spot_spacing}mm, "
              f"{self.num_melt_groups}分组交错(point skip)")

        groups = self.generate_contour_groups()
        full_sequence = []
        total_spots = 0
        total_segments = 0

        for g_idx, group in enumerate(groups):
            n_rings = len(group['rings'])
            group_seq = self._route_contour_group_with_melt_groups(group)
            full_sequence.extend(group_seq)

            total_spots += len(group_seq)
            print(f"    Contour Group {g_idx + 1} (offset={group['distance']:.3f}mm, "
                  f"{n_rings}条轮廓线): {len(group_seq)} 个点")

        print(f"  -> 轮廓总点数: {total_spots}")
        return full_sequence
