import numpy as np
from shapely.geometry import Polygon, MultiPolygon, Point
from shapely.ops import nearest_points


class ContourGenerator:
    def __init__(self, geometry, contour_spot_spacing, contour_line_spacing,
                 contour_dwell_time, contour_beam_current):
        self.geometry = geometry
        self.spot_spacing = contour_spot_spacing
        self.line_spacing = contour_line_spacing
        self.dwell_time = contour_dwell_time
        self.beam_current = contour_beam_current

        # 文献基准实验参数
        self.num_lines = 3
        self.num_melt_groups = 8
        self.contour_offset = 0.35
        self.outer_projection = 0.05
        self.inner_projection = 0.35

    def _extract_rings(self, geom):
        rings = []
        if isinstance(geom, Polygon):
            rings.append(geom.exterior)
            rings.extend(geom.interiors)
        elif isinstance(geom, MultiPolygon):
            for poly in geom.geoms:
                rings.append(poly.exterior)
                rings.extend(poly.interiors)
        return rings

    def project_and_filter_infill(self, spots, melting_indices):
        """就地修改全局点阵，实现文献的投影策略 (Arrangement B6)"""
        print("\n  [配合逻辑] 执行 Infill 投影与过冲过滤 (Arrangement B6)...")
        inner_bound = self.geometry.buffer(-self.inner_projection)
        outer_bound = self.geometry.buffer(-self.outer_projection)

        if inner_bound.is_empty:
            print("  [警告] 几何体太小，无法生成安全偏移！")
            return melting_indices

        target_lines = self._extract_rings(inner_bound)
        new_indices = []
        projected_count, discarded_count = 0, 0

        for idx in melting_indices:
            x, y = spots[idx]
            pt = Point(x, y)

            if inner_bound.contains(pt):
                # 安全区域，保留原样
                new_indices.append(idx)
            elif outer_bound.contains(pt):
                # 处于投影区，投影到轮廓基准线
                min_dist = float('inf')
                best_proj = pt
                for ring in target_lines:
                    p1, p2 = nearest_points(pt, ring)
                    dist = pt.distance(p2)
                    if dist < min_dist:
                        min_dist = dist
                        best_proj = p2

                # ★ 就地修改原基板点阵坐标
                spots[idx] = (best_proj.x, best_proj.y)
                new_indices.append(idx)
                projected_count += 1
            else:
                # 过冲点，直接剔除
                discarded_count += 1

        print(
            f"  -> 原填充点: {len(melting_indices)} | 投影点: {projected_count} | 剔除过冲点: {discarded_count} | 最终保留: {len(new_indices)}")
        return new_indices

    def generate_contour_groups(self):
        groups = []
        for i in range(self.num_lines):
            offset = self.contour_offset - (self.num_lines - 1 - i) * self.line_spacing
            if offset > 0:
                bg = self.geometry.buffer(-offset)
                if not bg.is_empty:
                    rings = self._extract_rings(bg)
                    if rings: groups.append({'distance': offset, 'rings': rings})
        # 降序排序：确保先打距离最远（最靠近内部）的轮廓线
        groups.sort(key=lambda x: x['distance'], reverse=True)
        return groups

    def route_contour_group(self, group):
        sequence = []
        for ring in group['rings']:
            length = ring.length
            # 强制点数为 num_melt_groups 的整数倍
            num_spots = max(self.num_melt_groups,
                            int(np.ceil((length / self.spot_spacing) / self.num_melt_groups) * self.num_melt_groups))
            spacing = length / num_spots
            for i in range(num_spots):
                pt = ring.interpolate(i * spacing)
                sequence.append({
                    'x': pt.x, 'y': pt.y, 'dwell_time': self.dwell_time,
                    'beam_current': self.beam_current, 'is_virtual': False, 'type': 'contour'
                })
        return sequence

    def generate_full_contour_sequence(self):
        print("\n  [Contour] 开始生成独立轮廓线...")
        groups = self.generate_contour_groups()
        full_seq = []
        for g in groups:
            full_seq.extend(self.route_contour_group(g))
        return full_seq