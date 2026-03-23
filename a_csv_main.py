import os
import csv
import numpy as np
from shapely.geometry import Point
import matplotlib.pyplot as plt

from a_csv_contour_algo import ContourGenerator
from a_csv_infill_algo import (generate_substrate_spots, assign_groups,
                               filter_spots_in_geometry, generate_melting_path,
                               build_final_path_with_virtuals, insert_virtuals_for_path,
                               export_to_binary, visualize_result, generate_virtual_jump_points)

CURRENT_CHANGE_DWELL_US = 25.0
CURRENT_CHANGE_LINEAR_POINTS = 20
CURRENT_CHANGE_LINEAR_MA = 0.1
CURRENT_CHANGE_STABLE_TIME_US = 100_000.0
LAYER_INDEX = 1

def create_annulus_geometry(cx, cy, r_in, r_out):
    center = Point(cx, cy)
    outer_circle = center.buffer(r_out)
    if r_in > 0:
        inner_circle = center.buffer(r_in)
        return outer_circle.difference(inner_circle)
    return outer_circle

def load_experiment_csv(filename):
    configs = []
    print(f"\n[-] 正在读取实验参数表: {filename}")
    try:
        with open(filename, 'r', encoding='utf-8-sig') as f:
            reader = csv.reader(f)
            rows = list(reader)
            start_row = 1 if not rows[0][1].replace('.', '').replace('-', '').isdigit() else 0

            for row in rows[start_row:]:
                if len(row) < 12: continue
                configs.append({
                    'type': str(row[0]).strip(),
                    'cx': float(row[1]), 'cy': float(row[2]),
                    'in_curr': float(row[3]), 'in_dwell': float(row[4]),
                    'r_in': float(row[5]), 'r_out': float(row[6]),
                    'spot_dist': float(row[7]), 'n_groups': int(row[8]),
                    'ct_curr': float(row[9]), 'ct_dwell': float(row[10]),
                    'inner_proj': float(row[11])
                })
        print(f"✓ 成功加载 {len(configs)} 组环形实验配置")
        return configs
    except Exception as e:
        print(f"❌ 读取 CSV 失败: {e}")
        return []

def create_template_csv(filename="annulus_experiment.csv"):
    headers = ['点阵类型', '中心X', '中心Y', '束流(填充)', '停留时间(填充)', '内径', '半径(外径)', '点间距(mm)',
               '分组数', '束流(轮廓)', '停留时间(轮廓)', '内部投影偏移量']
    data = [
        ['sc', -30, 30, 10.0, 400, 10.0, 20.0, 1.0, 25, 8.0, 30, 0.380],
        ['sc', 30, 30, 10.0, 300, 10.0, 20.0, 1.0, 25, 8.0, 30, 0.350],
        ['sc', -30, -30, 10.0, 200, 10.0, 20.0, 1.0, 25, 8.0, 30, 0.300],
        ['sc', 30, -30, 10.0, 100, 10.0, 20.0, 1.0, 25, 8.0, 30, 0.250]
    ]
    with open(filename, 'w', newline='', encoding='utf-8-sig') as f:
        csv.writer(f).writerows([headers] + data)
    print(f"✓ 已在当前目录生成实验模板: {filename}")

def process_single_region(cfg, region_idx, j_max):
    print(f"\n" + "-" * 40)
    print(f"【处理区域 {region_idx + 1}】 坐标:({cfg['cx']}, {cfg['cy']}) | 投影偏移变量: {cfg['inner_proj']}mm")

    geom = create_annulus_geometry(cfg['cx'], cfg['cy'], cfg['r_in'], cfg['r_out'])
    square_size = cfg['r_out'] * 2
    base_spots = generate_substrate_spots(square_size, cfg['spot_dist'], cfg['type'], layer_index=0)
    spots = [(x + cfg['cx'], y + cfg['cy']) for x, y in base_spots]

    group_map, label_map = assign_groups(spots, cfg['n_groups'], cfg['spot_dist'], square_size, cfg['type'])

    raw_indices, _ = filter_spots_in_geometry(spots, geom)
    if not raw_indices:
        print("  [警告] 该几何体内未能生成有效的网格点。")
        return [], []

    contour_gen = ContourGenerator(
        geometry=geom,
        contour_dwell_time=cfg['ct_dwell'],
        contour_beam_current=cfg['ct_curr'],
        inner_projection_offset=cfg['inner_proj']
    )
    melting_indices = contour_gen.project_and_filter_infill(spots, raw_indices)
    melting_spots_set = set(melting_indices)

    infill_seq = generate_melting_path(spots, group_map, melting_spots_set, cfg['n_groups'], j_max)
    infill_path = build_final_path_with_virtuals(spots, infill_seq, cfg['in_dwell'], cfg['in_curr'], label_map)
    contour_path = contour_gen.generate_full_contour_sequence()

    final_infill = insert_virtuals_for_path(infill_path, j_max)
    final_contour = insert_virtuals_for_path(contour_path, j_max)

    return final_infill, final_contour

def append_path_with_jump(main_path, new_path, j_max):
    if not new_path:
        return
    if main_path:
        last_pt = main_path[-1]
        first_pt = new_path[0]
        dist = np.sqrt((first_pt['x'] - last_pt['x']) ** 2 + (first_pt['y'] - last_pt['y']) ** 2)
        if dist > j_max:
            v_pts = generate_virtual_jump_points((last_pt['x'], last_pt['y']), (first_pt['x'], first_pt['y']), j_max)
            for vp in v_pts:
                main_path.append({
                    'x': vp[0], 'y': vp[1],
                    'dwell_time': 25,
                    'beam_current': last_pt['beam_current'],
                    'is_virtual': True,
                    'type': 'virtual'
                })
    main_path.extend(new_path)


def make_transition_point(x, y, beam_current):
    return {
        'x': x,
        'y': y,
        'dwell_time': CURRENT_CHANGE_DWELL_US,
        'beam_current': beam_current,
        'is_virtual': True,
        'type': 'currentChanging'
    }


def generate_segmented_edge_points(start, end, step_dist):
    sx, sy = start
    ex, ey = end
    dist = np.sqrt((ex - sx) ** 2 + (ey - sy) ** 2)
    if dist < 1e-9:
        return [end]

    n_steps = max(1, int(np.ceil(dist / step_dist)))
    return [
        (sx + (ex - sx) * i / n_steps, sy + (ey - sy) * i / n_steps)
        for i in range(1, n_steps + 1)
    ]


def build_clockwise_square_vertices(start, end):
    sx, sy = start
    ex, ey = end
    center_x = (sx + ex) / 2.0
    center_y = (sy + ey) / 2.0
    half_dx = (ex - sx) / 2.0
    half_dy = (ey - sy) / 2.0

    perp_x = -half_dy
    perp_y = half_dx

    corner_b = (center_x + perp_x, center_y + perp_y)
    corner_d = (center_x - perp_x, center_y - perp_y)
    return [start, corner_b, end, corner_d]


def build_square_transition_coords(start, end, step_dist, target_count):
    start_pt, corner_b, end_pt, corner_d = build_clockwise_square_vertices(start, end)
    if target_count <= 0:
        target_count = 1

    coords = [start_pt]
    clockwise_edges = [
        (start_pt, corner_b),
        (corner_b, end_pt),
        (end_pt, corner_d),
        (corner_d, start_pt)
    ]

    while True:
        for edge_start, edge_end in clockwise_edges:
            coords.extend(generate_segmented_edge_points(edge_start, edge_end, step_dist))
            if len(coords) >= target_count and edge_end == end_pt:
                return coords


def build_current_change_profile(start_current, end_current):
    delta = end_current - start_current
    if abs(delta) < 1e-9:
        return []

    direction = 1.0 if delta > 0 else -1.0
    remaining = abs(delta)
    current = start_current
    profile = []

    while remaining > 1e-9:
        step = min(CURRENT_CHANGE_LINEAR_MA, remaining)
        current += direction * step
        profile.extend([round(current, 6)] * CURRENT_CHANGE_LINEAR_POINTS)
        remaining -= step

    return profile


def plan_current_change_path(contour_path, infill_path, jump_dist):
    if not contour_path or not infill_path:
        return [], {
            'required_us': 0.0,
            'square_us': 0.0,
            'extra_us': 0.0,
            'delta_ma': 0.0,
            'ramp_count': 0,
            'point_count': 0
        }

    start_pt = contour_path[-1]
    end_pt = infill_path[0]
    start = (start_pt['x'], start_pt['y'])
    end = (end_pt['x'], end_pt['y'])
    start_current = float(start_pt.get('beam_current', 0.0))
    end_current = float(end_pt.get('beam_current', 0.0))
    delta_ma = abs(end_current - start_current)
    ramp_profile = build_current_change_profile(start_current, end_current)

    if not ramp_profile:
        coords = build_square_transition_coords(start, end, jump_dist, target_count=1)
        return (
            [make_transition_point(x, y, end_current) for x, y in coords],
            {
                'required_us': 0.0,
                'square_us': len(coords) * CURRENT_CHANGE_DWELL_US,
                'extra_us': 0.0,
                'delta_ma': delta_ma,
                'ramp_count': 0,
                'point_count': len(coords)
            }
        )

    stable_points = int(np.ceil(CURRENT_CHANGE_STABLE_TIME_US / CURRENT_CHANGE_DWELL_US))
    stage_currents = list(ramp_profile)
    stage_currents.extend([end_current] * stable_points)

    required_count = len(stage_currents)
    coords = build_square_transition_coords(start, end, jump_dist, required_count)
    square_us = len(coords) * CURRENT_CHANGE_DWELL_US

    if len(stage_currents) < len(coords):
        stage_currents.extend([end_current] * (len(coords) - len(stage_currents)))

    transition_path = [
        make_transition_point(x, y, beam_current)
        for (x, y), beam_current in zip(coords, stage_currents)
    ]

    return transition_path, {
        'required_us': required_count * CURRENT_CHANGE_DWELL_US,
        'square_us': square_us,
        'extra_us': max(0.0, square_us - required_count * CURRENT_CHANGE_DWELL_US),
        'delta_ma': delta_ma,
        'ramp_count': len(ramp_profile),
        'point_count': len(transition_path)
    }

def main():
    print("=" * 60)
    print("  多变量环形阵列点扫控制台 (全局宏观扫描时序版 v2)")
    print("=" * 60)

    print("\n[选项] 1. 运行已有 CSV 实验表   2. 生成全新 4组测试 CSV 模板")
    choice = input("请选择 (默认1): ").strip() or '1'
    if choice == '2':
        create_template_csv()
        return

    csv_name = input("\n请输入 CSV 文件名 (默认 annulus_experiment.csv): ").strip() or "annulus_experiment.csv"
    if not csv_name.endswith('.csv'): csv_name += '.csv'

    configs = load_experiment_csv(csv_name)
    if not configs: return

    j_max = float(input("全局最大跳跃距离 j_max (mm, 默认 2.0): ").strip() or 2.0)
    sub_size = float(input("用于可视化图表的基板尺寸 (mm, 默认 150): ").strip() or 150)

    global_infill_path = []
    global_contour_path = []

    for i, cfg in enumerate(configs):
        infill_region, contour_region = process_single_region(cfg, region_idx=i, j_max=j_max)
        append_path_with_jump(global_infill_path, infill_region, j_max)
        append_path_with_jump(global_contour_path, contour_region, j_max)

    print("\n[-] 正在执行全局时序组装: 所有轮廓 -> 电流切换 -> 所有填充")
    current_change_path, transition_meta = plan_current_change_path(global_contour_path, global_infill_path, j_max)

    global_path = []
    global_path.extend(global_contour_path)
    global_path.extend(current_change_path)
    global_path.extend(global_infill_path)

    contour_count = sum(1 for p in global_path if p.get('type') == 'contour')
    current_change_count = sum(1 for p in global_path if p.get('type') == 'currentChanging')
    infill_count = sum(1 for p in global_path if p.get('type') == 'infill')
    virt_count = sum(1 for p in global_path if p.get('is_virtual'))
    print(f"✓ 实验矩阵生成完毕！总路径点数: {len(global_path)} "
          f"(轮廓: {contour_count} | 电流切换: {current_change_count} | 填充: {infill_count} | 虚拟/缓冲: {virt_count})")
    print(f"  [轮廓->填充] 束流变化: {transition_meta['delta_ma']:.3f} mA | "
          f"方形路径时间: {transition_meta['square_us'] / 1000.0:.3f} ms | "
          f"所需缓冲时间: {transition_meta['required_us'] / 1000.0:.3f} ms")
    if transition_meta['ramp_count'] > 0:
        print(f"  [轮廓->填充] 线性变流点数: {transition_meta['ramp_count']} | "
              f"超出需求时间: {transition_meta['extra_us'] / 1000.0:.3f} ms")

    if input("\n导出二进制加工文件? (y/n, 默认y): ").strip().lower() != 'n':
        contour_name = f"layer_{LAYER_INDEX}-gun_0-1-contour"
        current_name = f"layer_{LAYER_INDEX}-gun_0-1-currentChanging"
        fill_name = f"layer_{LAYER_INDEX}-gun_0-1-fill"
        export_to_binary(global_contour_path, contour_name)
        if current_change_path:
            export_to_binary(current_change_path, current_name)
        export_to_binary(global_infill_path, fill_name)

    if input("生成结果可视化图片? (y/n, 默认y): ").strip().lower() != 'n':
        viz_name = input("图片名 (默认 experiment_result): ").strip() or "experiment_result"
        visualize_result(sub_size, None, [], set(), global_path, z_val=0, filename=viz_name)

if __name__ == "__main__":
    main()
