import warnings
import struct
import os
import numpy as np
import matplotlib
import csv
from scipy.spatial import cKDTree
from matplotlib.patches import Rectangle, Circle
from matplotlib.collections import PatchCollection
import matplotlib.pyplot as plt

# ==========================================
# 设置
# ==========================================
try:
    matplotlib.use('TkAgg')
except:
    pass

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial']
plt.rcParams['axes.unicode_minus'] = False
warnings.filterwarnings('ignore', message='.*Glyph.*missing from font.*')

# ==========================================
# 特定分散熔化顺序字典 (对应 4x4, 5x5, 6x6, 7x7, 8x8)
# ==========================================
MELT_ORDERS = {
    4: [
        [0, 8, 5, 9],
        [4, 12, 1, 13],
        [7, 15, 6, 10],
        [3, 11, 2, 14]
    ],
    5: [
        [0, 21, 17, 13, 9],
        [14, 5, 1, 22, 18],
        [23, 19, 10, 6, 2],
        [7, 3, 24, 15, 11],
        [16, 12, 8, 4, 20]
    ],
    6: [
        [0, 21, 2, 19, 34, 17],
        [30, 15, 32, 13, 28, 11],
        [24, 9, 26, 7, 22, 5],
        [18, 3, 20, 1, 16, 35],
        [12, 33, 14, 31, 10, 29],
        [6, 27, 8, 25, 4, 23]
    ],
    7: [
        [0, 43, 37, 31, 25, 19, 13],
        [20, 7, 1, 44, 38, 32, 26],
        [33, 27, 14, 8, 2, 45, 39],
        [46, 40, 34, 21, 15, 9, 3],
        [10, 4, 47, 41, 28, 22, 16],
        [23, 17, 11, 5, 48, 35, 29],
        [36, 30, 24, 18, 12, 6, 42]
    ],
    8: [
        [0, 25, 57, 45, 14, 39, 5, 35],
        [50, 37, 4, 31, 60, 27, 53, 17],
        [62, 23, 52, 19, 48, 7, 41, 29],
        [44, 11, 40, 1, 34, 22, 56, 15],
        [32, 59, 26, 13, 46, 10, 36, 3],
        [18, 6, 38, 63, 28, 58, 24, 49],
        [30, 54, 21, 51, 16, 43, 8, 61],
        [12, 42, 9, 33, 2, 55, 20, 47]
    ]
}


# ==========================================
# PART 1: 核心生成逻辑 (支持 SC & HCP)
# ==========================================

def generate_spots_integrated(config, layer_index=0):
    pattern_type = config['type'].lower()
    cx, cy = config['center_x'], config['center_y']
    w, h = config['width'], config['height']
    dx = config['spot_distance']

    if dx > min(w, h):
        print(f"    [警告] 区域{config.get('region_id', '?')} 点间距({dx}) > 边长({min(w, h)})，可能仅生成1个点或0个点。")

    spots = []
    eps = 0.001
    x_min, x_max = cx - w / 2.0, cx + w / 2.0
    y_min, y_max = cy - h / 2.0, cy + h / 2.0

    if 'hcp' in pattern_type:
        dy = dx * np.sqrt(3) / 2.0
        is_layer_b = (layer_index % 2 == 1)

        layer_offset_x = 0.5 * dx if is_layer_b else 0.0
        layer_offset_y = (dx * np.sqrt(3) / 6.0) if is_layer_b else 0.0

        y_start_gen = y_min - dy * 2
        y_end_gen = y_max + dy * 2
        raw_y_coords = np.arange(y_start_gen, y_end_gen, dy)

        for row_idx, raw_y in enumerate(raw_y_coords):
            y = raw_y + layer_offset_y
            if not (y_min - eps <= y <= y_max + eps):
                continue

            row_stagger = 0.5 * dx if (row_idx % 2 == 1) else 0.0
            total_x_offset = row_stagger + layer_offset_x

            current_row_x_min = x_min - dx * 2
            current_row_x_max = x_max + dx * 2
            raw_x_coords = np.arange(current_row_x_min, current_row_x_max, dx)
            x_coords_shifted = raw_x_coords + total_x_offset

            for x in x_coords_shifted:
                if x_min - eps <= x <= x_max + eps:
                    spots.append((x, y))
    else:
        dy = dx
        raw_x = np.arange(x_min, x_max + eps / 100, dx)
        raw_y = np.arange(y_min, y_max + eps / 100, dy)

        for x in raw_x:
            for y in raw_y:
                if (x_min - eps <= x <= x_max + eps) and (y_min - eps <= y <= y_max + eps):
                    spots.append((x, y))

    return spots


def assign_groups(spots, n_groups, spot_distance, pattern_type='sc'):
    if len(spots) == 0: return {}, {}
    group_map, label_map = {}, {}
    letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'

    block_size = max(1, int(np.sqrt(n_groups)))
    dx = spot_distance
    dy = spot_distance * np.sqrt(3) / 2.0 if 'hcp' in pattern_type.lower() else spot_distance

    min_x = min(p[0] for p in spots)
    min_y = min(p[1] for p in spots)
    max_x = max(p[0] for p in spots)
    points_per_line = int(round((max_x - min_x) / dx)) + 1

    for idx, (x, y) in enumerate(spots):
        i_idx = int(round((x - min_x) / dx))
        j_idx = int(round((y - min_y) / dy))

        block_i, block_j = i_idx // block_size, j_idx // block_size
        local_i, local_j = i_idx % block_size, j_idx % block_size

        # === 应用分散熔化顺序矩阵 ===
        if block_size in MELT_ORDERS:
            row_index = block_size - 1 - local_j
            col_index = local_i
            # 防止索引边界意外溢出
            if 0 <= row_index < block_size and 0 <= col_index < block_size:
                group_id = MELT_ORDERS[block_size][row_index][col_index]
            else:
                group_id = (local_j * block_size + local_i) % n_groups
        else:
            group_id = (local_j * block_size + local_i) % n_groups

        # === 组外块编号为字母 ===
        blocks_per_row = (points_per_line + block_size - 1) // block_size
        block_num = block_j * blocks_per_row + block_i

        block_letter = ""
        temp_num = block_num
        while True:
            block_letter = letters[temp_num % 26] + block_letter
            temp_num = temp_num // 26 - 1
            if temp_num < 0:
                break

        group_map[idx] = group_id
        # 新标签格式：例如 A0 (A块的第0号点)
        label_map[idx] = f"{block_letter}{group_id}"

    return group_map, label_map


# ==========================================
# PART 2: 路径规划 (区域内)
# ==========================================

def build_graph_optimized(spots, group_spots, j_max):
    n_spots = len(group_spots)
    if n_spots == 0: return np.zeros((0, 0), dtype=int)
    coords = np.array([spots[idx] for idx in group_spots])
    tree = cKDTree(coords)
    pairs = tree.query_pairs(r=j_max, output_type='ndarray')
    adj = np.zeros((n_spots, n_spots), dtype=int)
    if len(pairs) > 0:
        for i, j in pairs: adj[i][j] = adj[j][i] = 1
    return adj


def heuristic_dfs(adj_matrix, start_idx=0):
    n = adj_matrix.shape[0]
    if n == 0: return []
    visited = [False] * n
    path = [start_idx]
    visited[start_idx] = True

    def get_flexibility(v, visited_status):
        return sum(1 for i in range(n) if adj_matrix[v][i] == 1 and not visited_status[i])

    max_attempts = n * 10
    attempts = 0
    while len(path) < n and attempts < max_attempts:
        attempts += 1
        current = path[-1]
        neighbors = [i for i in range(n) if adj_matrix[current][i] == 1 and not visited[i]]
        if neighbors:
            neighbors.sort(key=lambda x: get_flexibility(x, visited))
            next_v = neighbors[0]
            path.append(next_v)
            visited[next_v] = True
            attempts = 0
        else:
            found = False
            for i in range(len(path) - 1, -1, -1):
                v = path[i]
                potential = [j for j in range(n) if adj_matrix[v][j] == 1 and not visited[j]]
                if potential:
                    path = path[:i + 1]
                    found = True
                    break
            if not found: break
    return path


def generate_region_path(spots, group_map, n_groups, j_min, j_max):
    sequence = []
    for group_id in range(n_groups):
        group_spots = [idx for idx, gid in group_map.items() if gid == group_id]
        if not group_spots: continue

        if not sequence:
            start_spot = min(group_spots, key=lambda i: spots[i][0] + spots[i][1])
        else:
            last_coord = spots[sequence[-1]]
            candidates = []
            for idx in group_spots:
                d = np.linalg.norm(np.array(last_coord) - np.array(spots[idx]))
                if j_min <= d <= j_max: candidates.append((abs(d - j_max), idx))
            if candidates:
                candidates.sort()
                start_spot = candidates[0][1]
            else:
                start_spot = min(group_spots, key=lambda i: np.linalg.norm(np.array(last_coord) - np.array(spots[i])))

        start_idx = group_spots.index(start_spot)
        adj = build_graph_optimized(spots, group_spots, j_max)
        local_path = heuristic_dfs(adj, start_idx)
        sequence.extend([group_spots[i] for i in local_path])
    return sequence


def process_single_region(config, region_id, global_max_jump, layer_index):
    config['region_id'] = region_id + 1
    print(f"\n  【区域{region_id + 1}】 {config['type'].upper()} | 尺寸: {config['width']}x{config['height']}")

    spots = generate_spots_integrated(config, layer_index)
    print(f"    -> 生成 {len(spots)} 个网格点")

    if not spots:
        return {'path_coords': [], 'path_labels': [], 'dwell_time': 0, 'center_x': config['center_x'],
                'center_y': config['center_y'],
                'width': config['width'], 'height': config['height'], 'config': config}

    group_map, label_map = assign_groups(spots, config['n_groups'], config['spot_distance'], config['type'])
    block_size = max(1, int(np.sqrt(config['n_groups'])))
    j_min = min(block_size * config['spot_distance'], global_max_jump)
    if j_min > global_max_jump: j_min = config['spot_distance']

    sequence = generate_region_path(spots, group_map, config['n_groups'], j_min, global_max_jump)
    path_coords = [spots[i] for i in sequence]
    path_labels = [label_map[i] for i in sequence]

    return {
        'region_id': region_id, 'config': config,
        'center_x': config['center_x'], 'center_y': config['center_y'],
        'width': config['width'], 'height': config['height'],
        'path_coords': path_coords,
        'path_labels': path_labels,  # 返回给可视化系统用于画出文本标签
        'dwell_time': config['dwell_time'], 'spot_distance': config['spot_distance']
    }


# ==========================================
# PART 3: 全局排序与连接
# ==========================================

def determine_region_order_nearest(regions_data):
    valid_indices = [i for i, rd in enumerate(regions_data) if len(rd['path_coords']) > 0]
    if not valid_indices: return []

    order = []
    unvisited = set(valid_indices)

    start_idx = 0 if 0 in unvisited else valid_indices[0]
    order.append(start_idx)
    unvisited.remove(start_idx)

    while unvisited:
        curr = order[-1]
        last_pt = regions_data[curr]['path_coords'][-1]

        dists = []
        for i in unvisited:
            first_pt = regions_data[i]['path_coords'][0]
            d = (first_pt[0] - last_pt[0]) ** 2 + (first_pt[1] - last_pt[1]) ** 2
            dists.append((d, i))

        dists.sort()
        next_idx = dists[0][1]
        order.append(next_idx)
        unvisited.remove(next_idx)
    return order


def determine_region_order_farthest(regions_data):
    valid_indices = [i for i, rd in enumerate(regions_data) if len(rd['path_coords']) > 0]
    if not valid_indices: return []
    order = []
    unvisited = set(valid_indices)
    start_idx = 0 if 0 in unvisited else valid_indices[0]
    order.append(start_idx)
    unvisited.remove(start_idx)

    while unvisited:
        curr = order[-1]
        last_pt = regions_data[curr]['path_coords'][-1]
        dists = []
        for i in unvisited:
            first_pt = regions_data[i]['path_coords'][0]
            d = (first_pt[0] - last_pt[0]) ** 2 + (first_pt[1] - last_pt[1]) ** 2
            dists.append((d, i))
        dists.sort(reverse=True)
        next_idx = dists[0][1]
        order.append(next_idx)
        unvisited.remove(next_idx)
    return order


def generate_virtual_jump_points(start, end, max_dist):
    sx, sy = start
    ex, ey = end
    dist = np.sqrt((ex - sx) ** 2 + (ey - sy) ** 2)
    if dist <= max_dist + 0.001: return []

    virtuals = []
    curr = max_dist
    dx, dy = (ex - sx) / dist, (ey - sy) / dist
    while curr < dist - 0.001:
        virtuals.append((sx + dx * curr, sy + dy * curr))
        curr += max_dist
    return virtuals


def build_global_path(regions_data, region_order, max_jump_distance):
    global_path = []
    print("\n  构建全局路径...")
    for i, r_idx in enumerate(region_order):
        rd = regions_data[r_idx]
        pts = rd['path_coords']
        if not pts: continue

        if global_path:
            prev = global_path[-1]
            v_pts = generate_virtual_jump_points((prev[0], prev[1]), pts[0], max_jump_distance)
            for vp in v_pts:
                global_path.append((vp[0], vp[1], 0, True, -1))

        for p in pts:
            global_path.append((p[0], p[1], rd['dwell_time'], False, r_idx))
    return global_path


# ==========================================
# PART 4: 二进制导出
# ==========================================

def export_to_binary_integrated(global_path, output_filename, beam_current, focus_val):
    print(f"\n开始生成二进制文件: {output_filename}")
    segments = []
    if not global_path: return False
    prev_x, prev_y = global_path[0][0], global_path[0][1]

    if not global_path[0][3]:
        num_lines = int(round(global_path[0][2] / 25.0))
        for _ in range(num_lines):
            segments.append((prev_x, prev_y, prev_x + 0.001, prev_y))

    for i in range(1, len(global_path)):
        curr_x, curr_y, dwell, is_virtual, _ = global_path[i]

        if abs(curr_x - prev_x) > 0.0001 or abs(curr_y - prev_y) > 0.0001:
            segments.append((prev_x, prev_y, curr_x, curr_y))

        if not is_virtual:
            num_lines = int(round(dwell / 25.0))
            for _ in range(num_lines):
                segments.append((curr_x, curr_y, curr_x + 0.001, curr_y))

        prev_x, prev_y = curr_x, curr_y

    total_vertices = len(segments) * 2
    try:
        if not output_filename.endswith('.bin'): output_filename += '.bin'
        with open(output_filename, 'wb') as f:
            f.write(struct.pack('dd', 6.0, float(total_vertices)))
            for seg in segments:
                f.write(struct.pack('6d', seg[0], seg[1], focus_val, beam_current, 0.0, 0.0))
                f.write(struct.pack('6d', seg[2], seg[3], focus_val, beam_current, 0.0, 0.0))
        print(f"✓ 二进制文件生成成功: {output_filename}")
        return True
    except Exception as e:
        print(f"❌ 错误: {e}")
        return False


# ==========================================
# PART 5: 可视化
# ==========================================

def visualize_multi_region(substrate_size, regions_data, region_order,
                           global_path, max_jump_distance, layer_idx=0, filename="result"):
    print("\n生成可视化预览...")
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.set_aspect('equal')
    half = substrate_size / 2.0
    ax.set_xlim(-half - 5, half + 5)
    ax.set_ylim(-half - 5, half + 5)
    ax.grid(True, alpha=0.3)

    layer_name = 'B' if layer_idx % 2 == 1 else 'A'
    ax.set_title(f'Spot Melting Path - {layer_name}层 (Jump Limit: {max_jump_distance}mm)')

    ax.add_patch(Rectangle((-half, -half), substrate_size, substrate_size, fill=False, edgecolor='gray', linewidth=2))

    colors = plt.cm.tab10.colors

    for idx, r_idx in enumerate(region_order):
        rd = regions_data[r_idx]
        color = colors[r_idx % len(colors)]
        w, h = rd['width'], rd['height']
        cx, cy = rd['center_x'], rd['center_y']

        ax.add_patch(
            Rectangle((cx - w / 2, cy - h / 2), w, h, fill=False, edgecolor=color, linestyle='--', linewidth=1))

        n_pts = len(rd['path_coords'])
        label_text = f"R{r_idx + 1}\n{rd['config']['type']}\n间距:{rd['config']['spot_distance']}\n停留:{rd['config']['dwell_time']}"
        ax.text(cx, cy + h / 2 + 2, label_text, ha='center', va='bottom', color=color, fontweight='bold', fontsize=8)

        xs = [p[0] for p in rd['path_coords']]
        ys = [p[1] for p in rd['path_coords']]
        lbls = rd.get('path_labels', [])

        if xs:
            radius = 0.5 if len(xs) < 20 else 0.25
            patches = [Circle((x, y), radius=radius) for x, y in zip(xs, ys)]
            p = PatchCollection(patches, facecolors=color, edgecolors='none', alpha=0.6)
            ax.add_collection(p)

            # --- 核心可视化增强：点数不爆炸时，将 "A0" 这种序号直接写在图片对应位置 ---
            if len(xs) <= 200:
                for x, y, lbl in zip(xs, ys, lbls):
                    # 为了看清楚，圆圈变小一点时使用黑色文字，较大圆圈使用白色文字
                    ax.text(x, y, lbl, ha='center', va='center',
                            fontsize=6, color='black', fontweight='semibold')

            ax.plot(xs, ys, '-', color=color, alpha=0.3, linewidth=0.5)

    virtual_x = [p[0] for p in global_path if p[3]]
    virtual_y = [p[1] for p in global_path if p[3]]

    jump_x, jump_y = [], []
    for i in range(len(global_path) - 1):
        curr = global_path[i]
        next_p = global_path[i + 1]

        if curr[3] or next_p[3] or (curr[4] != next_p[4]):
            jump_x.extend([curr[0], next_p[0], None])
            jump_y.extend([curr[1], next_p[1], None])

    ax.plot(jump_x, jump_y, 'r--', linewidth=0.8, alpha=0.5, label='Jump Path')

    if virtual_x:
        ax.scatter(virtual_x, virtual_y, c='red', marker='x', s=20, label='Virtual Points', zorder=10)

    plt.savefig(f"{filename}.png", dpi=200, bbox_inches='tight')
    print(f"✓ 图片已保存: {filename}.png")
    plt.show()


# ==========================================
# PART 6: 主程序
# ==========================================

def create_template_csv(filename="region_config_integrated.csv"):
    headers = ['点阵类型(sc/hcp)', '中心X', '中心Y', '宽W', '高H', '点间距', '停留时间(μs)', '分组数']
    data = [headers, ['sc', 0, 50, 30, 30, 1.0, 600, 25]]  # 默认给25组触发5x5
    try:
        with open(filename, 'w', newline='', encoding='utf-8-sig') as f:
            csv.writer(f).writerows(data)
        print(f"模板已创建: {filename}")
    except:
        pass


def load_regions_from_csv(filename):
    configs = []
    try:
        with open(filename, 'r', encoding='utf-8-sig') as f:
            reader = csv.reader(f)
            rows = list(reader)
            start_row = 1 if rows and not rows[0][1].replace('.', '').replace('-', '').isdigit() else 0

            for i, row in enumerate(rows[start_row:]):
                if len(row) < 6: continue
                try:
                    if row[0].replace('.', '').replace('-', '').isdigit():
                        configs.append({
                            'type': 'sc',
                            'center_x': float(row[0]), 'center_y': float(row[1]),
                            'width': float(row[2]), 'height': float(row[2]),
                            'spot_distance': float(row[3]),
                            'dwell_time': float(row[4]) if len(row) > 4 else 600,
                            'n_groups': int(row[5]) if len(row) > 5 else 16
                        })
                    else:
                        configs.append({
                            'type': row[0].strip(),
                            'center_x': float(row[1]), 'center_y': float(row[2]),
                            'width': float(row[3]), 'height': float(row[4]),
                            'spot_distance': float(row[5]),
                            'dwell_time': float(row[6]) if len(row) > 6 else 600,
                            'n_groups': int(row[7]) if len(row) > 7 else 16
                        })
                except ValueError:
                    pass
        print(f"✓ 已加载 {len(configs)} 个区域配置")
        return configs
    except Exception as e:
        print(f"读取CSV失败: {e}")
        return None


def main():
    print("=" * 60)
    print("  多模式点扫路径生成器 v5 (SC/HCP 模式 + 离散矩阵优化)")
    print("=" * 60)

    try:
        sub_size = float(input("\n基板边长(mm, 默认200): ") or 200)
        max_jump = float(input("最大跳转距离(mm, 默认9): ") or 9)
        layer_in = input("HCP层索引 (0=A, 1=B, 默认0): ")
        layer_idx = int(layer_in) if layer_in.strip() else 0
    except:
        sub_size, max_jump, layer_idx = 200, 9, 0

    print("\n[区域配置] 1.手动输入 2.读取CSV 3.生成CSV模板")
    choice = input("选择: ").strip() or '1'

    configs = []
    if choice == '3':
        create_template_csv();
        return
    elif choice == '2':
        fname = input("CSV文件名: ").strip()
        if not fname.endswith('.csv'): fname += '.csv'
        configs = load_regions_from_csv(fname)
    else:
        print("请使用CSV模式以获得最佳体验")
        return

    if not configs: return

    regions_data = []
    for i, cfg in enumerate(configs):
        regions_data.append(process_single_region(cfg, i, max_jump, layer_idx))

    print("\n[排序策略]")
    print("1. 最近邻 (顺序扫描，图像整洁，推荐)")
    print("2. 最远距离 (分散热量，图像会有大量交叉线)")
    sort_choice = input("选择 (默认1): ").strip() or '1'

    if sort_choice == '2':
        order = determine_region_order_farthest(regions_data)
    else:
        order = determine_region_order_nearest(regions_data)

    if not order:
        print("无有效路径生成，请检查CSV数据。")
        return

    global_path = build_global_path(regions_data, order, max_jump)

    print("-" * 30)
    if input("导出二进制文件? (y/n, 默认y): ").lower() != 'n':
        export_to_binary_integrated(global_path, input("文件名: ") or "final_path", 0, 0)

    if input("生成可视化图片? (y/n, 默认y): ").lower() != 'n':
        visualize_multi_region(sub_size, regions_data, order, global_path, max_jump, layer_idx,
                               input("图片名: ") or "result")

    print("\n完成!")


if __name__ == "__main__":
    main()