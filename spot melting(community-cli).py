import warnings
import struct
import os
import numpy as np
import matplotlib
from scipy.spatial import cKDTree
from scipy.sparse import lil_matrix
from matplotlib.patches import Rectangle, Circle
from matplotlib.collections import PatchCollection
import matplotlib.pyplot as plt
import shapely
from shapely.geometry import Polygon, Point, MultiPolygon
from shapely.prepared import prep
from tqdm import tqdm
import networkx as nx
from numba import njit

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
# 0. 分散熔化顺序矩阵
# ==========================================
MELT_ORDERS = {
    4: [[0, 8, 5, 9], [4, 12, 1, 13], [7, 15, 6, 10], [3, 11, 2, 14]],
    5: [[0, 21, 17, 13, 9], [14, 5, 1, 22, 18], [23, 19, 10, 6, 2],
        [7, 3, 24, 15, 11], [16, 12, 8, 4, 20]],
    6: [[0, 21, 2, 19, 34, 17], [30, 15, 32, 13, 28, 11],
        [24, 9, 26, 7, 22, 5], [18, 3, 20, 1, 16, 35],
        [12, 33, 14, 31, 10, 29], [6, 27, 8, 25, 4, 23]],
    7: [[0, 43, 37, 31, 25, 19, 13], [20, 7, 1, 44, 38, 32, 26],
        [33, 27, 14, 8, 2, 45, 39], [46, 40, 34, 21, 15, 9, 3],
        [10, 4, 47, 41, 28, 22, 16], [23, 17, 11, 5, 48, 35, 29],
        [36, 30, 24, 18, 12, 6, 42]],
    8: [[0, 25, 57, 45, 14, 39, 5, 35], [50, 37, 4, 31, 60, 27, 53, 17],
        [62, 23, 52, 19, 48, 7, 41, 29], [44, 11, 40, 1, 34, 22, 56, 15],
        [32, 59, 26, 13, 46, 10, 36, 3], [18, 6, 38, 63, 28, 58, 24, 49],
        [30, 54, 21, 51, 16, 43, 8, 61], [12, 42, 9, 33, 2, 55, 20, 47]]
}


# ==========================================
# 1. CLI 切片读取器
# ==========================================
class CLIReader:
    def __init__(self, filepath):
        self.filepath = filepath
        self.layers = {}
        self.units = 1.0
        self.sorted_z_levels = []

    def read(self):
        print(f"[-] 正在读取文件: {self.filepath}")
        if not os.path.exists(self.filepath):
            print(f"[!] 错误：文件不存在 {self.filepath}")
            return False
        try:
            with open(self.filepath, 'rb') as f:
                header_bytes = b""
                while True:
                    char = f.read(1)
                    if not char:
                        break
                    header_bytes += char
                    if b"$$HEADEREND" in header_bytes:
                        break
                try:
                    h_str = header_bytes.decode('ascii', errors='ignore')
                    for line in h_str.splitlines():
                        if '$$UNITS' in line:
                            self.units = float(line.split('/')[1])
                            print(f"[-] 单位系数: {self.units}")
                except:
                    pass
                buffer = f.read(1024)
                start_offset = buffer.find(b'\x7f\x00')
                if start_offset == -1:
                    print("[!] 未找到有效的二进制数据标记")
                    return False
                f.seek(len(header_bytes) + start_offset)
                current_z = 0.0
                while True:
                    cmd_bytes = f.read(2)
                    if len(cmd_bytes) < 2:
                        break
                    cmd = struct.unpack('<H', cmd_bytes)[0]
                    if cmd == 127:
                        z_bytes = f.read(4)
                        current_z = struct.unpack('<f', z_bytes)[0] * self.units
                        self.layers[current_z] = []
                    elif cmd == 130:
                        meta = f.read(12)
                        pid, pdir, n_points = struct.unpack('<III', meta)
                        coord_bytes = f.read(n_points * 2 * 4)
                        floats = struct.unpack(f'<{n_points * 2}f', coord_bytes)
                        coords = [(floats[i] * self.units, floats[i + 1] * self.units)
                                  for i in range(0, len(floats), 2)]
                        if current_z in self.layers:
                            self.layers[current_z].append({'dir': pdir, 'coords': coords})
                    elif cmd == 132:
                        meta = f.read(8)
                        _, n = struct.unpack('<II', meta)
                        f.seek(n * 4 * 4, 1)
                    elif cmd == 0:
                        continue
            self.sorted_z_levels = sorted(self.layers.keys())
            print(f"[-] 解析成功: 共 {len(self.layers)} 层")
            return True
        except Exception as e:
            print(f"[!] 读取失败: {e}")
            import traceback
            traceback.print_exc()
            return False

    def get_layer_by_index(self, index):
        if 0 <= index < len(self.sorted_z_levels):
            z = self.sorted_z_levels[index]
            return z, self.layers[z]
        return None, None

    def get_all_z_levels(self):
        return self.sorted_z_levels


# ==========================================
# 2. 几何处理工具
# ==========================================
def build_geometry_from_layer(layer_data):
    polys = []
    for contour in layer_data:
        coords = contour['coords']
        if len(coords) >= 3:
            try:
                poly = Polygon(coords)
                if poly.is_valid:
                    polys.append(poly)
                else:
                    poly = poly.buffer(0)
                    if poly.is_valid and not poly.is_empty:
                        polys.append(poly)
            except:
                pass
    if not polys:
        return None
    combined = Polygon()
    for p in polys:
        combined = combined.symmetric_difference(p)
    return combined


def get_geometry_bounds(geometry):
    if geometry is None or geometry.is_empty:
        return None
    minx, miny, maxx, maxy = geometry.bounds
    return {
        'min_x': minx, 'max_x': maxx,
        'min_y': miny, 'max_y': maxy,
        'width': maxx - minx, 'height': maxy - miny,
        'center_x': (minx + maxx) / 2, 'center_y': (miny + maxy) / 2
    }


# ==========================================
# 3. 基板点阵生成
# ==========================================
def generate_substrate_spots(square_size, spot_distance, pattern_type='sc', layer_index=0):
    half_size = square_size / 2.0
    dx = spot_distance
    eps = 0.001

    if 'hcp' in pattern_type.lower():
        dy = dx * np.sqrt(3) / 2.0
        is_layer_b = (layer_index % 2 == 1)
        layer_offset_x = 0.5 * dx if is_layer_b else 0.0
        layer_offset_y = (dx * np.sqrt(3) / 6.0) if is_layer_b else 0.0

        raw_y = np.arange(-half_size, half_size + dy + eps, dy)
        spots = []
        for row_idx, y_base in enumerate(raw_y):
            y = y_base + layer_offset_y
            if not (-half_size - eps <= y <= half_size + eps):
                continue
            row_stagger = 0.5 * dx if (row_idx % 2 == 1) else 0.0
            total_x_offset = row_stagger + layer_offset_x
            raw_x = np.arange(-half_size, half_size + dx + eps, dx)
            x = raw_x + total_x_offset
            # Numpy 向量化过滤当前行
            valid_mask = (x >= -half_size - eps) & (x <= half_size + eps)
            valid_x = x[valid_mask]
            spots.extend([(vx, y) for vx in valid_x])
        return spots
    else:
        raw_range = np.arange(0, square_size + eps, dx)
        shift = raw_range[-1] / 2.0 if len(raw_range) > 0 else 0
        coords1d = raw_range - shift

        X, Y = np.meshgrid(coords1d, coords1d)
        mask = (X >= -half_size - eps) & (X <= half_size + eps) & \
               (Y >= -half_size - eps) & (Y <= half_size + eps)

        valid_x = X[mask].flatten()
        valid_y = Y[mask].flatten()
        return list(zip(valid_x, valid_y))


# ==========================================
# 4. 分组逻辑
# ==========================================
def assign_groups(spots, n_groups, spot_distance, square_size, pattern_type='sc'):
    if len(spots) == 0:
        return {}, {}

    group_map = {}
    label_map = {}
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
        block_i = i_idx // block_size
        block_j = j_idx // block_size
        local_i = i_idx % block_size
        local_j = j_idx % block_size

        if block_size in MELT_ORDERS:
            row_index = block_size - 1 - local_j
            col_index = local_i
            if 0 <= row_index < block_size and 0 <= col_index < block_size:
                group_id = MELT_ORDERS[block_size][row_index][col_index]
            else:
                group_id = (local_j * block_size + local_i) % n_groups
        else:
            group_id = (local_j * block_size + local_i) % n_groups

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
        label_map[idx] = f"{block_letter}{group_id}"

    return group_map, label_map


# ==========================================
# 5. 筛选熔化区域内的点
# ==========================================
def filter_spots_in_geometry(spots, geometry):
    """筛选在几何轮廓内的点"""
    if geometry is None or geometry.is_empty:
        return [], set()

    print("  正在执行向量化点过滤...")
    spots_array = np.array(spots)

    # 包围盒预过滤向量化
    minx, miny, maxx, maxy = geometry.bounds
    bbox_mask = (spots_array[:, 0] >= minx) & (spots_array[:, 0] <= maxx) & \
                (spots_array[:, 1] >= miny) & (spots_array[:, 1] <= maxy)

    candidate_indices = np.where(bbox_mask)[0]
    candidates = spots_array[candidate_indices]

    if len(candidates) == 0:
        return [], set()

    try:
        points = shapely.points(candidates[:, 0], candidates[:, 1])
        contains_mask = shapely.contains(geometry, points)
        melting_indices = candidate_indices[contains_mask].tolist()
    except AttributeError:
        # 如果 Shapely 版本太老 (<2.0) 的备选方案
        prepared_geom = prep(geometry)
        melting_indices = [idx for idx, pt in zip(candidate_indices, candidates)
                           if prepared_geom.contains(Point(pt[0], pt[1]))]

    print(f"  包围盒预过滤: {len(spots)} → {len(candidates)} 个候选点")
    return melting_indices, set(melting_indices)


# ==========================================
# 6. 社区检测与连接图
# ==========================================
def build_networkx_graph(spots, spot_indices, j_max):
    """构建点的连接图（批量插入优化）"""
    G = nx.Graph()
    G.add_nodes_from(spot_indices)

    coords = np.array([spots[idx] for idx in spot_indices])
    if len(coords) > 1:
        tree = cKDTree(coords)
        # 直接输出 ndarray，避免对象解包
        pairs = tree.query_pairs(r=j_max, output_type='ndarray')

        if len(pairs) > 0:
            # 批量将局部索引映射回全局索引
            idx_array = np.array(spot_indices)
            global_u = idx_array[pairs[:, 0]]
            global_v = idx_array[pairs[:, 1]]

            # 向量化计算距离权重
            diff = coords[pairs[:, 0]] - coords[pairs[:, 1]]
            distances = np.linalg.norm(diff, axis=1)

            # 批量添加带权边，避免 NetworkX 极慢的循环
            edges = zip(global_u, global_v, distances)
            G.add_weighted_edges_from(edges)

    return G


def detect_communities_louvain(G):
    if G.number_of_nodes() == 0:
        return []
    if G.number_of_nodes() == 1:
        return [set(G.nodes())]
    if G.number_of_edges() == 0:
        return [{node} for node in G.nodes()]

    communities = nx.community.louvain_communities(G, weight='weight', seed=42)
    return list(communities)


# ==========================================
# 7. 路径规划
# ==========================================
def order_communities_tsp_cached(spots, communities):
    """
    使用社区质心 (Centroid) 加速距离矩阵计算，
    并应用 Numba JIT 编译的 2-opt 优化 TSP 排序。
    """
    n_comm = len(communities)
    if n_comm <= 1:
        return list(range(n_comm))
    if n_comm == 2:
        return [0, 1]

    # ★ 计算社区质心
    centroids = []
    for comm in communities:
        comm_coords = np.array([spots[idx] for idx in comm])
        centroids.append(comm_coords.mean(axis=0))
    centroids = np.array(centroids)

    # 向量化计算质心之间的距离矩阵
    diff = centroids[:, np.newaxis, :] - centroids[np.newaxis, :, :]
    dist_matrix = np.linalg.norm(diff, axis=2)

    # TSP 求解
    tour = _nearest_insertion(dist_matrix)
    tour_arr = np.array(tour, dtype=np.int32)
    # 调用 Numba 加速版的 2-opt
    tour_arr = _two_opt(tour_arr, dist_matrix)

    return tour_arr.tolist()


def _nearest_insertion(dist_matrix):
    n = len(dist_matrix)
    if n <= 2:
        return list(range(n))

    in_tour = [False] * n
    tour = [0]
    in_tour[0] = True

    nearest = np.argmin(dist_matrix[0][1:]) + 1
    tour.append(nearest)
    in_tour[nearest] = True

    while len(tour) < n:
        best_dist = float('inf')
        best_node = -1
        for node in range(n):
            if in_tour[node]:
                continue
            for t_node in tour:
                if dist_matrix[t_node][node] < best_dist:
                    best_dist = dist_matrix[t_node][node]
                    best_node = node

        best_pos = 0
        best_increase = float('inf')
        for i in range(len(tour)):
            j = (i + 1) % len(tour)
            increase = (dist_matrix[tour[i]][best_node] +
                        dist_matrix[best_node][tour[j]] -
                        dist_matrix[tour[i]][tour[j]])
            if increase < best_increase:
                best_increase = increase
                best_pos = i + 1

        tour.insert(best_pos, best_node)
        in_tour[best_node] = True

    return tour


# ★ 真正的机器码编译加速
@njit(fastmath=True)
def _two_opt(tour_arr, dist_matrix):
    """
    使用 Numba JIT 编译的 2-opt 优化，
    消除 Python 解释器开销，执行速度提升百倍。
    """
    n = len(tour_arr)
    if n <= 3:
        return tour_arr

    improved = True
    while improved:
        improved = False
        for i in range(n - 1):
            for j in range(i + 2, n):
                if j == n - 1:
                    d_before = dist_matrix[tour_arr[i], tour_arr[i + 1]]
                    d_after = dist_matrix[tour_arr[i], tour_arr[j]]
                else:
                    d_before = (dist_matrix[tour_arr[i], tour_arr[i + 1]] +
                                dist_matrix[tour_arr[j], tour_arr[(j + 1) % n]])
                    d_after = (dist_matrix[tour_arr[i], tour_arr[j]] +
                               dist_matrix[tour_arr[i + 1], tour_arr[(j + 1) % n]])

                if d_after < d_before - 1e-10:
                    left = i + 1
                    right = j
                    while left < right:
                        temp = tour_arr[left]
                        tour_arr[left] = tour_arr[right]
                        tour_arr[right] = temp
                        left += 1
                        right -= 1
                    improved = True
    return tour_arr


def get_community_endpoints(spots, communities, tour):
    n = len(tour)
    endpoints = {}

    for order_pos in range(n):
        comm_idx = tour[order_pos]
        comm_list = list(communities[comm_idx])

        if n == 1:
            start = min(comm_list, key=lambda i: spots[i][0] + spots[i][1])
            end = max(comm_list, key=lambda i: spots[i][0] + spots[i][1])
            endpoints[comm_idx] = (start, end)
            continue

        if order_pos == 0:
            start = min(comm_list, key=lambda i: spots[i][0] + spots[i][1])
        else:
            prev_comm_idx = tour[order_pos - 1]
            prev_end = endpoints[prev_comm_idx][1]
            prev_coord = np.array(spots[prev_end])
            start = min(comm_list,
                        key=lambda i: np.linalg.norm(np.array(spots[i]) - prev_coord))

        if order_pos == n - 1:
            start_coord = np.array(spots[start])
            end = max(comm_list,
                      key=lambda i: np.linalg.norm(np.array(spots[i]) - start_coord))
        else:
            next_comm_idx = tour[order_pos + 1]
            next_comm_list = list(communities[next_comm_idx])
            next_coords = np.array([spots[idx] for idx in next_comm_list])
            next_tree = cKDTree(next_coords)
            curr_coords = np.array([spots[idx] for idx in comm_list])
            dists, _ = next_tree.query(curr_coords, k=1)
            best_local = np.argmin(dists)
            end = comm_list[best_local]

        if start == end and len(comm_list) > 1:
            other = [idx for idx in comm_list if idx != start]
            end = other[0]

        endpoints[comm_idx] = (start, end)

    return endpoints


def build_adj_matrix_sparse(spots, node_list, G_sub):
    n = len(node_list)
    node_to_local = {node: i for i, node in enumerate(node_list)}

    if n > 200:
        adj_sparse = lil_matrix((n, n), dtype=np.int8)
        for u, v in G_sub.edges():
            if u in node_to_local and v in node_to_local:
                i, j = node_to_local[u], node_to_local[v]
                adj_sparse[i, j] = 1
                adj_sparse[j, i] = 1
        adj = adj_sparse.toarray()
    else:
        adj = np.zeros((n, n), dtype=np.int8)
        for u, v in G_sub.edges():
            if u in node_to_local and v in node_to_local:
                i, j = node_to_local[u], node_to_local[v]
                adj[i][j] = adj[j][i] = 1

    return adj


def heuristic_dfs_simple(adj_matrix, coords_array, start_idx):
    n = adj_matrix.shape[0]

    if n == 0:
        return []
    if n == 1:
        return [0]

    visited = np.zeros(n, dtype=bool)
    path = [start_idx]
    visited[start_idx] = True

    degrees = np.sum(adj_matrix, axis=1)
    neighbors_of_start = np.where(adj_matrix[start_idx] == 1)[0]
    degrees[neighbors_of_start] -= 1

    while len(path) < n:
        current = path[-1]
        neighbors = np.where((adj_matrix[current] == 1) & ~visited)[0]

        if len(neighbors) > 0:
            best_neighbor_idx = np.argmin(degrees[neighbors])
            next_v = neighbors[best_neighbor_idx]
        else:
            unvisited_idx = np.where(~visited)[0]
            if len(unvisited_idx) == 0:
                break

            unvisited_coords = coords_array[unvisited_idx]
            current_coord = coords_array[current]
            distances = np.linalg.norm(unvisited_coords - current_coord, axis=1)
            nearest_local_idx = np.argmin(distances)
            next_v = unvisited_idx[nearest_local_idx]

        path.append(next_v)
        visited[next_v] = True

        unvisited_neighbors = np.where((adj_matrix[next_v] == 1) & ~visited)[0]
        if len(unvisited_neighbors) > 0:
            degrees[unvisited_neighbors] -= 1

    return path


def generate_melting_path(spots, group_map, melting_spots_set, n_groups, j_max):
    sequence = []
    total_communities = 0

    print("\n  开始路径规划 (Numba 加速版)...")
    print(f"  参数: {n_groups} 组, j_max={j_max:.2f} mm")

    for group_id in tqdm(range(n_groups), desc="  分组处理"):
        group_spots = [idx for idx, gid in group_map.items()
                       if gid == group_id and idx in melting_spots_set]

        if not group_spots:
            continue

        G = build_networkx_graph(spots, group_spots, j_max)
        communities = detect_communities_louvain(G)

        if not communities:
            continue

        total_communities += len(communities)

        if len(communities) > 1:
            tour = order_communities_tsp_cached(spots, communities)
        else:
            tour = [0]

        endpoints = get_community_endpoints(spots, communities, tour)

        if sequence:
            last_coord = np.array(spots[sequence[-1]])
            first_comm_idx = tour[0]
            first_comm_list = list(communities[first_comm_idx])
            closest_start = min(first_comm_list,
                                key=lambda i: np.linalg.norm(np.array(spots[i]) - last_coord))
            old_start, old_end = endpoints[first_comm_idx]
            endpoints[first_comm_idx] = (closest_start, old_end)

        for comm_idx in tour:
            comm_nodes = list(communities[comm_idx])
            if not comm_nodes:
                continue

            start_node, _ = endpoints[comm_idx]
            if start_node not in comm_nodes:
                start_node = comm_nodes[0]

            G_sub = G.subgraph(comm_nodes)
            adj = build_adj_matrix_sparse(spots, comm_nodes, G_sub)

            node_to_local = {node: i for i, node in enumerate(comm_nodes)}
            coords = np.array([spots[idx] for idx in comm_nodes])
            start_local = node_to_local[start_node]

            local_path = heuristic_dfs_simple(adj, coords, start_local)

            for local_idx in local_path:
                global_idx = comm_nodes[local_idx]
                if global_idx not in set(sequence):
                    sequence.append(global_idx)

    print(f"  ✓ 社区总数: {total_communities}, 路径点数: {len(sequence)}")
    return sequence


# ==========================================
# 8. 虚拟点与最终路径
# ==========================================
def generate_virtual_jump_points(start, end, max_dist):
    sx, sy = start
    ex, ey = end
    dist = np.sqrt((ex - sx) ** 2 + (ey - sy) ** 2)

    if dist <= max_dist + 0.001:
        return []

    virtuals = []
    dx, dy = (ex - sx) / dist, (ey - sy) / dist
    curr = max_dist

    while curr < dist - 0.001:
        virtuals.append((sx + dx * curr, sy + dy * curr))
        curr += max_dist

    return virtuals


def build_final_path_with_virtuals(spots, sequence, max_jump_distance, dwell_time, label_map):
    final_path = []
    jump_count = 0

    for spot_idx in sequence:
        curr_x, curr_y = spots[spot_idx]

        if final_path:
            last_pt = final_path[-1]
            last_x, last_y = last_pt['x'], last_pt['y']
            dist = np.sqrt((curr_x - last_x) ** 2 + (curr_y - last_y) ** 2)

            if dist > max_jump_distance:
                jump_count += 1
                v_points = generate_virtual_jump_points(
                    (last_x, last_y), (curr_x, curr_y), max_jump_distance)

                for vp in v_points:
                    final_path.append({
                        'x': vp[0], 'y': vp[1],
                        'dwell_time': 0,
                        'is_virtual': True,
                        'index': -1,
                        'label': ''
                    })

        final_path.append({
            'x': curr_x, 'y': curr_y,
            'dwell_time': dwell_time,
            'is_virtual': False,
            'index': spot_idx,
            'label': label_map.get(spot_idx, '')
        })

    if jump_count > 0:
        virt_count = sum(1 for p in final_path if p['is_virtual'])
        print(f"  ✓ 跳转 {jump_count} 次，插入 {virt_count} 个虚拟点")

    return final_path


# ==========================================
# 9. 导出功能
# ==========================================
def export_to_binary(final_path, output_filename, beam_current=0, focus_val=0):
    print(f"\n开始生成二进制文件: {output_filename}")

    if not final_path:
        print("❌ 路径为空")
        return False

    segments = []
    prev_x, prev_y = final_path[0]['x'], final_path[0]['y']

    if not final_path[0]['is_virtual']:
        num_lines = int(round(final_path[0]['dwell_time'] / 25.0))
        for _ in range(num_lines):
            segments.append((prev_x, prev_y, prev_x + 0.001, prev_y))

    for i in range(1, len(final_path)):
        pt = final_path[i]
        curr_x, curr_y = pt['x'], pt['y']

        if abs(curr_x - prev_x) > 0.0001 or abs(curr_y - prev_y) > 0.0001:
            segments.append((prev_x, prev_y, curr_x, curr_y))

        if not pt['is_virtual']:
            num_lines = int(round(pt['dwell_time'] / 25.0))
            for _ in range(num_lines):
                segments.append((curr_x, curr_y, curr_x + 0.001, curr_y))

        prev_x, prev_y = curr_x, curr_y

    total_vertices = len(segments) * 2

    try:
        if not output_filename.endswith('.bin'):
            output_filename += '.bin'

        with open(output_filename, 'wb') as f:
            f.write(struct.pack('dd', 6.0, float(total_vertices)))
            for seg in segments:
                f.write(struct.pack('6d', seg[0], seg[1], focus_val, beam_current, 0.0, 0.0))
                f.write(struct.pack('6d', seg[2], seg[3], focus_val, beam_current, 0.0, 0.0))

        print(f"✓ 文件生成成功: {output_filename}")
        print(f"  - 线段数: {len(segments)}")
        print(f"  - 顶点数: {total_vertices}")
        return True

    except Exception as e:
        print(f"❌ 导出失败: {e}")
        return False


# ==========================================
# 10. 可视化
# ==========================================
def visualize_result(square_size, geometry, spots, melting_spots_set, final_path,
                     label_map, z_val, filename="result"):
    print("\n生成可视化预览...")

    fig, ax = plt.subplots(figsize=(14, 14))
    ax.set_aspect('equal')

    half = square_size / 2.0
    padding = 5
    ax.set_xlim(-half - padding, half + padding)
    ax.set_ylim(-half - padding, half + padding)
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('X (mm)', fontsize=12)
    ax.set_ylabel('Y (mm)', fontsize=12)
    ax.set_title(f'CLI 切片点扫路径 (Z={z_val:.3f}mm)', fontsize=14, fontweight='bold')

    ax.add_patch(Rectangle((-half, -half), square_size, square_size,
                           fill=False, edgecolor='gray', linewidth=2))

    if geometry is not None:
        if isinstance(geometry, Polygon):
            geoms = [geometry]
        elif isinstance(geometry, MultiPolygon):
            geoms = list(geometry.geoms)
        else:
            geoms = [geometry]

        for poly in geoms:
            if hasattr(poly, 'exterior'):
                x, y = poly.exterior.xy
                ax.plot(x, y, color='red', linewidth=2.5, linestyle='--', alpha=0.8)
                for interior in poly.interiors:
                    ix, iy = interior.xy
                    ax.plot(ix, iy, color='orange', linewidth=2, linestyle=':', alpha=0.8)

    non_melting_x = [spots[i][0] for i in range(len(spots)) if i not in melting_spots_set]
    non_melting_y = [spots[i][1] for i in range(len(spots)) if i not in melting_spots_set]
    if non_melting_x:
        ax.scatter(non_melting_x, non_melting_y, c='lightgray', s=15,
                   alpha=0.3, edgecolors='none')

    real_pts = [p for p in final_path if not p['is_virtual']]
    virt_pts = [p for p in final_path if p['is_virtual']]
    real_x = [p['x'] for p in real_pts]
    real_y = [p['y'] for p in real_pts]
    virt_x = [p['x'] for p in virt_pts]
    virt_y = [p['y'] for p in virt_pts]

    spot_radius = 0.25

    if real_x:
        circles = [Circle((rx, ry), spot_radius) for rx, ry in zip(real_x, real_y)]
        pc = PatchCollection(circles,
                             facecolor='steelblue', edgecolor='navy',
                             linewidth=0.5, alpha=0.7, zorder=100)
        ax.add_collection(pc)

        all_x = [p['x'] for p in final_path]
        all_y = [p['y'] for p in final_path]
        ax.plot(all_x, all_y, 'g-', alpha=0.3, linewidth=0.8, zorder=50)

        ax.plot(real_x[0], real_y[0], 'o', color='lime', markersize=15,
                markeredgecolor='darkgreen', markeredgewidth=2, label='起点', zorder=200)
        ax.plot(real_x[-1], real_y[-1], 's', color='red', markersize=12,
                markeredgecolor='darkred', markeredgewidth=2, label='终点', zorder=200)

    if virt_x:
        ax.scatter(virt_x, virt_y, c='orange', marker='x', s=30,
                   alpha=0.6, label=f'虚拟跳转点 ({len(virt_x)})', zorder=80)

    jump_x, jump_y = [], []
    for i in range(len(final_path) - 1):
        curr = final_path[i]
        next_p = final_path[i + 1]
        if curr['is_virtual'] or next_p['is_virtual']:
            jump_x.extend([curr['x'], next_p['x'], None])
            jump_y.extend([curr['y'], next_p['y'], None])

    if jump_x:
        ax.plot(jump_x, jump_y, 'r--', linewidth=0.8, alpha=0.4, zorder=60)

    ax.plot([], [], 'o', color='steelblue', markersize=8,
            markeredgecolor='navy', label=f'熔化点 ({len(real_x)}, ⌀500μm)')
    ax.legend(loc='upper right', fontsize=9)

    info_text = (f"基板: {square_size}×{square_size} mm\n"
                 f"总点数: {len(final_path)}\n"
                 f"熔化点: {len(real_pts)}\n"
                 f"虚拟点: {len(virt_pts)}")
    ax.text(0.02, 0.98, info_text, transform=ax.transAxes,
            fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.savefig(f"{filename}.png", dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ 图片已保存: {filename}.png")
    plt.show()


# ==========================================
# 11. 统计分析
# ==========================================
def analyze_path(final_path):
    if len(final_path) < 2:
        return

    real_pts = [p for p in final_path if not p['is_virtual']]
    virt_pts = [p for p in final_path if p['is_virtual']]

    print("\n" + "=" * 50)
    print("路径统计分析")
    print("=" * 50)
    print(f"总点数: {len(final_path)}")
    print(f"  - 真实熔化点: {len(real_pts)}")
    print(f"  - 虚拟跳转点: {len(virt_pts)}")

    steps = []
    for i in range(len(final_path) - 1):
        p1, p2 = final_path[i], final_path[i + 1]
        dist = np.sqrt((p1['x'] - p2['x']) ** 2 + (p1['y'] - p2['y']) ** 2)
        steps.append(dist)

    if steps:
        print(f"\n步长统计:")
        print(f"  最小: {min(steps):.4f} mm")
        print(f"  最大: {max(steps):.4f} mm")
        print(f"  平均: {np.mean(steps):.4f} mm")
        print(f"  总路径长度: {sum(steps):.2f} mm")


# ==========================================
# 12. 主程序
# ==========================================
def main():
    print("=" * 60)
    print("  CLI 切片点扫路径生成器 (Numba/Shapely 深度优化版)")
    print("  - Numba JIT 机器码编译加速 TSP 路径规划")
    print("  - Shapely C级 接口彻底向量化过滤点")
    print("  - NumPy Meshgrid 替代纯 Python 双重循环")
    print("  - O(N) 质心降维加速社区计算")
    print("=" * 60)

    try:
        print("\n【步骤1】设置基板参数")
        square_size = float(input("基板边长 (mm, 默认200): ").strip() or 200)
        spot_distance = float(input("网格间距 (mm, 默认1.0): ").strip() or 1.0)
        pattern_type = input("点阵类型 (sc/hcp, 默认sc): ").strip().lower() or 'sc'

        layer_offset = 0
        if 'hcp' in pattern_type:
            layer_offset = int(input("HCP层偏移 (0=A层, 1=B层, 默认0): ").strip() or 0)

        print("\n[-] 正在生成基板点阵...")
        spots = generate_substrate_spots(square_size, spot_distance, pattern_type, layer_offset)
        print(f"✓ 基板点数: {len(spots)}")

        print("\n【步骤2】分组设置")
        n_groups = int(input("分组数 (16/25/36/49/64, 默认36): ").strip() or 36)
        group_map, label_map = assign_groups(spots, n_groups, spot_distance,
                                             square_size, pattern_type)
        block_size = int(np.sqrt(n_groups))
        print(f"✓ 分组完成 (block_size={block_size})")

        print("\n【步骤3】读取 CLI 文件")
        cli_path = input("CLI文件路径: ").strip().replace('"', '').replace("'", "")
        reader = CLIReader(cli_path)
        if not reader.read():
            return

        z_levels = reader.get_all_z_levels()
        print(f"可用层: {len(z_levels)} 层, Z范围: {min(z_levels):.4f} ~ {max(z_levels):.4f} mm")

        print("\n【步骤4】选择层")
        layer_idx = int(input(f"层索引 (0~{len(z_levels) - 1}): ").strip())
        z_val, layer_data = reader.get_layer_by_index(layer_idx)

        if not layer_data:
            print("该层无数据")
            return

        print(f"[-] 构建几何体...")
        geometry = build_geometry_from_layer(layer_data)

        if geometry is None or geometry.is_empty:
            print("无有效几何体")
            return

        bounds = get_geometry_bounds(geometry)
        print(f"  模型尺寸: {bounds['width']:.2f} x {bounds['height']:.2f} mm")

        print("\n【步骤5】筛选熔化点")
        melting_indices, melting_spots_set = filter_spots_in_geometry(spots, geometry)
        print(f"✓ 熔化点: {len(melting_indices)} ({len(melting_indices) / len(spots) * 100:.1f}%)")

        if not melting_indices:
            print("错误：无熔化点")
            return

        print("\n【步骤6】路径参数")
        dwell_time = float(input("停留时间 (μs, 默认600): ").strip() or 600)
        default_j_max = block_size * spot_distance * 2
        j_max = float(input(f"最大跳跃距离 (mm, 默认{default_j_max:.1f}): ").strip() or default_j_max)

        print("\n【步骤7】生成路径")
        sequence = generate_melting_path(spots, group_map, melting_spots_set, n_groups, j_max)

        if not sequence:
            print("❌ 路径生成失败")
            return

        final_path = build_final_path_with_virtuals(spots, sequence, j_max, dwell_time, label_map)

        real_count = sum(1 for p in final_path if not p['is_virtual'])
        virt_count = sum(1 for p in final_path if p['is_virtual'])
        print(f"✓ 最终路径: {real_count} 熔化点 + {virt_count} 虚拟点")

        analyze_path(final_path)

        print("\n【步骤8】导出")
        if input("导出二进制? (y/n, 默认y): ").strip().lower() != 'n':
            bin_name = input("文件名 (默认output): ").strip() or "output"
            export_to_binary(final_path, bin_name)

        print("\n【步骤9】可视化")
        if input("生成图片? (y/n, 默认y): ").strip().lower() != 'n':
            viz_name = input("图片名 (默认result): ").strip() or "result"
            visualize_result(square_size, geometry, spots, melting_spots_set,
                             final_path, label_map, z_val, viz_name)

        print("\n" + "=" * 60)
        print("✓ 完成!")
        print("=" * 60)

    except KeyboardInterrupt:
        print("\n取消操作")
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    plt.rcParams['figure.max_open_warning'] = 0
    main()