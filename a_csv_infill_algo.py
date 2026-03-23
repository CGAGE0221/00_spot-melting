import struct
import warnings
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
from matplotlib.collections import PatchCollection
from scipy.spatial import cKDTree
from scipy.sparse import lil_matrix
import shapely
from shapely.geometry import Point, Polygon, MultiPolygon
from shapely.prepared import prep
from tqdm import tqdm
import networkx as nx
from numba import njit

warnings.filterwarnings('ignore', message='.*Glyph.*missing from font.*')
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

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
            if not (-half_size - eps <= y <= half_size + eps): continue
            row_stagger = 0.5 * dx if (row_idx % 2 == 1) else 0.0
            total_x_offset = row_stagger + layer_offset_x
            raw_x = np.arange(-half_size, half_size + dx + eps, dx)
            x = raw_x + total_x_offset
            valid_mask = (x >= -half_size - eps) & (x <= half_size + eps)
            spots.extend([(vx, y) for vx in x[valid_mask]])
        return spots
    else:
        raw_range = np.arange(0, square_size + eps, dx)
        shift = raw_range[-1] / 2.0 if len(raw_range) > 0 else 0
        coords1d = raw_range - shift
        X, Y = np.meshgrid(coords1d, coords1d)
        mask = (X >= -half_size - eps) & (X <= half_size + eps) & \
               (Y >= -half_size - eps) & (Y <= half_size + eps)
        return list(zip(X[mask].flatten(), Y[mask].flatten()))


def assign_groups(spots, n_groups, spot_distance, square_size, pattern_type='sc'):
    if len(spots) == 0: return {}, {}
    group_map, label_map = {}, {}
    letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    block_size = max(1, int(np.sqrt(n_groups)))

    dx = spot_distance
    dy = spot_distance * np.sqrt(3) / 2.0 if 'hcp' in pattern_type.lower() else spot_distance
    min_x, min_y = min(p[0] for p in spots), min(p[1] for p in spots)
    max_x = max(p[0] for p in spots)
    points_per_line = int(round((max_x - min_x) / dx)) + 1

    for idx, (x, y) in enumerate(spots):
        i_idx, j_idx = int(round((x - min_x) / dx)), int(round((y - min_y) / dy))
        block_i, block_j = i_idx // block_size, j_idx // block_size
        local_i, local_j = i_idx % block_size, j_idx % block_size

        if block_size in MELT_ORDERS:
            row_index, col_index = block_size - 1 - local_j, local_i
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
            if temp_num < 0: break

        group_map[idx] = group_id
        label_map[idx] = f"{block_letter}{group_id}"
    return group_map, label_map


def filter_spots_in_geometry(spots, geometry):
    if geometry is None or geometry.is_empty: return [], set()
    spots_array = np.array(spots)
    minx, miny, maxx, maxy = geometry.bounds
    bbox_mask = (spots_array[:, 0] >= minx) & (spots_array[:, 0] <= maxx) & \
                (spots_array[:, 1] >= miny) & (spots_array[:, 1] <= maxy)

    candidate_indices = np.where(bbox_mask)[0]
    candidates = spots_array[candidate_indices]
    if len(candidates) == 0: return [], set()

    prepared_geom = prep(geometry)
    melting_indices = [idx for idx, pt in zip(candidate_indices, candidates)
                       if prepared_geom.contains(Point(pt[0], pt[1]))]
    return melting_indices, set(melting_indices)


def build_networkx_graph(spots, spot_indices, j_max):
    G = nx.Graph()
    G.add_nodes_from(spot_indices)
    coords = np.array([spots[idx] for idx in spot_indices])
    if len(coords) > 1:
        tree = cKDTree(coords)
        pairs = tree.query_pairs(r=j_max, output_type='ndarray')
        if len(pairs) > 0:
            idx_array = np.array(spot_indices)
            global_u = idx_array[pairs[:, 0]]
            global_v = idx_array[pairs[:, 1]]
            diff = coords[pairs[:, 0]] - coords[pairs[:, 1]]
            distances = np.linalg.norm(diff, axis=1)
            edges = zip(global_u, global_v, distances)
            G.add_weighted_edges_from(edges)
    return G


def detect_communities_louvain(G):
    if G.number_of_nodes() == 0: return []
    if G.number_of_nodes() == 1: return [set(G.nodes())]
    if G.number_of_edges() == 0: return [{node} for node in G.nodes()]
    return list(nx.community.louvain_communities(G, weight='weight', seed=42))


def order_communities_tsp_cached(spots, communities):
    n_comm = len(communities)
    if n_comm <= 1: return list(range(n_comm))
    if n_comm == 2: return [0, 1]

    centroids = []
    for comm in communities:
        comm_coords = np.array([spots[idx] for idx in comm])
        centroids.append(comm_coords.mean(axis=0))
    centroids = np.array(centroids)

    diff = centroids[:, np.newaxis, :] - centroids[np.newaxis, :, :]
    dist_matrix = np.linalg.norm(diff, axis=2)

    tour = _nearest_insertion(dist_matrix)
    tour_arr = np.array(tour, dtype=np.int32)
    tour_arr = _two_opt(tour_arr, dist_matrix)
    return tour_arr.tolist()


def _nearest_insertion(dist_matrix):
    n = len(dist_matrix)
    if n <= 2: return list(range(n))
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
            if in_tour[node]: continue
            for t_node in tour:
                if dist_matrix[t_node][node] < best_dist:
                    best_dist = dist_matrix[t_node][node]
                    best_node = node
        best_pos = 0
        best_increase = float('inf')
        for i in range(len(tour)):
            j = (i + 1) % len(tour)
            increase = (dist_matrix[tour[i]][best_node] + dist_matrix[best_node][tour[j]] - dist_matrix[tour[i]][
                tour[j]])
            if increase < best_increase:
                best_increase = increase
                best_pos = i + 1
        tour.insert(best_pos, best_node)
        in_tour[best_node] = True
    return tour


@njit(fastmath=True)
def _two_opt(tour_arr, dist_matrix):
    n = len(tour_arr)
    if n <= 3: return tour_arr
    improved = True
    while improved:
        improved = False
        for i in range(n - 1):
            for j in range(i + 2, n):
                if j == n - 1:
                    d_before = dist_matrix[tour_arr[i], tour_arr[i + 1]]
                    d_after = dist_matrix[tour_arr[i], tour_arr[j]]
                else:
                    d_before = (dist_matrix[tour_arr[i], tour_arr[i + 1]] + dist_matrix[
                        tour_arr[j], tour_arr[(j + 1) % n]])
                    d_after = (dist_matrix[tour_arr[i], tour_arr[j]] + dist_matrix[
                        tour_arr[i + 1], tour_arr[(j + 1) % n]])
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
            start = min(comm_list, key=lambda i: np.linalg.norm(np.array(spots[i]) - prev_coord))
        if order_pos == n - 1:
            start_coord = np.array(spots[start])
            end = max(comm_list, key=lambda i: np.linalg.norm(np.array(spots[i]) - start_coord))
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
    if n == 0: return []
    if n == 1: return [0]
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
            if len(unvisited_idx) == 0: break
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
    visited_set = set()
    print("  [填充路径] 执行 Louvain社区 + Heuristic DFS...")
    for group_id in tqdm(range(n_groups), desc="  分组处理", leave=False):
        group_spots = [idx for idx, gid in group_map.items() if gid == group_id and idx in melting_spots_set]
        if not group_spots: continue
        G = build_networkx_graph(spots, group_spots, j_max)
        communities = detect_communities_louvain(G)
        if not communities: continue
        if len(communities) > 1:
            tour = order_communities_tsp_cached(spots, communities)
        else:
            tour = [0]
        endpoints = get_community_endpoints(spots, communities, tour)

        if sequence:
            last_coord = np.array(spots[sequence[-1]])
            first_comm_idx = tour[0]
            first_comm_list = list(communities[first_comm_idx])
            closest_start = min(first_comm_list, key=lambda i: np.linalg.norm(np.array(spots[i]) - last_coord))
            old_start, old_end = endpoints[first_comm_idx]
            endpoints[first_comm_idx] = (closest_start, old_end)

        for comm_idx in tour:
            comm_nodes = list(communities[comm_idx])
            if not comm_nodes: continue
            start_node, _ = endpoints[comm_idx]
            if start_node not in comm_nodes: start_node = comm_nodes[0]
            G_sub = G.subgraph(comm_nodes)
            adj = build_adj_matrix_sparse(spots, comm_nodes, G_sub)
            node_to_local = {node: i for i, node in enumerate(comm_nodes)}
            coords = np.array([spots[idx] for idx in comm_nodes])
            start_local = node_to_local[start_node]
            local_path = heuristic_dfs_simple(adj, coords, start_local)
            for local_idx in local_path:
                global_idx = comm_nodes[local_idx]
                if global_idx not in visited_set:
                    sequence.append(global_idx)
                    visited_set.add(global_idx)
    return sequence


def generate_virtual_jump_points(start, end, max_dist):
    sx, sy = start
    ex, ey = end
    dist = np.sqrt((ex - sx) ** 2 + (ey - sy) ** 2)
    if dist <= max_dist + 0.001: return []
    virtuals = []
    dx, dy = (ex - sx) / dist, (ey - sy) / dist
    curr = max_dist
    while curr < dist - 0.001:
        virtuals.append((sx + dx * curr, sy + dy * curr))
        curr += max_dist
    return virtuals


def build_final_path_with_virtuals(spots, sequence, dwell_time, beam_current, label_map):
    final_path = []
    for spot_idx in sequence:
        curr_x, curr_y = spots[spot_idx]
        final_path.append({
            'x': curr_x, 'y': curr_y, 'dwell_time': dwell_time, 'beam_current': beam_current,
            'is_virtual': False, 'index': spot_idx, 'label': label_map.get(spot_idx, ''), 'type': 'infill'
        })
    return final_path


def insert_virtuals_for_path(path_sequence, j_max):
    """区域内自动插入有束流断点，以25μs显性标出跳跃轨迹"""
    result = []
    for curr in path_sequence:
        if result:
            last = result[-1]
            dist = np.sqrt((curr['x'] - last['x']) ** 2 + (curr['y'] - last['y']) ** 2)
            if dist > j_max:
                v_points = generate_virtual_jump_points((last['x'], last['y']), (curr['x'], curr['y']), j_max)
                for vp in v_points:
                    result.append({
                        'x': vp[0], 'y': vp[1],
                        'dwell_time': 25,
                        'beam_current': last['beam_current'],
                        'is_virtual': True,
                        'type': 'virtual'
                    })
        result.append(curr)
    return result


def export_to_binary(final_path, output_filename, focus_val=0):
    print(f"\n[-] 开始生成二进制文件: {output_filename}.bin")
    if not final_path: return False
    segments = []
    prev_x, prev_y = final_path[0]['x'], final_path[0]['y']
    prev_bc = final_path[0].get('beam_current', 0.0)

    for pt in final_path:
        curr_x, curr_y, curr_bc = pt['x'], pt['y'], pt.get('beam_current', 0.0)
        if abs(curr_x - prev_x) > 0.0001 or abs(curr_y - prev_y) > 0.0001:
            # 保持移动段束流连续，避免移动时被强制关束。
            segments.append((prev_x, prev_y, curr_x, curr_y, prev_bc))

        num_lines = max(1, int(round(pt['dwell_time'] / 25.0)))
        for _ in range(num_lines):
            segments.append((curr_x, curr_y, curr_x + 0.001, curr_y, curr_bc))

        prev_x, prev_y = curr_x, curr_y
        prev_bc = curr_bc

    try:
        if not output_filename.endswith('.bin'): output_filename += '.bin'
        with open(output_filename, 'wb') as f:
            f.write(struct.pack('dd', 6.0, float(len(segments) * 2)))
            for seg in segments:
                f.write(struct.pack('6d', seg[0], seg[1], focus_val, seg[4], 0.0, 0.0))
                f.write(struct.pack('6d', seg[2], seg[3], focus_val, seg[4], 0.0, 0.0))
        print(f"✓ 文件生成成功: {output_filename}")
        return True
    except Exception as e:
        print(f"❌ 导出失败: {e}")
        return False


def visualize_result(square_size, geometry, spots, melting_spots_set, final_path, z_val=0, filename="result"):
    print(f"\n[-] 正在生成高分辨率可视化图片: {filename}.png")
    fig, ax = plt.subplots(figsize=(14, 14))
    ax.set_aspect('equal')

    if final_path:
        all_x = [p['x'] for p in final_path]
        all_y = [p['y'] for p in final_path]
        ax.set_xlim(min(all_x) - 10, max(all_x) + 10)
        ax.set_ylim(min(all_y) - 10, max(all_y) + 10)
    else:
        ax.set_xlim(-square_size / 2, square_size / 2)
        ax.set_ylim(-square_size / 2, square_size / 2)

    ax.grid(True, alpha=0.3)
    ax.set_xlabel('X (mm)', fontsize=12)
    ax.set_ylabel('Y (mm)', fontsize=12)
    ax.set_title('Spot Melting Experiment - Annulus Array', fontsize=14, fontweight='bold')

    infill_pts = [p for p in final_path if not p['is_virtual'] and p.get('type') == 'infill']
    contour_pts = [p for p in final_path if not p['is_virtual'] and p.get('type') == 'contour']
    current_change_pts = [p for p in final_path if p.get('type') == 'currentChanging']
    virt_pts = [p for p in final_path if p['is_virtual'] and p.get('type') != 'currentChanging']

    if infill_pts:
        circles = [Circle((p['x'], p['y']), 0.25) for p in infill_pts]
        ax.add_collection(
            PatchCollection(circles, facecolor='steelblue', edgecolor='navy', linewidth=0.5, alpha=0.7, zorder=100))

    if contour_pts:
        ax.scatter([p['x'] for p in contour_pts], [p['y'] for p in contour_pts], c='orange', s=10, label='Contour',
                   zorder=120)

    if current_change_pts:
        ax.scatter([p['x'] for p in current_change_pts], [p['y'] for p in current_change_pts],
                   c='black', marker='^', s=20, alpha=0.8, label='Current Changing', zorder=90)

    if virt_pts:
        ax.scatter([p['x'] for p in virt_pts], [p['y'] for p in virt_pts], c='red', marker='x', s=15, alpha=0.6,
                   label='Virtual Jumps', zorder=80)

    if final_path:
        ax.plot(all_x, all_y, 'g-', alpha=0.2, linewidth=0.5, zorder=50)
        ax.plot(all_x[0], all_y[0], 'o', color='lime', markersize=12, label='Start', zorder=200)
        ax.plot(all_x[-1], all_y[-1], 's', color='red', markersize=10, label='End', zorder=200)

    ax.legend(loc='upper right', fontsize=9)
    plt.savefig(f"{filename}.png", dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ 图片已保存: {filename}.png")
