import csv
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PatchCollection
from matplotlib.patches import Circle, Rectangle
from shapely.geometry import Point

from a_csv_contour_algo import ContourGenerator
from a_csv_infill_algo import (
    assign_groups,
    build_final_path_with_virtuals,
    export_to_binary,
    filter_spots_in_geometry,
    generate_melting_path,
    generate_substrate_spots,
    generate_virtual_jump_points,
    insert_virtuals_for_path,
)

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
        with open(filename, "r", encoding="utf-8-sig") as f:
            reader = csv.reader(f)
            rows = list(reader)
            start_row = 1 if rows and not rows[0][1].replace(".", "").replace("-", "").isdigit() else 0

            for row in rows[start_row:]:
                if len(row) < 12:
                    continue
                configs.append({
                    "type": str(row[0]).strip(),
                    "cx": float(row[1]),
                    "cy": float(row[2]),
                    "in_curr": float(row[3]),
                    "in_dwell": float(row[4]),
                    "r_in": float(row[5]),
                    "r_out": float(row[6]),
                    "spot_dist": float(row[7]),
                    "n_groups": int(row[8]),
                    "ct_curr": float(row[9]),
                    "ct_dwell": float(row[10]),
                    "inner_proj": float(row[11]),
                })
        print(f"[OK] 成功加载 {len(configs)} 组环形实验配置")
        return configs
    except Exception as e:
        print(f"[ERR] 读取 CSV 失败: {e}")
        return []


def create_template_csv(filename="annulus_experiment.csv"):
    headers = [
        "点阵类型",
        "中心X",
        "中心Y",
        "束流(填充)",
        "停留时间(填充)",
        "内径",
        "半径(外径)",
        "点间距(mm)",
        "分组数",
        "束流(轮廓)",
        "停留时间(轮廓)",
        "内部投影偏移量",
    ]
    data = [
        ["sc", -30, 30, 10.0, 400, 10.0, 20.0, 1.0, 25, 8.0, 30, 0.380],
        ["sc", 30, 30, 10.0, 300, 10.0, 20.0, 1.0, 25, 8.0, 30, 0.350],
        ["sc", -30, -30, 10.0, 200, 10.0, 20.0, 1.0, 25, 8.0, 30, 0.300],
        ["sc", 30, -30, 10.0, 100, 10.0, 20.0, 1.0, 25, 8.0, 30, 0.250],
    ]
    with open(filename, "w", newline="", encoding="utf-8-sig") as f:
        csv.writer(f).writerows([headers] + data)
    print(f"[OK] 已生成模板文件: {filename}")


def process_single_region(cfg, region_idx, j_max):
    print("\n" + "-" * 40)
    print(
        f"[区域 {region_idx + 1}] 中心=({cfg['cx']}, {cfg['cy']}), "
        f"内部投影偏移量={cfg['inner_proj']} mm"
    )

    geom = create_annulus_geometry(cfg["cx"], cfg["cy"], cfg["r_in"], cfg["r_out"])
    square_size = cfg["r_out"] * 2
    base_spots = generate_substrate_spots(square_size, cfg["spot_dist"], cfg["type"], layer_index=0)
    spots = [(x + cfg["cx"], y + cfg["cy"]) for x, y in base_spots]

    group_map, label_map = assign_groups(
        spots, cfg["n_groups"], cfg["spot_dist"], square_size, cfg["type"]
    )

    raw_indices, _ = filter_spots_in_geometry(spots, geom)
    if not raw_indices:
        print("  [WARN] 该区域内没有有效网格点")
        return [], []

    contour_gen = ContourGenerator(
        geometry=geom,
        contour_dwell_time=cfg["ct_dwell"],
        contour_beam_current=cfg["ct_curr"],
        inner_projection_offset=cfg["inner_proj"],
    )
    melting_indices = contour_gen.project_and_filter_infill(spots, raw_indices)
    melting_spots_set = set(melting_indices)

    infill_seq = generate_melting_path(spots, group_map, melting_spots_set, cfg["n_groups"], j_max)
    infill_path = build_final_path_with_virtuals(
        spots, infill_seq, cfg["in_dwell"], cfg["in_curr"], label_map
    )
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
        dist = np.sqrt((first_pt["x"] - last_pt["x"]) ** 2 + (first_pt["y"] - last_pt["y"]) ** 2)
        if dist > j_max:
            v_pts = generate_virtual_jump_points(
                (last_pt["x"], last_pt["y"]),
                (first_pt["x"], first_pt["y"]),
                j_max,
            )
            for vp in v_pts:
                main_path.append({
                    "x": vp[0],
                    "y": vp[1],
                    "dwell_time": 25,
                    "beam_current": last_pt["beam_current"],
                    "is_virtual": True,
                    "type": "virtual",
                })
    main_path.extend(new_path)


def visualize_result(square_size, final_path, filename="result"):
    print(f"\n[-] 正在生成可视化图片: {filename}.png")
    fig, ax = plt.subplots(figsize=(14, 14))
    ax.set_aspect("equal")

    half_size = square_size / 2.0
    ax.set_xlim(-half_size, half_size)
    ax.set_ylim(-half_size, half_size)

    ax.grid(True, alpha=0.3)
    ax.set_xlabel("X (mm)", fontsize=12)
    ax.set_ylabel("Y (mm)", fontsize=12)
    ax.set_title("Spot Melting Experiment - Annulus Array", fontsize=14, fontweight="bold")
    ax.add_patch(Rectangle(
        (-half_size, -half_size),
        square_size,
        square_size,
        fill=False,
        edgecolor="gray",
        linewidth=2,
    ))

    infill_pts = [p for p in final_path if not p["is_virtual"] and p.get("type") == "infill"]
    contour_pts = [p for p in final_path if not p["is_virtual"] and p.get("type") == "contour"]
    virt_pts = [p for p in final_path if p["is_virtual"]]

    if infill_pts:
        circles = [Circle((p["x"], p["y"]), 0.25) for p in infill_pts]
        ax.add_collection(PatchCollection(
            circles,
            facecolor="steelblue",
            edgecolor="navy",
            linewidth=0.5,
            alpha=0.7,
            zorder=100,
        ))

    if contour_pts:
        ax.scatter(
            [p["x"] for p in contour_pts],
            [p["y"] for p in contour_pts],
            c="orange",
            s=10,
            label="Contour",
            zorder=120,
        )

    if virt_pts:
        ax.scatter(
            [p["x"] for p in virt_pts],
            [p["y"] for p in virt_pts],
            c="red",
            marker="x",
            s=15,
            alpha=0.6,
            label="Virtual Jumps",
            zorder=80,
        )

    if final_path:
        all_x = [p["x"] for p in final_path]
        all_y = [p["y"] for p in final_path]
        ax.plot(all_x, all_y, "g-", alpha=0.2, linewidth=0.5, zorder=50)
        ax.plot(all_x[0], all_y[0], "o", color="lime", markersize=12, label="Start", zorder=200)
        ax.plot(all_x[-1], all_y[-1], "s", color="red", markersize=10, label="End", zorder=200)

    ax.legend(loc="upper right", fontsize=9)
    plt.savefig(f"{filename}.png", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[OK] 图片已保存: {filename}.png")


def main():
    print("=" * 60)
    print("  多变量环形阵列点扫控制台")
    print("=" * 60)

    print("\n[选项] 1. 运行已有 CSV 实验表  2. 生成新的 4 组测试 CSV 模板")
    choice = input("请选择 (默认1): ").strip() or "1"
    if choice == "2":
        create_template_csv()
        return

    csv_name = input("\n请输入 CSV 文件名 (默认 annulus_experiment.csv): ").strip() or "annulus_experiment.csv"
    if not csv_name.endswith(".csv"):
        csv_name += ".csv"

    configs = load_experiment_csv(csv_name)
    if not configs:
        return

    j_max = float(input("全局最大跳跃距离 j_max (mm, 默认 2.0): ").strip() or 2.0)
    sub_size = float(input("用于可视化图表的基板尺寸 (mm, 默认 150): ").strip() or 150)

    global_infill_path = []
    global_contour_path = []

    for i, cfg in enumerate(configs):
        infill_region, contour_region = process_single_region(cfg, region_idx=i, j_max=j_max)
        append_path_with_jump(global_infill_path, infill_region, j_max)
        append_path_with_jump(global_contour_path, contour_region, j_max)

    print("\n[-] 正在执行全局时序组装: 所有轮廓 -> 所有填充")

    global_path = []
    global_path.extend(global_contour_path)
    global_path.extend(global_infill_path)

    contour_count = sum(1 for p in global_path if p.get("type") == "contour")
    infill_count = sum(1 for p in global_path if p.get("type") == "infill")
    virt_count = sum(1 for p in global_path if p.get("is_virtual"))
    print(
        f"[OK] 实验矩阵生成完毕, 总路径点数: {len(global_path)} "
        f"(轮廓: {contour_count} | 填充: {infill_count} | 虚拟/缓冲: {virt_count})"
    )

    if input("\n导出二进制加工文件? (y/n, 默认y): ").strip().lower() != "n":
        contour_name = f"layer_{LAYER_INDEX}-gun_0-1-contour"
        fill_name = f"layer_{LAYER_INDEX}-gun_0-1-fill"
        export_to_binary(global_contour_path, contour_name)
        export_to_binary(global_infill_path, fill_name)

    if input("生成结果可视化图片? (y/n, 默认y): ").strip().lower() != "n":
        viz_name = input("图片名?(默认 experiment_result): ").strip() or "experiment_result"
        visualize_result(sub_size, global_path, filename=viz_name)


if __name__ == "__main__":
    main()
