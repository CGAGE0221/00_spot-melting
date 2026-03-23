import os
# 导入你底层的轮廓和填充模块 (需要确保这三个文件在同一目录下)
from a_contour_algo import ContourGenerator
from a_infill_algo import (CLIReader, build_geometry_from_layer, generate_substrate_spots,
                         assign_groups, filter_spots_in_geometry, generate_melting_path,
                         build_final_path_with_virtuals, insert_virtuals_for_path,
                         export_to_binary, visualize_result)

def main():
    print("=" * 60)
    print("  电子束点扫实验操作台 (主控程序)")
    print("=" * 60)

    # ========== 1. 用户交互：切片文件与基板输入 ==========
    print("\n【步骤1】输入基础信息")
    cli_path = input("  请输入 CLI 切片文件路径: ").strip().replace('"', '').replace("'", "")
    reader = CLIReader(cli_path)
    if not reader.read():
        return

    z_levels = reader.get_all_z_levels()
    layer_idx = int(input(f"  请选择加工层索引 (0~{len(z_levels) - 1}): ").strip())
    z_val, layer_data = reader.get_layer_by_index(layer_idx)
    geometry = build_geometry_from_layer(layer_data)
    if geometry is None or geometry.is_empty:
        print("  错误：未能在该层读取到有效的几何边界。")
        return

    square_size = float(input("  基板边长 (mm, 默认200): ").strip() or 200)
    spot_distance = float(input("  网格间距 (mm, 默认1.0): ").strip() or 1.0)
    pattern_type = input("  点阵类型 (sc/hcp, 默认sc): ").strip().lower() or 'sc'
    n_groups = int(input("  分组数 (默认36): ").strip() or 36)

    # ========== 2. 用户交互：实验变量设置 (DoE) ==========
    print("\n【步骤2】输入实验参数")
    print("  --- 内部填充参数 (Infill) ---")
    infill_dwell = float(input("  填充点停留时间 (μs, 默认333): ").strip() or 333)
    infill_bc = float(input("  填充束流大小 (mA, 默认5.5): ").strip() or 5.5)
    j_max = float(input("  最大跳跃距离 (mm, 默认2.0): ").strip() or 2.0)

    print("\n  --- 外部轮廓参数 (Contour) ---")
    contour_ss = float(input("  轮廓点间距 (mm, 默认0.10): ").strip() or 0.10)
    contour_ls = float(input("  轮廓线间距 (mm, 默认0.09): ").strip() or 0.09)
    contour_dwell = float(input("  轮廓点停留时间 (μs, 默认30): ").strip() or 30)
    contour_bc = float(input("  轮廓束流大小 (mA, 默认8.0): ").strip() or 8.0)

    # ========== 3. 底层算法调度 (自动运行，无需干预) ==========
    print("\n【步骤3】系统正在自动计算轮廓与填充路径...")

    # 3.1 实例化底层的轮廓生成器
    contour_gen = ContourGenerator(geometry, contour_ss, contour_ls, contour_dwell, contour_bc)

    # 3.2 生成初始基板并进行预过滤
    spots = generate_substrate_spots(square_size, spot_distance, pattern_type, layer_idx)
    group_map, label_map = assign_groups(spots, n_groups, spot_distance, square_size, pattern_type)
    raw_indices, _ = filter_spots_in_geometry(spots, geometry)

    if not raw_indices:
        print("  错误：几何体内无可用熔化点，请检查网格参数。")
        return

    # 3.3 轮廓与填充配合 (执行策略B6：防过冲投影)
    melting_indices = contour_gen.project_and_filter_infill(spots, raw_indices)
    melting_spots_set = set(melting_indices)

    # 3.4 调度底层模块生成各自的路径
    infill_sequence = generate_melting_path(spots, group_map, melting_spots_set, n_groups, j_max)
    infill_path = build_final_path_with_virtuals(spots, infill_sequence, infill_dwell, infill_bc, label_map)
    contour_path = contour_gen.generate_full_contour_sequence()

    # 3.5 按照物理加工时序合并路径，并插入全局长距离虚拟跳转点
    combined_path = infill_path + contour_path
    final_path = insert_virtuals_for_path(combined_path, j_max)

    # ========== 4. 用户交互：导出与检查 ==========
    print("\n【步骤4】计算完成，请选择输出")
    infill_count = sum(1 for p in final_path if p.get('type') == 'infill')
    contour_count = sum(1 for p in final_path if p.get('type') == 'contour')
    virt_count = sum(1 for p in final_path if p.get('is_virtual'))
    print(f"  统计结果: {infill_count} 填充点 + {contour_count} 轮廓点 + {virt_count} 虚拟点")

    if input("  -> 导出二进制加工文件? (y/n, 默认y): ").strip().lower() != 'n':
        bin_name = input("  文件名 (默认output): ").strip() or "output"
        export_to_binary(final_path, bin_name)

    if input("  -> 生成结果可视化图片? (y/n, 默认y): ").strip().lower() != 'n':
        viz_name = input("  图片名 (默认result): ").strip() or "result"
        visualize_result(square_size, geometry, spots, melting_spots_set, final_path, z_val, viz_name)

if __name__ == "__main__":
    main()