import os
import shutil


def batch_copy_abab_layers():
    # ================= 配置区域 =================
    # 路径沿用你之前的（注意：如果文件夹名改了，请在这里更新）
    base_dir = r"D:\spot melting sequence\实验\第四次-新顺序-零件\20260311"

    total_layers = 200


    # 文件名后缀
    suffix = "-gun_0-1-fill.bin"
    # ===========================================

    # 定义两个源文件路径
    source_odd = os.path.join(base_dir, f"layer_1{suffix}")  # 模板 A
    source_even = os.path.join(base_dir, f"layer_2{suffix}")  # 模板 B

    # 1. 专家检查：必须确保两个模板都存在
    if not os.path.exists(source_odd):
        print(f"❌ 错误：缺少基数层模板：{source_odd}")
        return
    if not os.path.exists(source_even):
        print(f"❌ 错误：缺少偶数层模板：{source_even}")
        print("提示：对于 HCP 结构，你必须手动先画好第 2 层(Layer 2)作为 B 层模板。")
        return

    print(f"📂 工作目录: {base_dir}")
    print(f"✅ 检测到 Layer 1 (A) 和 Layer 2 (B)，开始执行 ABAB 堆叠复制...")

    count_a = 0
    count_b = 0

    # 2. 循环生成 (从第 3 层开始，一直到第 200 层)
    for i in range(3, total_layers + 1):
        target_filename = f"layer_{i}{suffix}"
        target_full_path = os.path.join(base_dir, target_filename)

        try:
            # 判断奇偶性
            if i % 2 == 1:
                # 奇数层 (3, 5, 7...) -> 复制 Layer 1
                shutil.copy2(source_odd, target_full_path)
                # print(f"Layer {i} (A) Created")
                count_a += 1
            else:
                # 偶数层 (4, 6, 8...) -> 复制 Layer 2
                shutil.copy2(source_even, target_full_path)
                # print(f"Layer {i} (B) Created")
                count_b += 1

        except Exception as e:
            print(f"⚠️ 生成 Layer {i} 失败: {e}")

    # 3. 总结
    print("-" * 30)
    print(f"处理完成 (HCP - ABAB 模式)")
    print(f"基数层 (A) 新增: {count_a} 个 (复制自 layer_1)")
    print(f"偶数层 (B) 新增: {count_b} 个 (复制自 layer_2)")
    print(f"总层数: {total_layers}")


if __name__ == "__main__":
    batch_copy_abab_layers()
    input("\n按回车键退出...")