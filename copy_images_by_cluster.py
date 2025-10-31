import os
import sys
import json
import shutil
from tqdm import tqdm


def copy_images_by_cluster(clustering_results_path, images_root_dir, output_dir):

    # 确保输出目录存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"{output_dir}目录创建成功")
    else:
        print(f"{output_dir}目录已存在")

    # 复制源目录结构到目标目录
    copy_struct_path = os.path.join(images_root_dir, 'test')
    dirs = [entry for entry in os.listdir(copy_struct_path) if os.path.isdir(os.path.join(copy_struct_path, entry))]
    for dir_name in dirs:
        copy_path = os.path.join(output_dir, dir_name)
        if not os.path.exists(copy_path):
            try:
                os.mkdir(copy_path)
            except OSError as e:
                print(f"创建目录失败: {e}")

    # 加载聚类结果
    with open(clustering_results_path, "r") as json_file:
        clustering_results = json.load(json_file)

    pbar = tqdm(total=len(dirs), file=sys.stdout)  # 主进度条
    for class_name in dirs:
        pbar.set_description(f"[Processing {class_name}]")
        if class_name not in clustering_results:
            raise ValueError(f"类别 '{class_name}' 在聚类结果中不存在")
        class_clustering_results = clustering_results[class_name]
        pbar2 = tqdm(total=len(class_clustering_results), file=sys.stdout, leave=False)
        for kmeans_label, path, _ in class_clustering_results:  # 按照聚类标签组织文件
            cluster_dir = os.path.join(output_dir, class_name, str(kmeans_label))
            if not os.path.exists(cluster_dir):  # 创建或确保目标文件夹存在
                os.makedirs(cluster_dir)

            source_path = os.path.join(images_root_dir, path)
            target_path = os.path.join(cluster_dir, os.path.basename(path))
            pbar2.set_description(f"[Copying {path}]")

            shutil.copy2(source_path, target_path)  # 复制图像文件
            pbar2.update()
        pbar2.set_description(f"[Copying Finished]")
        pbar2.close()
        pbar.update()
    pbar.set_description("[All Finished]")
    pbar.close()


if __name__ == "__main__":
    # 设置参数
    clustering_results_path = './clustering_results_12.json'  # 聚类结果文件路径
    images_root_dir = '../../data/minerals/'  # 图像文件根目录
    output_dir = '../../data/minerals/organized_clusters_12/'  # 输出文件夹路径

    # 执行函数
    copy_images_by_cluster(clustering_results_path, images_root_dir, output_dir)
