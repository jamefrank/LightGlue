import torch
import os
from lightglue import LightGlue, SuperPoint, DISK
from lightglue.utils import load_image, rbd
from lightglue import viz2d
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde
import cv2
"""
评价指标、算法设计、流程梳理的建议（嗯嗯不错好70%，豆包大模型30%）
算法实现（豆包大模型：贡献度100%）
提示词引导工程（CSDN:嗯嗯不错好100%）
CSDN链接：https://blog.csdn.net/lrx6666666）
"""
# 设置中文字体
plt.rcParams['font.family'] = 'WenQuanYi Zen Hei'  # 文泉驿正黑
# 解决负号显示问题
plt.rcParams['axes.unicode_minus'] = False
 
 
def initialize_model():
    """初始化特征提取器和匹配器"""
    extractor = SuperPoint(max_num_keypoints=8192).eval().cuda()
    matcher = LightGlue(features='superpoint').eval().cuda()
    return extractor, matcher
 
 
def load_and_process_images(extractor, matcher, image_path1, image_path2):
    """加载图像，提取特征并进行匹配"""
    try:
        image0 = load_image(image_path1).cuda()
        image1 = load_image(image_path2).cuda()
        feats0 = extractor.extract(image0)
        feats1 = extractor.extract(image1)
        matches01 = matcher({'image0': feats0, 'image1': feats1})
        feats0, feats1, matches01 = [rbd(x) for x in [feats0, feats1, matches01]]
        matches = matches01['matches']
        points0 = feats0['keypoints'][matches[..., 0]].cpu().numpy()
        points1 = feats1['keypoints'][matches[..., 1]].cpu().numpy()
        scores = matches01['scores'].detach().cpu().numpy()
        return image0, image1, feats0, feats1, points0, points1, scores
    except Exception as e:
        print(f"图像加载、特征提取或匹配过程中出现错误: {e}")
        return None, None, None, None, None, None, None
 
 
def save_combined_images(image0, image1, output_path):
    """保存合并的原始图像"""
    plt.figure(figsize=(12, 6))
    viz2d.plot_images([image0, image1])
    plt.axis('off')
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()
 
 
def save_feature_points_image(image0, image1, feats0, feats1, output_path):
    """保存特征点图像"""
    plt.figure(figsize=(12, 6))
    viz2d.plot_images([image0, image1])
    viz2d.plot_keypoints([
        feats0['keypoints'].cpu().numpy(),
        feats1['keypoints'].cpu().numpy()
    ], colors=['red', 'blue'], ps=8)
    plt.title('特征点分布图')
    plt.axis('off')
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()
 
 
def save_matched_points_image(image0, image1, points0, points1, scores, output_path):
    """保存特征点连线图像，根据匹配得分用不同颜色连接"""
    plt.figure(figsize=(12, 6))
    viz2d.plot_images([image0, image1])
 
    # 定义颜色映射
    color_map = {
        (0.0, 0.2): 'red',
        (0.2, 0.4): 'yellow',
        (0.4, 0.6): 'green',
        (0.6, 0.8): 'blue',
        (0.8, 1.0): 'purple'
    }
 
    for score_range, color in color_map.items():
        low_score, high_score = score_range
        mask = (scores >= low_score) & (scores < high_score)
        filtered_points0 = points0[mask]
        filtered_points1 = points1[mask]
        if len(filtered_points0) > 0:
            viz2d.plot_matches(filtered_points0, filtered_points1, color=color, lw=0.5)
 
    plt.title('特征点匹配连线图（根据得分区分颜色）')
    plt.axis('off')
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()
 
 
def create_heatmap(data, h, w):
    """创建热力图数据"""
    heatmap = np.zeros((h, w))
    for point in data:
        x, y = int(point[0]), int(point[1])
        if 0 <= x < w and 0 <= y < h:
            heatmap[y, x] += 1
    return heatmap
 
 
def normalize_and_smooth(heatmap):
    """归一化和平滑处理热力图"""
    normalized = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
    smoothed = cv2.GaussianBlur(normalized, (5, 5), 0)
    return smoothed
 
 
def save_heatmap(image_np, heatmap, title, label, output_path):
    """保存热力图"""
    plt.figure(figsize=(12, 6))
    plt.imshow(image_np, cmap='gray')
    im = plt.imshow(heatmap, cmap='viridis', alpha=0.5)
    plt.title(title)
    plt.axis('off')
    plt.colorbar(im, label=label)
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()
 
 
def save_combined_heatmap(image0_np, image1_np, heatmap0, heatmap1, title, label, output_path):
    """保存合并的热力图"""
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(image0_np, cmap='gray')
    im0 = plt.imshow(heatmap0, cmap='viridis', alpha=0.5)
    plt.title(f'{title}（图像0）')
    plt.axis('off')
    plt.colorbar(im0, label=label)
 
    plt.subplot(1, 2, 2)
    plt.imshow(image1_np, cmap='gray')
    im1 = plt.imshow(heatmap1, cmap='viridis', alpha=0.5)
    plt.title(f'{title}（图像1）')
    plt.axis('off')
    plt.colorbar(im1, label=label)
 
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()
 
 
def process_image_pair(extractor, matcher, image_path1, image_path2, output_dir):
    """处理一对图像"""
    image0, image1, feats0, feats1, points0, points1, scores = load_and_process_images(
        extractor, matcher, image_path1, image_path2)
    if image0 is None or image1 is None:
        return
 
    h, w = image0.shape[-2:]
    image0_np = np.transpose(image0.cpu().squeeze(0).numpy(), (1, 2, 0))
    image1_np = np.transpose(image1.cpu().squeeze(0).numpy(), (1, 2, 0))
 
    base_name = os.path.splitext(os.path.basename(image_path1))[0]
    pair_output_dir = os.path.join(output_dir, base_name)
    os.makedirs(pair_output_dir, exist_ok=True)
 
    # 保存合并的原始图像
    save_combined_images(image0, image1, os.path.join(pair_output_dir, f'{base_name}_combined_images.png'))
 
    # 保存特征点图像
    save_feature_points_image(image0, image1, feats0, feats1, os.path.join(pair_output_dir, f'{base_name}_feature_points.png'))
 
    # 保存特征点连线图像
    save_matched_points_image(image0, image1, points0, points1, scores, os.path.join(pair_output_dir, f'{base_name}_matched_points.png'))
 
    # 特征点密度热力图
    density_map0 = create_heatmap(feats0['keypoints'].cpu().numpy(), h, w)
    density_map1 = create_heatmap(feats1['keypoints'].cpu().numpy(), h, w)
    density_map0_smoothed = normalize_and_smooth(density_map0)
    density_map1_smoothed = normalize_and_smooth(density_map1)
    save_heatmap(image0_np, density_map0_smoothed, '图像0特征点密度热力图', '归一化密度',
                 os.path.join(pair_output_dir, f'{base_name}_feature_point_density_map_0.png'))
    save_heatmap(image1_np, density_map1_smoothed, '图像1特征点密度热力图', '归一化密度',
                 os.path.join(pair_output_dir, f'{base_name}_feature_point_density_map_1.png'))
    save_combined_heatmap(image0_np, image1_np, density_map0_smoothed, density_map1_smoothed,
                          '特征点密度热力图', '归一化密度',
                          os.path.join(pair_output_dir, f'{base_name}_feature_point_density_map.png'))
 
    # 匹配分数热力图
    score_map0 = np.zeros((h, w))
    score_map1 = np.zeros((h, w))
    for i, (p0, p1) in enumerate(zip(points0, points1)):
        x0, y0 = int(p0[0]), int(p0[1])
        x1, y1 = int(p1[0]), int(p1[1])
        if 0 <= x0 < w and 0 <= y0 < h:
            score_map0[y0, x0] = scores[i]
        if 0 <= x1 < w and 0 <= y1 < h:
            score_map1[y1, x1] = scores[i]
    score_map0_smoothed = normalize_and_smooth(score_map0)
    score_map1_smoothed = normalize_and_smooth(score_map1)
    save_heatmap(image0_np, score_map0_smoothed, '图像0匹配分数热力图', '归一化分数',
                 os.path.join(pair_output_dir, f'{base_name}_matching_score_heatmap_0.png'))
    save_heatmap(image1_np, score_map1_smoothed, '图像1匹配分数热力图', '归一化分数',
                 os.path.join(pair_output_dir, f'{base_name}_matching_score_heatmap_1.png'))
    save_combined_heatmap(image0_np, image1_np, score_map0_smoothed, score_map1_smoothed,
                          '匹配分数热力图', '归一化分数',
                          os.path.join(pair_output_dir, f'{base_name}_matching_score_heatmap.png'))
 
    # 匹配密度热力图
    x0 = points0[:, 0]
    y0 = points0[:, 1]
    xy0 = np.vstack([x0, y0])
    z0 = gaussian_kde(xy0)(xy0)
 
    x1 = points1[:, 0]
    y1 = points1[:, 1]
    xy1 = np.vstack([x1, y1])
    z1 = gaussian_kde(xy1)(xy1)
 
    density_map_matching0 = np.zeros((h, w))
    density_map_matching1 = np.zeros((h, w))
    for i, (x, y) in enumerate(zip(x0, y0)):
        x, y = int(x), int(y)
        if 0 <= x < w and 0 <= y < h:
            density_map_matching0[y, x] = z0[i]
 
    for i, (x, y) in enumerate(zip(x1, y1)):
        x, y = int(x), int(y)
        if 0 <= x < w and 0 <= y < h:
            density_map_matching1[y, x] = z1[i]
 
    density_map_matching0_smoothed = normalize_and_smooth(density_map_matching0)
    density_map_matching1_smoothed = normalize_and_smooth(density_map_matching1)
    save_heatmap(image0_np, density_map_matching0_smoothed, '图像0匹配密度热力图', '归一化密度',
                 os.path.join(pair_output_dir, f'{base_name}_matching_density_heatmap_0.png'))
    save_heatmap(image1_np, density_map_matching1_smoothed, '图像1匹配密度热力图', '归一化密度',
                 os.path.join(pair_output_dir, f'{base_name}_matching_density_heatmap_1.png'))
    save_combined_heatmap(image0_np, image1_np, density_map_matching0_smoothed, density_map_matching1_smoothed,
                          '匹配密度热力图', '归一化密度',
                          os.path.join(pair_output_dir, f'{base_name}_matching_density_heatmap.png'))
 
    print(f"处理完成，结果已保存至 {pair_output_dir} 目录：")
    print(f"├─ {base_name}_combined_images.png    # 合并的原始图像")
    print(f"├─ {base_name}_feature_points.png     # 特征点分布图")
    print(f"├─ {base_name}_matched_points.png     # 特征点匹配连线图")
    print(f"├─ {base_name}_feature_point_density_map_0.png # 图像0特征点密度热力图")
    print(f"├─ {base_name}_feature_point_density_map_1.png # 图像1特征点密度热力图")
    print(f"├─ {base_name}_feature_point_density_map.png # 特征点密度热力图")
    print(f"├─ {base_name}_matching_score_heatmap_0.png # 图像0匹配分数热力图")
    print(f"├─ {base_name}_matching_score_heatmap_1.png # 图像1匹配分数热力图")
    print(f"├─ {base_name}_matching_score_heatmap.png # 匹配分数热力图")
    print(f"├─ {base_name}_matching_density_heatmap_0.png # 图像0匹配密度热力图")
    print(f"├─ {base_name}_matching_density_heatmap_1.png # 图像1匹配密度热力图")
    print(f"└─ {base_name}_matching_density_heatmap.png # 匹配密度热力图")
 
 
def main():
    # 图像目录
    after_dir = 'data/test/1'
    bef_dir = 'data/test/0'
    output_dir = 'data/test/output1'
    os.makedirs(output_dir, exist_ok=True)
 
    # 初始化模型
    extractor, matcher = initialize_model()
 
    # 获取图像文件列表
    after_files = sorted([os.path.join(after_dir, f) for f in os.listdir(after_dir)])
    bef_files = sorted([os.path.join(bef_dir, f) for f in os.listdir(bef_dir)])
 
    if len(after_files) != len(bef_files):
        print("前后图像文件数量不一致，请检查目录。")
        return
 
    # 批量处理图像对
    for image_path1, image_path2 in zip(after_files, bef_files):
        process_image_pair(extractor, matcher, image_path1, image_path2, output_dir)
 
 
if __name__ == "__main__":
    main()

