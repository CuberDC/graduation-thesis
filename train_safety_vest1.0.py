from ultralytics import YOLO
import os
import torch
import yaml
from torchvision import transforms

def load_data_yaml(yaml_path):
    with open(yaml_path, 'r') as file:
        data = yaml.safe_load(file)
    return data

def check_dataset_paths(data):
    # 检查 data.yaml 中定义的 train 和 val 路径是否存在
    for key in ['train', 'val']:
        if key in data:
            path = data[key]
            if not os.path.exists(path):
                print(f"Warning: Dataset path '{path}' does not exist. Please check the path.")
                os.makedirs(path, exist_ok=True)  # 创建缺失的目录
        else:
            print(f"Warning: '{key}' key not found in data.yaml.")

def check_labels(label_folder):
    # 检查标签文件是否存在
    for label_file in os.listdir(label_folder):
        if label_file.endswith('.txt'):
            print(f"✅ 找到标签文件: {label_file}")
        else:
            print(f"⚠️ 找到非标签文件: {label_file}")

def check_images(image_folder):
    # 检查图像文件是否存在
    image_files = [f for f in os.listdir(image_folder) if f.endswith(('.jpg', '.jpeg', '.png'))]
    if not image_files:
        print(f"⚠️ 在 {image_folder} 中未找到任何图像文件。")
    else:
        print(f"✅ 找到 {len(image_files)} 张图像文件。")

def check_dataset(dataset_root):
    # dataset_root 为包含 train, val, test 的根目录
    for split in ['train', 'val', 'test']:
        image_path = os.path.join(dataset_root, split, 'images')
        label_path = os.path.join(dataset_root, split, 'labels')

        if not os.path.exists(image_path) or not os.listdir(image_path):
            print(f"❌ 在路径 {image_path} 中未找到任何图像文件。请检查文件夹。")
        if not os.path.exists(label_path) or not os.listdir(label_path):
            print(f"❌ 在路径 {label_path} 中未找到任何标签文件。请检查文件夹。")

def train_safety_vest():
    # 设置数据集根目录（包含 train、val、test 文件夹）
    dataset_root = './dataset_split'
    train_path = os.path.join(dataset_root, 'train')

    # 检查训练集路径是否存在
    if not os.path.exists(train_path):
        print(f"❌ 训练集路径 {train_path} 不存在")
        return

    # 加载数据配置文件（请确保 data.yaml 中的路径与你的数据结构对应）
    data = load_data_yaml('data.yaml')
    check_dataset_paths(data)

    # 加载图像并应用随机增强
    data_transforms = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
    ])

    # 检查训练集的标签文件和图像文件
    check_labels(os.path.join(train_path, 'labels'))
    check_images(os.path.join(train_path, 'images'))

    # 加载预训练模型
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = YOLO('C:/Users/Donovan/graduation thesis/yolo11x.pt').to(device)

    # 配置训练参数
    training_params = {
        'data': 'data.yaml',
        'epochs': 100,
        'batch': 16,
        'imgsz': 640,
        'device': device,
        'lr0': 0.0005,
    }

    # 开始训练
    model.train(**training_params)

    # 保存训练后的模型为 best.pt
    new_model_path = 'best.pt'
    model.save(new_model_path)
    print(f"✅ 模型已保存为: {new_model_path}")

    # 评估模型性能
    results = model.val(device=device)
    precision = results.p[0]   # Precision
    recall = results.r[0]      # Recall
    mAP50 = results.mAP[0]     # mAP@0.5
    mAP50_95 = results.mAP50_95  # mAP@0.5:0.95

    print("✅ 评估结果：")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"mAP50: {mAP50:.4f}")
    print(f"mAP50-95: {mAP50_95:.4f}")

    # 检查整个数据集（train、val、test）
    check_dataset(dataset_root)

if __name__ == '__main__':
    train_safety_vest()
