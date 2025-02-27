import cv2
import pygame
import os
import torch
import time
from ultralytics import YOLO

# 初始化 pygame 音频模块
pygame.mixer.init()
if not pygame.mixer.get_init():
    print("❌ Pygame 音频初始化失败")
    exit()
else:
    print("✅ Pygame 音频初始化成功")

# 检查 GPU 是否可用
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🚀 运行设备: {device}")

# 加载 YOLO 模型
model_path = "best.pt"  # 替换为你的模型文件路径
if not os.path.exists(model_path):
    print(f"❌ 模型文件 {model_path} 未找到")
    exit()
model = YOLO(model_path).to(device)

# 声音文件路径
sound_file = r".\alert_vest.mp3"
if not os.path.exists(sound_file):
    print(f"❌ 警报音频 {sound_file} 未找到")
    exit()
alert_sound = pygame.mixer.Sound(sound_file)

# 变量：防止声音重复播放
is_playing_sound = False

# 创建保存图片的目录（用于保存未穿安全背心的图片）
save_dir = "no_safety_gear"
os.makedirs(save_dir, exist_ok=True)

# 打开摄像头
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ 无法打开摄像头")
    exit()

# 获取摄像头当前默认分辨率
default_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
default_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"📸 当前摄像头默认分辨率: {default_width}x{default_height}")

# 如果你希望修改分辨率，请在下面输入你想要的宽和高（例如：1280 720），否则直接回车使用默认分辨率：
user_input = input("请输入想要的分辨率（宽 高），或直接回车使用默认分辨率：")
if user_input.strip():
    try:
        width, height = map(int, user_input.strip().split())
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        frame_width, frame_height = width, height
    except Exception as e:
        print("输入错误，使用默认分辨率")
        frame_width, frame_height = default_width, default_height
else:
    frame_width, frame_height = default_width, default_height

# 设置摄像头帧率
cap.set(cv2.CAP_PROP_FPS, 30)
print(f"📸 摄像头分辨率设置为: {frame_width}x{frame_height}, 30 FPS")

# 创建显示窗口，并确保尺寸一致
cv2.namedWindow("安全检测系统", cv2.WINDOW_NORMAL)
cv2.resizeWindow("安全检测系统", frame_width, frame_height)

# 定义检测目标的颜色（注意：安全背心统一设为类别 1）
class_colors = {
    0: (255, 0, 0),   # Person（人）- 红色
    1: (0, 0, 255),   # Safety-vest（安全背心）- 蓝色
    8: (0, 255, 0)    # Glasses（眼镜）- 绿色
}

last_save_time = 0
save_interval = 10  # 每 10 秒保存一张未穿安全背心的图片
frame_id = 0

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ 无法从摄像头读取画面")
            break

        frame_id += 1
        # 保证 YOLO 处理后的图像尺寸与摄像头一致
        frame = cv2.resize(frame, (frame_width, frame_height))

        # YOLO 进行检测
        results = model(frame)

        safety_vest_detected = False
        person_detected = False

        # 遍历检测到的物体
        for result in results:
            for box in result.boxes:
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                # 判断是否检测到安全背心或人
                if class_id == 1:  # 安全背心
                    safety_vest_detected = True
                if class_id == 0:  # 人
                    person_detected = True

                # 只对人、眼镜和安全背心绘制检测框
                if class_id in [0, 1, 8]:
                    if class_id == 0:
                        label = "Person"
                    elif class_id == 1:
                        label = "Safety-vest"
                    else:
                        label = "Glasses"
                    color = class_colors.get(class_id, (255, 255, 255))
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, f"{label} {confidence:.2f}", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # 警报逻辑：检测到人但未检测到安全背心时触发警报
        current_time = time.time()
        if person_detected and not safety_vest_detected:
            if current_time - last_save_time >= save_interval:
                img_path = os.path.join(save_dir, f"frame_{frame_id}.jpg")
                cv2.imwrite(img_path, frame)
                print(f"💾 已保存未穿安全背心的图片: {img_path}")
                last_save_time = current_time

            # 若警报未在播放则启动警报（循环播放）
            if not is_playing_sound:
                print("🚨 检测到人员未穿安全背心，播放警报...")
                alert_sound.play(-1)
                is_playing_sound = True
        else:
            # 当检测到安全背心时，若警报正在播放则停止播放
            if is_playing_sound:
                print("✅ 安全背心已检测到，停止警报")
                pygame.mixer.stop()
                is_playing_sound = False

        # 显示检测结果
        cv2.imshow('安全检测系统', frame)

        # 按 ESC 键退出
        if cv2.waitKey(1) & 0xFF == 27:
            break

finally:
    # 释放摄像头和相关资源
    pygame.mixer.stop()
    pygame.mixer.quit()
    cap.release()
    cv2.destroyAllWindows()
    print("📴 摄像头已关闭")
