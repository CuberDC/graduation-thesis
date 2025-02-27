from ultralytics import YOLO

def main():
    model_path = "./best.pt"  # 请替换为你的模型文件路径
    print("正在加载模型...")
    model = YOLO(model_path)

    print("开始验证模型...")
    results = model.val(data="data.yaml", imgsz=640, conf=0.001, verbose=True)

    # 输出完整的验证结果对象（DetMetrics 对象）
    print("\n完整测试结果对象：")
    print(results)

    # 提取并打印详细的测试指标
    print("\n详细测试指标：")
    metrics = results.results_dict
    for key, value in metrics.items():
        print(f"{key}: {value}")

    # 或者单独打印关键指标
    print("\n关键指标提取：")
    print("Precision:", metrics.get('metrics/precision(B)'))
    print("Recall:", metrics.get('metrics/recall(B)'))
    print("mAP50:", metrics.get('metrics/mAP50(B)'))
    print("mAP50-95:", metrics.get('metrics/mAP50-95(B)'))
    print("Fitness:", metrics.get('fitness'))

    # 打印其它有用信息
    print("\n其它信息：")
    print("保存结果目录：", results.save_dir)
    print("速度信息：", results.speed)
    print("任务类型：", results.task)
    print("类别名称：", results.names)

if __name__ == "__main__":
    main()
