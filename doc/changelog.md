2026-02-05 23:56:09 修复 training.ipynb 在仅安装 CPU 版 PyTorch 时因 device=0 导致的 ValueError：按 torch.cuda.is_available() 自动选择 device 与 batch，无 GPU 时用 CPU 且 batch=8。
2026-02-05 23:30:51 使用 conda 创建 yolo11 环境（Python 3.10），安装 CPU 版 PyTorch、并在该环境中通过 requirements.txt 安装 ultralytics 等项目依赖，完成 YOLOv11 训练/推理运行环境准备。
2026-02-05 23:28:43 抽取 DATASET_VERSION 配置到 config.py，并让 training.ipynb 与 download_dataset.py 共享，避免版本号重复维护。
2026-02-05 23:24:55 使用 Roboflow API key 安装依赖并创建下载脚本，调试数据集版本配置；阅读训练 notebook 的数据路径配置。
2026-02-05 22:58:11 检查本机是否已安装 cursor CLI，结果：已安装（版本 2.4.27）
