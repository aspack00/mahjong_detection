2026-02-07 22:21:08 新增 doc/分类对应关系.md（42 类麻将类别索引、原始分类与映射中文名，如三筒、四条）；新增 predict_batch.py，对 test/1/pics 三张图用 runs/detect/train2/weights/best.pt 预测，带框结果图写入 test/1/result，检测列表（序号、中心点像素/%、宽高、原始/映射分类）写入 test/1/result/结果.md。
2026-02-07 21:07:29 新增 doc/结果说明.md：说明 runs/detect/train2 下各 PNG/JPG 含义，以及如何通过 results.png、results.csv、BoxPR、混淆矩阵、val_*_pred 判断训练效果。
2026-02-07 12:14:48 training.ipynb：训练前用 settings.update 开启 TensorBoard 写入；在 TensorBoard 单元格前增加说明（需事件日志，PNG/CSV 不可用）。
2026-02-07 12:08:36 training.ipynb TensorBoard logdir 再次改为 runs（之前未生效），以同时显示 runs/train 与 runs/detect/train2。
2026-02-07 12:04:14 TensorBoard logdir 改为 runs，以同时显示 runs/train 与 runs/detect/train2；说明在 PowerShell 中结束 TensorBoard 进程用 Stop-Process -Id &lt;pid&gt;。
2026-02-05 23:56:09 修复 training.ipynb 在仅安装 CPU 版 PyTorch 时因 device=0 导致的 ValueError：按 torch.cuda.is_available() 自动选择 device 与 batch，无 GPU 时用 CPU 且 batch=8。
2026-02-05 23:30:51 使用 conda 创建 yolo11 环境（Python 3.10），安装 CPU 版 PyTorch、并在该环境中通过 requirements.txt 安装 ultralytics 等项目依赖，完成 YOLOv11 训练/推理运行环境准备。
2026-02-05 23:28:43 抽取 DATASET_VERSION 配置到 config.py，并让 training.ipynb 与 download_dataset.py 共享，避免版本号重复维护。
2026-02-05 23:24:55 使用 Roboflow API key 安装依赖并创建下载脚本，调试数据集版本配置；阅读训练 notebook 的数据路径配置。
2026-02-05 22:58:11 检查本机是否已安装 cursor CLI，结果：已安装（版本 2.4.27）
