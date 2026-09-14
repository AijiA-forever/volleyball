# 排球训练智能分析系统（Web 整合版）

本版本在 5 个月前的半成品（RAR）基础上，融合 `volleyball_analytics` 的分层思想与标定记录设计，
按项目计划书要求补齐“教练端 + 学生端”双角色功能，最大程度覆盖：
垫球/发球/传球/扣球动作识别、规范评分、自然语言反馈、标准动作库、训练记录、报表统计与视角标定。

## 快速启动（推荐）

1. 打开 `C:\Projects1\volleyball_system_v2`
2. 双击 `启动系统.bat`，或手动运行：

```bat
C:\miniconda\envs\volleyball_analytics\python.exe -m uvicorn main:app --host 127.0.0.1 --port 8000
```

3. 浏览器访问 <http://127.0.0.1:8000>
4. 点击右上角进入系统，选择“学生”或“教练”

## 角色功能

### 学生端
- 练习分析：上传训练视频 / 图片 / 摄像头实时检测
- 标准动作学习：观看教练上传的标准视频，支持 0.5x / 0.25x 慢放和关键帧跳转
- 我的训练记录：查看每次训练的得分、动作数、标准数与逐动作反馈

### 教练端
- 标准动作库管理：新建 / 编辑 / 删除标准动作，上传示范视频，在线调整角度阈值参数（JSON）
- 训练报表：学生列表、规范率、平均分、常见问题 TOP
- 视角标定：上传含场地的画面，用画布标注场地端点并保存到 SQLite

## 数据说明

- SQLite 数据库：`data/volleyball.db`
- 模型权重：`pose_best.pt`（姿态）+ `best.pt`（排球），与半成品相同
- 分析结果：`results/videos`、`results/images`
- 标准动作示范视频：`data/standards/videos`

## 主要接口

| 方法 | 路径 | 说明 |
| ---- | ---- | ---- |
| POST | /api/login | 学生/教练角色登录 |
| GET | /api/standards | 标准动作列表 |
| POST/PUT/DELETE | /api/standards | 标准动作增删改 |
| POST | /api/analyze/video | 视频动作分析并落库 |
| POST | /api/analyze/image | 图片动作分析并落库 |
| GET | /api/sessions | 训练记录列表 |
| GET | /api/reports/overview | 整体报表 |
| GET | /api/reports/student/{name} | 单个学生报表 |
| GET | /api/students | 学生列表 |
| WS | /ws/camera | 摄像头实时分析 |
| POST | /api/calibration | 保存视角标定 |

## 代码结构

- `main.py` —— FastAPI 后端
- `analysis_service.py` —— 分析引擎与训练记录落库编排
- `storage.py` —— SQLite 数据层
- `analyzer.py` / `body.py` / `pose_judge.py` / `volleyball_detect.py` —— 视觉分析引擎（源自半成品）
- `image_preprocessing.py` —— 图像预处理模块
- `index.html` —— 双角色前端

## 与项目计划书的对应

- 标准动作库（SQLite CRUD）—— `storage.py` + 教练端页面
- 12 个核心关键点 / 角度评分 / 文字反馈 —— `body.py` + `pose_judge.py`
- 学生训练记录与规范率报表 —— `training_sessions` / `session_attempts` 表 + 报表页面
- 排球检测与垫球次数 —— `volleyball_detect.py` + `analyzer.py`
- 坐标校准记录 —— 视角标定页面 + `calibrations` 表
- Web 形态替代原计划书中的 PySide6（已按你的要求保留浏览器端，便于现场演示）

## 部署与文件清单

陌生电脑部署、完整文件清单、视频分析操作步骤见：[项目文件清单与运行指南.md](docs/项目文件清单与运行指南.md)


## 扩展文档

- [实时分析功能说明与理论来源.md](docs/实时分析功能说明与理论来源.md)
- [产品化与软件工程全流程说明.md](docs/产品化与软件工程全流程说明.md)
- [系统维护与二次开发指南.md](docs/系统维护与二次开发指南.md)
- [项目文件清单与运行指南.md](docs/项目文件清单与运行指南.md)


## 账号与多机访问

- 默认教练账号：`coach` / `coach123`（请立即修改）
- 已有学生账号：`曲浩宾`、`刘浩彬`，默认密码 `123456`
- 教练注册码：`gdut2026`（可用环境变量 VB_COACH_CODE 修改）
- 同一账号只能一台设备在线；新设备登录会使旧设备自动退出
- 局域网多机访问见：[账号权限与多机访问说明.md](docs/账号权限与多机访问说明.md)

- [多人跟踪、双视角校正与数据集训练说明.md](docs/多人跟踪_双视角_数据集训练说明.md)

- 局域网摄像头请用 HTTPS：先运行 `生成HTTPS证书.bat`，再用 `启动系统_HTTPS.bat` 启动，访问 `https://本机IP:8000`

## 结题与产品化

- [结题与产品化路线图.md](docs/结题与产品化路线图.md)


## 环境变量（性能与设备）

| 变量 | 默认 | 说明 |
| ---- | ---- | ---- |
| `VB_DEVICE` | 自动 | `cpu` / `0` / `cuda:0` / `gpu`，强制指定推理设备 |
| `VB_POSE_IMGSZ` | 640 | 姿态模型输入尺寸，调小更快、精度略降 |
| `VB_POSE_CONF` | 0.3 | 姿态关键点置信度阈值 |
| `VB_BALL_IMGSZ` | 640 | 排球检测输入尺寸 |
| `VB_BALL_CONF` | 0.5 | 排球检测置信度阈值 |
| `VB_BALL_INTERVAL` | 1 | 实时分析每 N 帧检测一次排球。默认 1 保证动作计数准确；设为 2 可提速，但实测会明显少算动作 |
| `VB_BALL_INTERVAL_VIDEO` | 1 | 离线视频分析每 N 帧检测一次排球（默认 1，精度优先） |
| `VB_DEBUG_LOG` | 0 | 设为 1 打开逐帧调试打印 |

示例（CPU 且想更快）：

```bat
set VB_POSE_IMGSZ=480
set VB_BALL_IMGSZ=480
set VB_BALL_INTERVAL=1
python -m uvicorn main:app --host 0.0.0.0 --port 8000
```
