# NewVision

基于 C++20、OpenCV 和 Eigen 的自瞄工程，使用 xmake 构建。

## Ubuntu 环境

推荐 Ubuntu 24.04（x86_64 或 arm64）和 GCC 13。基础依赖包括：

- xmake >= 2.9.8
- GCC/G++ 13
- OpenCV、Eigen3、yaml-cpp、libusb 开发包
- pkg-config

一键安装（会调用 `sudo apt-get`，并在需要时从 xmake 官网安装新版 xmake）：

```bash
./scripts/setup_ubuntu.sh --install
export PATH="$HOME/.local/bin:$PATH"
```

只检查环境、不修改系统：

```bash
./scripts/setup_ubuntu.sh --check
```

海康和迈德威视相机的头文件与 x86_64/arm64 动态库已放在
`tools/camera_sdk`，无需通过 apt 安装。运行相机相关程序仍需要相机驱动、
USB 权限和实际硬件。

## 构建

默认使用系统依赖，OpenVINO 后端关闭：

```bash
xmake f -m debug -y
xmake
```

构建主程序或单个冒烟测试：

```bash
xmake build auto_aim
xmake build logger_smoke
xmake run logger_smoke
```

若不想安装系统版 OpenCV/yaml-cpp，可让 xrepo 管理它们：

```bash
xmake f --use_system_deps=n --use_xrepo_deps=y -m debug -y
xmake
```

## 可选 OpenVINO

OpenVINO 不属于基础依赖。先按 Intel OpenVINO 的安装方式配置环境，并确认：

```bash
pkg-config --modversion openvino
```

再启用后端：

```bash
xmake f --use_openvino=y -m debug -y
xmake build openvino_armor_smoke
xmake run openvino_armor_smoke
```

如果 `pkg-config` 找不到 OpenVINO，请先加载其 `setupvars.sh`，或把
`openvino.pc` 所在目录加入 `PKG_CONFIG_PATH`。
