# 通用几何与轮廓工具

只依赖 Eigen 和 OpenCV，不依赖能量机关的数据类型。

| 文件 | 功能 |
| --- | --- |
| `pose_projection.hpp/.cpp` | 三维点有限性、刚体变换校验、Rodrigues 旋转、正深度检查和含畸变的像素投影 |
| `contour.hpp/.cpp` | 有界 ROI 轮廓距离场、带截断的双线性采样、闭合轮廓按周长均匀采样 |

调用方添加 `tools` 到头文件搜索路径，使用 `#include "geometry/pose_projection.hpp"` 或 `#include "geometry/contour.hpp"`，函数位于 `tools::geometry` 命名空间。xmake 主库和能量机关独立测试已经配置这些依赖。

`PoseParameters` 顺序为 Rodrigues 旋转向量的三个分量和相机系平移的三个分量。旋转向量单位为弧度，平移与输入模型点使用同一长度单位。调用方提供有效的相机内参和畸变。

`makeContourDistanceField()` 在构建距离场前检查轮廓、坐标范围和像素预算；padding 范围为 0～4096 px。距离场为 `CV_32FC1`，`sampleDistanceField()` 接收原图像素坐标，在有效双线性采样范围外返回截断值。`appendContourSamples()` 向现有数组追加采样点，默认 96 点。

代码从能量机关三维重建层提取，算法来源与 MIT 许可见 [来源说明](../../src/rune/f2-building/README.md) 和 [RP-26Rune-LICENSE](../../src/rune/f2-building/RP-26Rune-LICENSE)。验证复用 `rune_building_smoke`、`rune_joint_building_smoke` 与 `rune_pipeline_smoke`。
