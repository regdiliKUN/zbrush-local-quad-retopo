# LocalQuadRetopo · 局部四边重拓扑

ZBrush 2022 / 2025 plug-in that rebuilds **only the masked area** of a mesh as quads. Everything outside the mask stays untouched, and the new patch meets the old mesh point for point along the mask border: no gaps, no overlaps.

ZBrush 2022 / 2025 插件：只把**遮罩区域**重新布成四边面，遮罩外的面保持原样，新旧两部分在遮罩边界上逐点对齐，不留缝、不叠面。

## Download · 下载

Get the installer from [Releases](../../releases):

- `LocalQuadRetopo_1.4.3_CN.zip` — 中文版
- `LocalQuadRetopo_1.4.3_EN.zip` — English

Unzip the whole package and double-click `INSTALL.bat`. If no suitable Python is found, the installer downloads a private one (about 60 MB, first install only).

解压整个压缩包，双击 `INSTALL.bat`。电脑上没有可用的 Python 时，安装程序会自动下载一个插件专用的（约 60 MB，只在第一次安装时）。

## Requirements · 运行环境

- Windows 10 / 11, 64-bit
- ZBrush 2022 (2022.0.5+) or ZBrush 2025
- Internet connection for the first install · 第一次安装需要联网

## Features · 功能

| | Regular · 规则布线 | Honeycomb · 蜂巢布线 |
|---|---|---|
| Look · 外观 | ZRemesher grid + transition ring · 规则网格加一圈过渡面 | Even-size quads · 大小均匀的四边面 |
| Speed · 速度 | ~5–25 s | ~2–15 s |
| Uses ZRemesher · 调用 ZRemesher | Yes · 是 | No · 否 |

- Target face count `0` = automatic, follows the density around the mask · 目标面数为 0 时按遮罩周围密度自动决定
- **Recommend Faces** button · 「推荐面数」按钮
- **Back to Original** keeps the source SubTool · 「回到原模型」随时切回
- A failed run rolls back automatically · 失败时自动回滚，不会破坏模型

Works well with Sculptris Pro: sculpt freely, mask the triangulated area, press one button to get quads back.

适合配合 Sculptris Pro：放开雕，雕完一处，画遮罩，按一下回到四边面。

## Documentation · 文档

- 中文：[使用说明](plugin/cn/使用说明.txt) · [用户指南](plugin/cn/用户指南.txt)
- English: [README](plugin/en/README.txt) · [User Guide](plugin/en/User%20Guide.txt)

## Limitations · 限制

- Models with UVs, materials, explicit normals or Polypaint are refused · 带 UV、材质、显式法线或顶点色的模型暂不支持
- Keep each mask moderate in size; very large areas take longer · 一次遮罩不宜过大
- The regular layout calls ZRemesher, which can crash ZBrush on a few extreme meshes — save first · 规则布线前建议先保存

## Repository layout · 目录

```
plugin/cn/   Chinese edition (installer, panel, Python backend, docs)
plugin/en/   English edition
```

`LocalQuadRetopo.txt` is the ZScript panel; `LocalQuadRetopo_2022.zsc` / `_2025.zsc` are the panels compiled by each ZBrush version. The `.py` files are the retopology backend (NumPy + SciPy).
