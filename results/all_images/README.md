# LuTan-1 最终 35 张结果图

本目录是可直接在 GitHub 浏览和下载的最终图片包。完整复现方法见
[`../../REPRODUCE_ALL_IMAGES.md`](../../REPRODUCE_ALL_IMAGES.md)，机器可读清单见
[`MANIFEST.json`](MANIFEST.json)。

- `main__`：8 张小批量投影、掩膜、地理编码、误差和三维散射点图；
- `full_area__`：11 张全区域结果与诊断图；
- `coordinates__`：6 张 Geo-BC/LuTan 严格同像素坐标图；
- `deformation__`：6 张形变速率和累计形变图；
- `3d__`：4 张全区域、范围定位、热点和逐建筑插值三维图。

文件名前缀只用于避免同名，不改变原图内容。运行以下命令可重新汇总并验证 35 张图与源图
逐字节一致：

```bash
cd a_geo_lutan
source env.sh
"$SAR_GEOCODE_PYTHON" code/collect_all_images.py
"$SAR_GEOCODE_PYTHON" code/collect_all_images.py --check
```
