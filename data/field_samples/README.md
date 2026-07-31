# 现场样张目录（field_samples）

放**现场实拍**的车顶胶条照片，供离线调试检测算法用。

## 为什么单独建这个目录

| 目录 | 用途 | 会不会被清空 |
|---|---|---|
| `data/raw_images/` | 相机连拍的**临时缓冲**，程序运行时会写入/覆盖 | ⚠️ 会 |
| `data/inspection/` | 正式检测记录，按 日期/PIN 分层，程序自动写 | ⚠️ 程序管理 |
| `mock/sample_images/` | 脚本生成的**仿真图**，不是真图 | 否 |
| **`data/field_samples/`** | **人工放置的现场真图，只读，稳定不动** | **否** |

## 怎么放

直接把照片拷进这个目录即可，建议按批次分子目录：

```
data/field_samples/
├── 20260731_产线A/
│   ├── ok_01.jpg          # 正常胶条
│   ├── ok_02.jpg
│   ├── ng_missing_01.jpg  # 缺失
│   └── ng_break_01.jpg    # 断胶
└── 20260801_产线B/
    └── ...
```

**命名建议**（不强制，但能大幅加快调参）：
- 正常样张前缀 `ok_`
- 缺陷样张前缀 `ng_<缺陷类型>_`，类型取值：`missing`（缺失）/ `break`（断胶）/ `overspray`（过喷）/ `width`（宽度超差）

这样跑验证脚本时可以直接用文件名当"标准答案"对比检出结果。

## 怎么跑检测

```bash
# 验证整个目录
python tools/validate_detector.py --src data/field_samples/20260731_产线A

# 结果看这里
#   tools/results/          带标注的可视化图 + HTML 报告
#   data/process_data/      中间过程 mask 图（排查用）
```

## 图片要求

当前检测器参数是按**相机实拍**调的，注意：

- **分辨率**：原设计 1920×1200（Basler acA1920-48gm）。ROI = `(0, 405, 1920, 259)`，即从顶部往下 405px 起、高 259px 的横带。
  代码里有 `min()` clamp，更窄的图能自适应，但**ROI 位置是按相机安装角度定的**——手机随手拍的照片胶条不一定落在这个带里，可能要重新标 ROI。
- **色彩**：算法内部转灰度，彩色/黑白都能吃。
- **标定**：`config/calibration.json` 里的 `mm_per_pixel`（兜底 0.30466）决定"宽度超差"判定。手机照片的比例尺完全不同，**宽度类判定会失真**，先只看"有没有胶条 / 断没断"这类形态判定。

> 如果这批照片是手机/其他相机拍的，先跑一次 `validate_detector.py` 看结果，大概率要重新调 ROI 和 `OTSU_THRESHOLD_DELTA`。把结果发出来一起看。

## 斜胶条 / 取景不同的照片：逐图 ROI 覆盖

现场照片常因**取景不同**（每张胶条在画面里的 y 位置不一样）或**胶条不水平（随车身前进而上抬）**，单个全局横带罩不住。
检测器支持**逐张照片单独标 ROI**，且能自动估算：

```bash
# 1) 自动估算每张照片的 ROI，写入 {src}/roi_overrides.json，并直接用它跑检测
python tools/validate_detector.py --src data/field_samples/20260727 --suggest-roi

# 2) 之后直接重跑（会自动读取同目录下的 roi_overrides.json，无需再带参数）
python tools/validate_detector.py --src data/field_samples/20260727

# 3) 也可手工指定（对所有图生效）
python tools/validate_detector.py --src <目录> --roi x,y,w,h

# 4) 或指定一份逐图 JSON：{"照片名.jpg":[x,y,w,h], ...}
python tools/validate_detector.py --src <目录> --roi-json roi_overrides.json
```

`roi_overrides.json` 形如：

```json
{
  "DSC_0001.jpg": [0, 150, 1600, 300],
  "DSC_0002.jpg": [0, 220, 1600, 320]
}
```

要点：
- `--suggest-roi` 的估框逻辑是"整图找最暗的水平/斜向条带外接框"，对斜胶条会自动罩住整条斜度。
  若照片里另有明显暗物（阴影、焊缝、车顶边缘），估框可能偏大——看 `tools/results/` 里的叠加验证图核对，手工在 JSON 里收一下即可。
- 全局 `ROI` 常量不受影响，相机固定机位的场景仍用一条加高带即可。


## 版本管理

本目录**不入 git**（照片体积大，且属于现场数据）。见根目录 `.gitignore`。
需要共享给他人时走网盘。
