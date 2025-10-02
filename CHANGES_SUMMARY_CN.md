# 修改摘要 / Summary of Changes

## 問題陳述 / Problem Statement
將模型的輸入尺寸改為 640x640，並將原本的任務從人臉檢測改為 COCO 物件檢測任務。

Change the model input size to 640x640 and convert the task from face detection to COCO object detection.

## 主要變更 / Main Changes

### 1. 輸入尺寸 / Input Size
- **原始 / Original**: 4:3 長寬比 (例如 320x240, 640x480)
- **修改後 / Modified**: 640x640 方形輸入
- **影響文件 / Affected files**: 
  - `vision/ssd/config/fd_config.py`
  - `paddle/vision/ssd/config/fd_config.py`

### 2. 檢測任務 / Detection Task
- **原始 / Original**: 人臉檢測 (2 類別: BACKGROUND, face)
- **修改後 / Modified**: COCO 物件檢測 (81 類別: BACKGROUND + 80 COCO 類別)
- **新增文件 / New files**:
  - `models/coco-model-labels.txt` - COCO 類別標籤
  - `vision/datasets/coco_dataset.py` - COCO 資料集載入器

### 3. 支援的類別 / Supported Classes
COCO 資料集包含 80 個類別，例如：
- 人物: person
- 交通工具: bicycle, car, motorcycle, airplane, bus, train, truck, boat
- 動物: bird, cat, dog, horse, sheep, cow, elephant, bear, zebra, giraffe
- 家居用品: chair, couch, bed, dining table, toilet, tv, laptop
- 等等...

The COCO dataset includes 80 classes such as person, vehicles, animals, household items, etc.

## 修改的文件列表 / Modified Files List

### 配置文件 / Configuration Files
1. `vision/ssd/config/fd_config.py`
   - 修改 640 配置為 640x640 方形
   - 調整特徵圖尺寸為方形

2. `paddle/vision/ssd/config/fd_config.py`
   - 與 PyTorch 版本保持一致

### 訓練相關 / Training Related
3. `train.py`
   - 新增 COCO 資料集類型支援
   - 支援 `--dataset_type coco` 參數

4. `vision/datasets/coco_dataset.py` (新文件 / New)
   - 完整的 COCO 資料集載入器
   - 支援標準 COCO JSON 格式

### 推論腳本 / Inference Scripts
5. `detect_imgs.py`
   - 使用 COCO 標籤
   - 顯示類別名稱和信心度

6. `detect_imgs_onnx.py`
   - ONNX 版本的檢測腳本
   - 640x640 輸入

7. `run_video_face_detect.py`
   - 視頻檢測支援 COCO
   - 顯示物件類別

8. `run_video_face_detect_onnx.py`
   - ONNX 版本視頻檢測

9. `convert_to_onnx.py`
   - 轉換為 640x640 ONNX 模型
   - 支援 81 類別

### 標籤文件 / Label Files
10. `models/coco-model-labels.txt` (新文件 / New)
    - BACKGROUND + 80 COCO 類別

### 文檔 / Documentation
11. `COCO_USAGE.md` (新文件 / New)
    - 完整使用指南
    - 訓練和推論範例

12. `test_coco_config.py` (新文件 / New)
    - 配置驗證腳本
    - 快速測試安裝是否正確

## 使用方法 / Usage

### 訓練 / Training
```bash
python train.py \
    --dataset_type coco \
    --datasets /path/to/coco/train2017 \
    --validation_dataset /path/to/coco/val2017 \
    --input_size 640 \
    --net RFB \
    --batch_size 16 \
    --num_epochs 200
```

### 檢測圖片 / Detect Images
```bash
python detect_imgs.py \
    --net_type RFB \
    --input_size 640 \
    --path ./test_images \
    --threshold 0.5
```

### 驗證配置 / Verify Configuration
```bash
python test_coco_config.py
```

## 注意事項 / Notes

1. **模型重訓練 / Model Retraining**: 
   - 需要在 COCO 資料集上重新訓練模型
   - 預訓練的人臉檢測權重不相容於 81 類別

2. **記憶體需求 / Memory Requirements**:
   - 640x640 需要更多 GPU 記憶體
   - 建議調整批次大小

3. **向後相容性 / Backward Compatibility**:
   - 保留了原始 VOC 資料集支援
   - 可以繼續使用 `--dataset_type voc` 進行人臉檢測

4. **資料集格式 / Dataset Format**:
   - COCO 資料集需要標準格式
   - 參見 `COCO_USAGE.md` 了解詳細結構

## 測試結果 / Test Results

所有配置已通過測試：
- ✓ 640x640 方形輸入
- ✓ 81 類別 (BACKGROUND + 80 COCO)
- ✓ 23,500 個錨點先驗
- ✓ RFB 和 Slim 網路都支援
- ✓ 前向傳播正常運作

All configurations tested successfully:
- ✓ 640x640 square input
- ✓ 81 classes (BACKGROUND + 80 COCO)
- ✓ 23,500 anchor priors
- ✓ Both RFB and Slim networks supported
- ✓ Forward pass works correctly

## 參考資料 / References

- COCO 資料集: https://cocodataset.org/
- 詳細使用說明: 參見 `COCO_USAGE.md`
- 配置測試: 執行 `python test_coco_config.py`
