# 模型说明

| 文件 | 用途 | 来源 | 许可证 | SHA256 |
|---|---|---|---|---|
| `nanodet.onnx` | COCO 人体检测（person 类） | [OpenCV Zoo / object_detection_nanodet](https://github.com/opencv/opencv_zoo/tree/main/models/object_detection_nanodet) | Apache-2.0 | `4b82da9944b88577175ee23a459dce2e26e6e4be573def65b1055dc2d9720186` |
| `yunet.onnx` | 轻量人脸检测 | [OpenCV Zoo / face_detection_yunet](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet) | MIT | `8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4` |

两个模型都只有几 MB，CPU 推理即可，不需要 GPU。

## 重新下载

```bash
curl -L -o models/nanodet.onnx \
  https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/object_detection_nanodet/object_detection_nanodet_2022nov.onnx
curl -L -o models/yunet.onnx \
  https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx
sha256sum models/*.onnx
```

> 注意：`raw.githubusercontent.com` 对这两个文件返回的是 Git LFS 指针，
> 必须用 `media.githubusercontent.com` 才能拿到真正的 ONNX 文件。
