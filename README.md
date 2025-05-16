# 项目环境说明

## 安装 GPU 版 PyTorch

本项目已适配 NVIDIA GPU（如 4060），并支持 CUDA 12.0。请在 uv 虚拟环境下运行以下命令安装 GPU 版 PyTorch：

```bash
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

安装完成后，可用如下代码测试 GPU 是否可用：

```python
import torch
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))
```

如输出为 `True` 和你的显卡型号，则说明 GPU 支持正常。

```bash
uv pip list | grep torch
```



