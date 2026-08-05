import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0" 

import torch

print("torch version:", torch.__version__)
print("torch compiled with CUDA:", torch.version.cuda)
print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))
print("CUDA_LAUNCH_BLOCKING:", os.environ.get("CUDA_LAUNCH_BLOCKING"))

print("torch.cuda.is_available():", torch.cuda.is_available())
print("torch.cuda.device_count():", torch.cuda.device_count())
