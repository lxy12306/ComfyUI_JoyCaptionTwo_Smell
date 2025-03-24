import transformers
from transformers import AutoModel
from packaging import version

from .uitls import *

class ClipModel:
    def __init__(self, base_model_path: str, dtype):
        """
        :param base_model_path: 本地模型根目录
        :param dtype: 指定模型加载的 torch dtype, 例如 torch.float16
        """
        self.base_model_path = base_model_path
        self.dtype = dtype
        self.clip_model = None  # 供后续使用

    def load_clip_model(self, use_new_model=True, debug=False):
        """
        检查 Transformers 版本并决定加载哪个 SigLIP 模型；
        如果本地无对应版本则从 Hugging Face 下载并加载该模型。
        """
        current_version = transformers.__version__
        if version.parse(current_version) > version.parse("4.49.0.dev0"):
            model_id = "google/siglip-so400m-patch14-384"
            if use_new_model:
                model_id = "google/siglip2-so400m-patch14-384"
            print(f"transformers 版本较新: {current_version}，使用模型: {model_id}")
        else:
            model_id = "google/siglip-so400m-patch14-384"
            print(f"transformers 版本较旧或相同: {current_version}，使用模型: {model_id}")

        # 加载/下载模型
        clip_path = load_hg_model(model_id, self.base_model_path, "clip")

        # 使用 AutoModel 加载，取出 vision_model
        full_model = AutoModel.from_pretrained(
            clip_path,
            trust_remote_code=True,
            torch_dtype=self.dtype
        )
        if debug:
            # 打印信息
            print("clip_model 类型:", type(full_model))
            methods = [name for name, func in inspect.getmembers(full_model, inspect.ismethod)]
            print("clip_model 中可用的方法:")
            for m in methods:
                print("-", m)
            self.clip_model = full_model.vision_model

            # 打印信息
            print("clip_model 类型:", type(self.clip_model))
            methods = [name for name, func in inspect.getmembers(self.clip_model, inspect.ismethod)]
            print("clip_model 中可用的方法:")
            for m in methods:
                print("-", m)

        # 如需后续使用，可以返回模型实例
        return self.clip_model