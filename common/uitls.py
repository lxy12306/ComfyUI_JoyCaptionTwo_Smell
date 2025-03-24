import gc
import json
import os
import inspect
from huggingface_hub import snapshot_download

from comfy.model_management import unload_all_models, soft_empty_cache

def load_hg_model(
    model_id: str,
    model_dir: str,
    exDir: str = '',
    resume_download: bool = True,
    proxies: dict = None
) -> str:
    """
    检查指定本地目录下是否存在模型，
    不存在则从Hugging Face下载，
    如下载失败则抛出异常。

    :param model_id: Hugging Face模型仓库名或路径 (e.g., 'organization/model_name')
    :param model_dir: 本地模型根目录
    :param exDir: 额外子目录
    :param resume_download: 是否启用断点续传
    :param proxies: 代理设置，例如 {'http': 'http://...', 'https': 'http://...'}
    :return: 下载或加载的模型本地路径
    """
    model_checkpoint = os.path.join(model_dir, exDir, os.path.basename(model_id))

    if not os.path.exists(model_checkpoint):
        print(f"本地未找到模型 '{model_checkpoint}'，尝试从 Hugging Face 下载...")
        try:
            snapshot_download(
                repo_id=model_id,
                local_dir=model_checkpoint,
                resume_download=resume_download,
                proxies=proxies,
                # 如果有需要，可指定分支、标签或提交ID
                # revision="main",
            )
            print(f"模型已成功下载到 '{model_checkpoint}'")
        except Exception as e:
            raise RuntimeError(f"从 Hugging Face 下载模型失败，错误信息: {e}")
    else:
        print(f"已检测到本地模型目录: {model_checkpoint}")

    return model_checkpoint
def clear_cache():
    gc.collect()
    unload_all_models()
    soft_empty_cache()

def modify_json_value(file_path, key_to_modify, new_value):
  """
  读取 JSON 文件，修改指定 key 的 value 值，并保存修改后的文件。

  Args:
    file_path: JSON 文件路径。
    key_to_modify: 需要修改的 key。
    new_value:  新的 value 值。
  """
  try:
    with open(file_path, 'r', encoding='utf-8') as f:
      data = json.load(f)

    # 查找并修改 key 的 value
    if key_to_modify in data:
      data[key_to_modify] = new_value
    else:
      print(f"Warning: Key '{key_to_modify}' not found in JSON file.")

    # 保存修改后的 JSON 文件
    with open(file_path, 'w', encoding='utf-8') as f:
      json.dump(data, f, indent=4)  # 使用 indent 参数格式化输出

    print(f"Successfully modified '{key_to_modify}' value in '{file_path}'.")

  except FileNotFoundError:
    print(f"Error: File '{file_path}' not found.")
  except json.JSONDecodeError:
    print(f"Error: Invalid JSON format in '{file_path}'.")

def read_json_file(file_path):
  """读取 JSON 文件并转换为 Python 字典。"""
  try:
    with open(file_path, 'r', encoding='utf-8') as f:
      data = json.load(f)
    return data
  except FileNotFoundError:
    print(f"Error: File '{file_path}' not found.")
    return None
  except json.JSONDecodeError:
    print(f"Error: Invalid JSON format in '{file_path}'.")
    return None