from huggingface_hub import HfApi, HfFolder, upload_file, create_repo

# 1. 登录 Hugging Face（第一次需要手动执行）
# 在终端执行：
# huggingface-cli login
# 然后输入你的 Access Token（在 https://huggingface.co/settings/tokens 里生成）

# 2. 设置你的仓库信息
repo_id = "zizi-0123/behavior_analysis_dataset"  # 例如： "jiahejin/afm-training-logs"
file_path = "/home/jjiahe/code/deepresearch_llm_modeling/train/sft/data/webwalkerqa_logs.zip"
path_in_repo = "data/webwalkerqa_logs.zip"   # 上传后在仓库中的路径
private_dataset = True # 设置为True以创建私有数据集

api = HfApi()
api.create_repo(repo_id=repo_id, repo_type="dataset", private=private_dataset, exist_ok=True)

# 3. 上传文件
upload_file(
    path_or_fileobj=file_path,
    path_in_repo=path_in_repo,
    repo_id=repo_id,
    repo_type="dataset"   # 如果是模型仓库则改为 "model"
)

print(f"✅ 已成功上传 {file_path} 到 {repo_id}/{path_in_repo}")