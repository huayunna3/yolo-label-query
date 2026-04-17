#!/usr/bin/env python3
"""
GitHub仓库创建脚本
需要GitHub Personal Access Token
"""

import json
import requests
import sys

def create_github_repo(token, repo_name, description="", private=False):
    """
    使用GitHub API创建仓库

    Args:
        token (str): GitHub Personal Access Token
        repo_name (str): 仓库名称
        description (str): 仓库描述
        private (bool): 是否为私有仓库
    """
    url = "https://api.github.com/user/repos"
    headers = {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github.v3+json"
    }
    data = {
        "name": repo_name,
        "description": description,
        "private": private,
        "auto_init": False  # 不自动初始化文件
    }

    try:
        response = requests.post(url, headers=headers, json=data)

        if response.status_code == 201:
            repo_info = response.json()
            print("✅ 仓库创建成功！")
            print(f"仓库名称: {repo_info['name']}")
            print(f"仓库URL: {repo_info['html_url']}")
            print(f"SSH URL: {repo_info['ssh_url']}")
            print(f"HTTPS URL: {repo_info['clone_url']}")
            return repo_info
        else:
            print(f"❌ 创建仓库失败: {response.status_code}")
            print(f"错误信息: {response.text}")
            return None

    except Exception as e:
        print(f"❌ 请求失败: {e}")
        return None

def add_readme_file():
    """创建README.md文件"""
    readme_content = """# YOLO Label Query Tool

一个用于查询和分析YOLO数据集标签的Python工具。

## 功能特点

- 将YOLO标注数据导入pandas DataFrame
- 提供丰富的pandas查询指令示例
- 支持JSON格式的标注数据处理
- 简单的交互式界面

## 安装依赖

```bash
pip install pandas
```

## 使用方法

1. 运行脚本：
```bash
python 标签数查询.py
```

2. 脚本会自动加载示例数据到pandas DataFrame

3. 使用pandas查询指令进行分析，例如：
```python
# 查看数据
df.head()

# 统计各类标签数量
df['transcription'].value_counts()

# 筛选包含特定字符的标注
df[df['transcription'].str.contains('⌀')]

# 查看困难样本
df[df['difficult'] == True]
```

## 数据格式

脚本支持以下格式的YOLO标注数据：
```json
[
  {
    "transcription": "文本内容",
    "points": [[x1,y1], [x2,y2], [x3,y3], [x4,y4]],
    "difficult": false
  }
]
```

## 示例查询

脚本中包含了详细的pandas查询指令注释，包括：
- 基本统计查询
- 条件筛选
- 字符串操作
- 分组统计
- 排序和过滤

## 许可证

MIT License
"""

    with open("README.md", "w", encoding="utf-8") as f:
        f.write(readme_content)
    print("📄 README.md 文件已创建")

def add_gitignore():
    """创建.gitignore文件"""
    gitignore_content = """# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Virtual Environment
venv/
env/
.venv/

# IDE
.vscode/
.idea/
*.swp
*.swo
*~

# OS
.DS_Store
.DS_Store?
._*
.Spotlight-V100
.Trashes
ehthumbs.db
Thumbs.db

# Data files
*.csv
*.xlsx
*.pkl
*.pickle

# Logs
*.log
"""

    with open(".gitignore", "w", encoding="utf-8") as f:
        f.write(gitignore_content)
    print("📁 .gitignore 文件已创建")

def main():
    print("=" * 50)
    print("GitHub仓库创建工具")
    print("=" * 50)

    # 获取token
    token = input("请输入GitHub Personal Access Token: ").strip()
    if not token:
        print("❌ 需要提供token")
        return

    # 仓库信息
    repo_name = "yolo-label-query"
    description = "Python脚本用于查询YOLO数据集标签数量"
    private = False  # 设为True创建私有仓库

    print(f"\n准备创建仓库: {repo_name}")
    print(f"描述: {description}")
    print(f"类型: {'私有' if private else '公开'}")

    confirm = input("\n确认创建？(y/n): ").strip().lower()
    if confirm != 'y':
        print("取消创建")
        return

    # 创建仓库
    repo_info = create_github_repo(token, repo_name, description, private)

    if repo_info:
        # 添加项目文件
        add_readme_file()
        add_gitignore()

        print("\n✅ 项目文件已准备就绪")
        print(f"📦 接下来可以推送代码到: {repo_info['ssh_url']}")

        # 显示推送命令
        print("\n推送命令:")
        print(f"git remote add origin {repo_info['ssh_url']}")
        print("git add README.md .gitignore")
        print('git commit -m "添加项目文档"')
        print("git push -u origin main")

if __name__ == "__main__":
    main()