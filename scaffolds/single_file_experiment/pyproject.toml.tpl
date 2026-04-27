[project]
name = "{{EXPERIMENT_NAME}}"
version = "0.1.0"
description = "{{DESCRIPTION}}"
requires-python = ">=3.11"
dependencies = [
    "mindlab-toolkit @ git+https://github.com/MindLab-Research/mindlab-toolkit.git",
    "tinker==0.15.0",
    "transformers>=4.50",
]

[tool.uv]
package = false
index-url = "https://pypi.tuna.tsinghua.edu.cn/simple"
index-strategy = "first-index"
python-downloads = "never"
