"""
Project-local configuration helpers.

轻量级且安全的项目级环境变量加载器。
在 AI Agent 或任何复杂应用中，它通常用于读取项目根目录下的 .env 文件，从而安全地注入 API Keys、数据库连接字符串等敏感配置。
"""

import os
import re
from pathlib import Path


ENV_KEY_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _strip_quotes(value):
    """去除值两端的引号（如果存在）"""
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value


def _parse_env_line(line):
    """.env 文件解析"""

    line = line.strip()

    # 自动跳过空行和以 # 开头的注释行。
    if not line or line.startswith("#"):
        return None

    # 支持并自动剥离行首的 export 关键字（例如 export KEY=VALUE）
    if line.startswith("export "):
        line = line[len("export "):].strip()

    if "=" not in line:
        raise ValueError(f"invalid .env line: {line}")

    name, value = line.split("=", 1)
    name = name.strip()
    # 使用正则表达式 ENV_KEY_PATTERN 确保变量名符合标准（仅允许字母、数字和下划线，且不能以数字开头），防止恶意或格式错误的键名注入。
    if not ENV_KEY_PATTERN.match(name):
        raise ValueError(f"invalid .env variable name: {name}")

    return name, _strip_quotes(value)


def find_project_env(start):
    """
    从指定的起始路径（文件或目录）开始，逐级向上遍历父目录，直到找到第一个 .env 文件为止。
    """
    current = Path(start).resolve()
    if current.is_file():
        current = current.parent
    for path in (current, *current.parents):
        env_path = path / ".env"
        if env_path.exists():
            return env_path
    return None


def load_project_env(start, override=True):
    """
    从项目根目录的 .env 文件加载环境变量，并注入到 os.environ 中。
    """
    env_path = find_project_env(start)
    if env_path is None:
        return {}
    loaded = {}

    for line in env_path.read_text(encoding="utf-8").splitlines():
        parsed = _parse_env_line(line)
        if parsed is None:
            continue
        name, value = parsed
        loaded[name] = value
        if override or name not in os.environ:
            os.environ[name] = value
    return loaded


def provider_env(name, legacy_names=(), default=""):
    for env_name in (name, *legacy_names):
        value = os.environ.get(env_name)
        if value:
            return value
    return default