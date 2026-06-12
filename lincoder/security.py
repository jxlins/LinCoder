"""
Security and redaction helpers for runtime artifacts.

运行时安全与脱敏工具模块。它主要负责三件事：
    第一，识别哪些环境变量可能是敏感信息。
    第二，在 trace、report、metadata、工具结果等运行产物中把密钥值替换成 <redacted>。
    第三，为 shell 工具构造一个经过白名单过滤的环境变量字典。
"""

import os

# 敏感环境变量标记
SENSITIVE_ENV_NAME_MARKERS = ("API_KEY", "TOKEN", "SECRET", "PASSWORD")
REDACTED_VALUE = "<redacted>"   # 脱敏后的替代文本


def _normalized_secret_names(secret_env_names):
    """把外部传入的敏感环境变量名统一转成大写集合。"""
    return {str(name).upper() for name in (secret_env_names or ())}


def looks_sensitive_env_name(name):
    """根据变量名本身判断它是否像敏感变量。"""
    upper = str(name).upper()
    return any(upper == marker or upper.endswith(marker) or upper.endswith(f"_{marker}") for marker in SENSITIVE_ENV_NAME_MARKERS)


def is_secret_env_name(name, secret_env_names=None):
    """
    综合判断是否是敏感环境变量
    综合两种依据：
        用户显式配置的密钥变量名
        变量名本身看起来像密钥
    """
    upper = str(name).upper()
    return upper in _normalized_secret_names(secret_env_names) or looks_sensitive_env_name(upper)


def configured_secret_env_items(env=None, secret_env_names=None):
    """只查找 用户显式配置 的敏感环境变量。"""
    env = os.environ if env is None else env
    configured_names = _normalized_secret_names(secret_env_names)
    items = [
        (name, value)
        for name, value in env.items()
        if str(name).upper() in configured_names and value
    ]
    items.sort(key=lambda item: item[0])
    return items


def detected_secret_env_items(env=None, secret_env_names=None):
    """自动检测所有疑似敏感环境变量"""
    env = os.environ if env is None else env
    items = [
        (name, value)
        for name, value in env.items()
        if is_secret_env_name(name, secret_env_names=secret_env_names) and value
    ]
    items.sort(key=lambda item: item[0])
    return items


def secret_env_summary(env=None, secret_env_names=None):
    """统计显式配置的敏感环境变量"""
    names = [name for name, _ in configured_secret_env_items(env=env, secret_env_names=secret_env_names)]
    return {
        "secret_env_count": len(names),
        "secret_env_names": names,
    }


def detected_secret_env_summary(env=None, secret_env_names=None):
    """统计自动检测到的敏感环境变量"""
    names = [name for name, _ in detected_secret_env_items(env=env, secret_env_names=secret_env_names)]
    return {
        "secret_env_count": len(names),
        "secret_env_names": names,
    }


def redact_text(text, env=None, secret_env_names=None):
    """
    对文本中的密钥值脱敏
    遍历所有检测到的敏感环境变量值，并把文本中出现的这些值替换成：<redacted>
    """
    text = str(text)
    for _, value in sorted(
        detected_secret_env_items(env=env, secret_env_names=secret_env_names),
        key=lambda item: len(item[1]),
        reverse=True,
    ):
        text = text.replace(value, REDACTED_VALUE)
    return text


def redact_artifact(value, key=None, env=None, secret_env_names=None):
    """脱敏任意结构化数据。"""
    if key and is_secret_env_name(key, secret_env_names=secret_env_names):
        return REDACTED_VALUE

    if isinstance(value, dict):
        return {
            str(item_key): redact_artifact(item_value, key=item_key, env=env, secret_env_names=secret_env_names)
            for item_key, item_value in value.items()
        }

    if isinstance(value, list):
        return [redact_artifact(item, key=key, env=env, secret_env_names=secret_env_names) for item in value]

    if isinstance(value, tuple):
        return [redact_artifact(item, key=key, env=env, secret_env_names=secret_env_names) for item in value]

    if isinstance(value, str):
        return redact_text(value, env=env, secret_env_names=secret_env_names)
    return value


def shell_env(env=None, allowlist=(), root="."):
    """构造运行 shell 命令时传入的环境变量。"""

    # 1. 默认使用系统环境变量
    env = os.environ if env is None else env

    # 2. 只保留白名单变量
    filtered = {
        name: env[name]
        for name in allowlist
        if name in env
    }

    # 3. 强制设置 PWD
    filtered["PWD"] = str(root)

    # 4. 保底加入 PATH
    if "PATH" not in filtered and env.get("PATH"):
        filtered["PATH"] = env["PATH"]

    return filtered