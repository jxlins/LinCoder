"""Narrow context passed from runtime into tool functions."""

from dataclasses import dataclass
from pathlib import Path
from typing import Callable


@dataclass
class ToolContext:
    """
    Agent 框架中工具执行环境（Tool Execution Environment）的核心抽象。
    主要作用是作为一个安全且受控的上下文容器，将运行时的必要依赖和约束传递给具体的工具函数，而不是让工具直接访问全局状态或操作系统。
    """
    root: Path                              # 定义了当前 Agent 的工作根目录
    path_resolver: Callable[[str], Path]    # 提供一个路径解析函数，工具函数可以通过它来访问文件系统，但只能访问被允许的路径
    shell_env_provider: Callable[[], dict]  # 提供一个函数，返回一个受限的环境变量字典，工具函数在执行 shell 命令时只能使用这些环境变量
    depth: int                              # 当前工具调用的深度，通常用于限制递归调用或跟踪调用层级
    max_depth: int                          # 定义了工具调用的最大深度，超过这个深度的调用将被拒绝，以防止无限递归或过深的调用链
    spawn_delegate: Callable[[dict], str]   # 提供一个函数，允许工具函数在需要时生成新的工具调用请求，这些请求会被发送回 Agent 的主循环进行调度和执行

    def path(self, raw_path):
        return self.path_resolver(str(raw_path))

    def shell_env(self):
        return self.shell_env_provider()