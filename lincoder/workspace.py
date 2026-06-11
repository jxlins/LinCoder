"""
工作区快照工具。

在 agent 真正调用模型之前，先从当前代码仓库里提取一小份稳定的项目信息，整理成文本，放进 prompt prefix 里，让模型知道自己正在什么仓库里工作。
"""

import subprocess
import textwrap
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


MAX_TOOL_OUTPUT = 4000  # 工具输出默认最多保留 4000 字符。
MAX_HISTORY = 12000     # 历史上下文最多保留 12000 字符。

# 这些文件最可能直接影响 agent 的行动方式。
# 我们不会预加载整个仓库，只会先给模型一小份“导航包”。
"""
定义了 agent 启动时优先读取的项目文件
    AGENTS.md：通常包含 agent 工作规则
    README.md：项目说明
    pyproject.toml：Python 项目配置
    package.json：前端或 Node 项目配置
"""
DOC_NAMES = ("AGENTS.md", "README.md", "pyproject.toml", "package.json")

"""
忽略目录集合
    .git：Git 内部数据
    .lincoder：agent 自己的运行数据
    __pycache__：Python 缓存
    .pytest_cache：测试缓存
    .ruff_cache：代码检查缓存
    .venv 和 venv：虚拟环境
"""
IGNORED_PATH_NAMES = {".git", ".lincoder", "__pycache__", ".pytest_cache", ".ruff_cache", ".venv", "venv"}

def now():
    """
    返回当前 UTC 时间，并转成 ISO 格式字符串。
    2026-06-11T15:30:12.123456+00:00
    """
    return datetime.now(timezone.utc).isoformat()

def clip(text, limit=MAX_TOOL_OUTPUT):
    """裁剪长文本"""
    text = str(text)
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n...[truncated {len(text) - limit} chars]"

def middle(text, limit):
    """保留文本两头的内容，中间用 ... 连接"""
    text = str(text).replace("\n", " ")
    if len(text) <= limit:
        return text
    if limit <= 3:
        return text[:limit]
    left = (limit - 3) // 2
    right = limit - 3 - left
    return text[:left] + "..." + text[-right:]

class WorkspaceContext:
    """
    这个类用于保存当前工作区的基础状态。

        cwd：当前运行目录
        repo_root：Git 仓库根目录
        branch：当前分支
        default_branch：默认分支
        status：Git 工作区状态
        recent_commits：最近提交记录
        project_docs：关键项目文档内容
    """
    def __init__(self, cwd, repo_root, branch, default_branch, status, recent_commits, project_docs):
        self.cwd = cwd
        self.repo_root = repo_root
        self.branch = branch
        self.default_branch = default_branch
        self.status = status
        self.recent_commits = recent_commits
        self.project_docs = project_docs

    @classmethod
    def build(cls, cwd, repo_root_override=None):
        """
        从真实文件系统和 Git 仓库中采集信息，并返回一个 WorkspaceContext 对象。
        """

        # 把传入的当前目录转换成绝对路径，并解析符号链接。
        cwd = Path(cwd).resolve()

        def git(args, fallback=""):
            """
            执行 Git 命令
                在 cwd 目录下执行 Git 命令
                捕获标准输出
                输出按文本处理
                命令失败会抛异常
                最多运行 5 秒
            """
            try:
                result = subprocess.run(
                    ["git", *args],
                    cwd=cwd,
                    capture_output=True,
                    text=True,
                    check=True,
                    timeout=5
                )
                return result.stdout.strip() or fallback
            except Exception:
                return fallback

        # 获取仓库根目录
        # git rev-parse --show-toplevel：这个 Git 命令用于获取当前仓库根目录。
        repo_root = (
            Path(repo_root_override).resolve()
            if repo_root_override is not None
            else Path(git(["rev-parse", "--show-toplevel"], str(cwd))).resolve()
        )

        docs = {}
        # 同时扫描 repo_root 和 cwd，这样在子目录启动时也能看到本地文档；
        # 但用相对路径做 key，避免同一份文档被重复收集。
        for base in (repo_root, cwd):
            for name in DOC_NAMES:
                path = base / name
                # 文件不存在就跳过。
                if not path.exists():
                    continue

                # 用相对路径作为 key，避免同一个文件被重复收集
                key = str(path.relative_to(repo_root))
                if key in docs:
                    continue

                # 读取文档内容，并裁剪到 1200 字符
                docs[key] = clip(path.read_text(encoding="utf-8", errors="replace"), 1200)

        return cls(
            cwd=str(cwd),
            repo_root=str(repo_root),
            branch=git(["branch", "--show-current"], "-") or "-",
            default_branch=(
                lambda branch: branch[len("origin/"):] if branch.startswith("origin/") else branch
            )(git(["symbolic-ref", "--short", "refs/remotes/origin/HEAD"], "origin/main") or "origin/main"),
            status=clip(git(["status", "--short"], "clean") or "clean", 1500),
            recent_commits=[line for line in git(["log", "--oneline", "-5"]).splitlines() if line],
            project_docs=docs,
        )

    def text(self):
        """渲染成 prompt 文本"""
        # 这段文本会被塞进 prompt prefix，作为相对稳定的基线上下文。
        # 1. 如果有最近提交，就逐行展示
        commits = "\n".join(f"- {line}" for line in self.recent_commits) or "- none"
        # 2. 展示项目文档列表，每个文档前面是相对路径，下面是内容摘要
        docs = "\n".join(f"- {path}\n{snippet}" for path, snippet in self.project_docs.items()) or "- none"

        return textwrap.dedent(
            f"""\
            Workspace:
            - cwd: {self.cwd}
            - repo_root: {self.repo_root}
            - branch: {self.branch}
            - default_branch: {self.default_branch}
            - status:
            {self.status}
            - recent_commits:
            {commits}
            - project_docs:
            {docs}
            """
        ).strip()

    def fingerprint(self):
        """生成当前工作区上下文的哈希。"""
        # 这个指纹用来判断仓库状态是否发生了足够大的变化，
        # 从而决定是否需要重建缓存中的 prompt prefix。
        payload = {
            "cwd": self.cwd,
            "repo_root": self.repo_root,
            "branch": self.branch,
            "default_branch": self.default_branch,
            "status": self.status,
            "recent_commits": list(self.recent_commits),
            "project_docs": dict(self.project_docs),
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()