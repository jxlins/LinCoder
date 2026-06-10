"""
工具定义与执行辅助逻辑。

可以把这个文件看成 agent 的能力白名单：模型能申请哪些动作、这些动作
如何做参数校验，以及最终如何执行，都是在这里定义的。
"""

import shutil
import subprocess
import textwrap
from functools import partial

"""
工具结构
    schema 让模型知道参数长什么样
    risky 决定它要不要走审批
    description 决定这个动作怎么被提示给模型
"""
BASE_TOOL_SPECS = {
    # 列目录
    "list_files": {
        "schema": {"path": "str='.'"},
        "risky": False,
        "description": "List files in the workspace.",
    },
    # 读文件
    "read_file": {
        "schema": {"path": "str", "start": "int=1", "end": "int=200"},
        "risky": False,
        "description": "Read a UTF-8 file by line range.",
    },
    # 搜字符串
    "search": {
        "schema": {"pattern": "str", "path": "str='.'"},
        "risky": False,
        "description": "Search the workspace with rg or a simple fallback.",
    },
    # 跑 shell
    "run_shell": {
        "schema": {"command": "str", "timeout": "int=20"},
        "risky": True,
        "description": "Run a shell command in the repo root.",
    },
    # 写文件
    "write_file": {
        "schema": {"path": "str", "content": "str"},
        "risky": True,
        "description": "Write a text file.",
    },
    # 精确替换文本
    "patch_file": {
        "schema": {"path": "str", "old_text": "str", "new_text": "str"},
        "risky": True,
        "description": "Replace one exact text block in a file.",
    },
}

# 条件注册的 delegate。当前 agent 深度还没超过上限时，它才会出现在工具表面里。
DELEGATE_TOOL_SPEC = {
    "schema": {"task": "str", "max_steps": "int=3"},
    "risky": False,
    "description": "Ask a bounded read-only child agent to investigate.",
}

def legal_tool_names():
    """
    返回系统中所有合法的工具名称集合
    """
    return set(BASE_TOOL_SPECS) | {"delegate"}

TOOL_EXAMPLES = {
    "list_files": '<tool>{"name":"list_files","args":{"path":"."}}</tool>',
    "read_file": '<tool>{"name":"read_file","args":{"path":"README.md","start":1,"end":80}}</tool>',
    "search": '<tool>{"name":"search","args":{"pattern":"binary_search","path":"."}}</tool>',
    "run_shell": '<tool>{"name":"run_shell","args":{"command":"uv run --with pytest python -m pytest -q","timeout":20}}</tool>',
    "write_file": '<tool name="write_file" path="binary_search.py"><content>def binary_search(nums, target):\n    return -1\n</content></tool>',
    "patch_file": '<tool name="patch_file" path="binary_search.py"><old_text>return -1</old_text><new_text>return mid</new_text></tool>',
    "delegate": '<tool>{"name":"delegate","args":{"task":"inspect README.md","max_steps":3}}</tool>',
}

def build_tool_registry(context):
    """
    根据当前的运行上下文（Context），动态组装出一个安全、有边界且立即可用的工具注册表，供大语言模型（LLM）调用

    工具不是动态发现的，而是显式注册的。
    这样模型看到的是一个有边界、可审计的动作集合。
    """

    tools = {
        name: {**spec, "run": partial(_TOOL_RUNNERS[name], context)}
        for name, spec in BASE_TOOL_SPECS.items()
    }

    # delegate（委托子 Agent）工具不是永远存在的。
    # 只有当当前 Agent 的递归深度（context.depth）小于最大允许深度（context.max_depth）时，它才会被加入注册表。
    if context.depth < context.max_depth:
        tools["delegate"] = {**DELEGATE_TOOL_SPEC, "run": partial(tool_delegate, context)}
    return tools

def tool_example(name):
    return TOOL_EXAMPLES.get(name, "")

def validate_tool(context, name, args):
    """
    在工具真正执行之前，先进行严格的参数校验。
    """

    args = args or {}

    # list_files：目标必须是目录
    if name == "list_files":
        path = context.path(args.get("path", "."))
        if not path.is_dir():
            raise ValueError("path is not a directory")
        return

    # read_file：目标必须是文件，行范围必须合法
    if name == "read_file":
        path = context.path(args["path"])
        if not path.is_file():
            raise ValueError("path is not a file")
        start = int(args.get("start", 1))
        end = int(args.get("end", 200))
        if start < 1 or end < start:
            raise ValueError("invalid line range")
        return

    # search：搜索模式必须非空，搜索路径必须是目录或文件
    if name == "search":
        pattern = str(args.get("pattern", "")).strip()
        if not pattern:
            raise ValueError("pattern must not be empty")
        context.path(args.get("path", "."))
        return

    # run_shell：命令必须非空，超时时间必须在 1~120 秒范围内
    if name == "run_shell":
        command = str(args.get("command", "")).strip()
        if not command:
            raise ValueError("command must not be empty")
        timeout = int(args.get("timeout", 20))
        if timeout < 1 or timeout > 120:
            raise ValueError("timeout must be in [1, 120]")
        return

    # write_file：目标路径不能是目录，必须有 content 参数，即不能把目录当文件写
    if name == "write_file":
        path = context.path(args["path"])
        if path.exists() and path.is_dir():
            raise ValueError("path is a directory")
        if "content" not in args:
            raise ValueError("missing content")
        return

    # patch_file：目标必须是文件，old_text 不能为空且必须在文件中唯一出现，new_text 参数必须存在
    if name == "patch_file":
        # patch_file 故意做得很严格：old_text 必须精确命中且只能出现一次，
        # 这样修改行为才是确定的，失败原因也更容易解释。
        path = context.path(args["path"])
        if not path.is_file():
            raise ValueError("path is not a file")
        old_text = str(args.get("old_text", ""))
        if not old_text:
            raise ValueError("old_text must not be empty")
        if "new_text" not in args:
            raise ValueError("missing new_text")
        text = path.read_text(encoding="utf-8")
        count = text.count(old_text)
        if count != 1:
            raise ValueError(f"old_text must occur exactly once, found {count}")
        return

    # delegate：task 参数必须非空，且当前递归深度不能超过上限
    if name == "delegate":
        task = str(args.get("task", "")).strip()
        if not task:
            raise ValueError("task must not be empty")
        if context.depth >= context.max_depth:
            raise ValueError("delegate depth exceeded")
        return

def tool_list_files(context, args):
    """
    安全地读取指定目录，过滤掉无关的系统文件，按“先文件夹后文件”的直观顺序排列，并将结果格式化为一组带有类型标识的相对路径列表，返回给大语言模型（LLM）。
    """

    # 默认值处理：如果 LLM 没有传入 path 参数，默认使用 "."（当前工作目录）
    # 安全沙箱：通过 context.path() 解析路径，确保 LLM 无法越权访问工作区之外的敏感系统文件。
    path = context.path(args.get("path", "."))
    if not path.is_dir():
        raise ValueError("path is not a directory")

    # 利用元组排序机制，实现了文件夹永远排在文件前面，且同级条目按字母不区分大小写排序。这极大地提升了 LLM 阅读目录结构的效率。
    entries = [
        item for item in sorted(path.iterdir(), key=lambda item:(item.is_file(), item.name.lower()))
        if item.name not in IGNORED_PATH_NAMES
    ]
    lines = []
    for entry in entries[:200]:
        # 为每个条目加上 [D]（Directory）或 [F]（File）前缀，让 LLM 能够一眼分辨出哪些是可以进入的子目录，哪些是文件。
        kind = "[D]" if entry.is_dir() else "[F]"
        # 使用 relative_to(context.root) 去除绝对路径的前缀，让返回结果更加简洁。
        lines.append(f"{kind} {entry.relative_to(context.root)}")
    return "\n".join(lines) or "(empty)"

def tool_read_file(context, args):
    """
    读文件工具
    """

    path = context.path(args["path"])
    if not path.is_file():
        raise ValueError("path is not a file")

    start = int(args.get("start", 1))
    end = int(args.get("end", 200))
    if start < 1 or end < start:
        raise ValueError("invalid line range")

    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    body = "\n".join(f"{number:>4}: {line}" for number, line in enumerate(lines[start - 1:end], start=start))
    return f"# {path.relative_to(context.root)}\n{body}"


def tool_search(context, args):
    pattern = str(args.get("pattern", "")).strip()
    if not pattern:
        raise ValueError("pattern must not be empty")

    path = context.path(args.get("path", "."))

    if shutil.which("rg"):
        # 优先用 rg，因为搜索会非常频繁，搜索延迟会直接影响 agent 控制循环。
        result = subprocess.run(
            ["rg", "-n", "--smart-case", "--max-count", "200", pattern, str(path)],
            cwd=context.root,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip() or result.stderr.strip() or "(no matches)"

    matches = []
    files = [path] if path.is_file() else [
        item for item in path.rglob("*")
        if item.is_file() and not any(part in IGNORED_PATH_NAMES for part in item.relative_to(context.root).parts)
    ]
    for file_path in files:
        for number, line in enumerate(file_path.read_text(encoding="utf-8", errors="replace").splitlines(), start=1):
            if pattern.lower() in line.lower():
                matches.append(f"{file_path.relative_to(context.root)}:{number}:{line}")
                if len(matches) >= 200:
                    return "\n".join(matches)
    return "\n".join(matches) or "(no matches)"


def tool_run_shell(context, args):
    command = str(args.get("command", "")).strip()
    if not command:
        raise ValueError("command must not be empty")

    timeout = int(args.get("timeout", 20))
    if timeout < 1 or timeout > 120:
        raise ValueError("timeout must be in [1, 120]")

    result = subprocess.run(
        command,
        cwd=context.root,
        shell=True,
        capture_output=True,
        text=True,
        timeout=timeout,
        # 这里传入的是过滤后的环境变量，而不是直接继承整个父 shell 环境，
        # 目的是减少敏感信息被意外带进命令执行环境的风险。
        env=context.shell_env(),
    )
    return textwrap.dedent(
        f"""\
        exit_code: {result.returncode}
        stdout:
        {result.stdout.strip() or "(empty)"}
        stderr:
        {result.stderr.strip() or "(empty)"}
        """
    ).strip()


def tool_write_file(context, args):
    path = context.path(args["path"])
    content = str(args["content"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return f"wrote {path.relative_to(context.root)} ({len(content)} chars)"


def tool_patch_file(context, args):
    path = context.path(args["path"])
    if not path.is_file():
        raise ValueError("path is not a file")

    old_text = str(args.get("old_text", ""))
    if not old_text:
        raise ValueError("old_text must not be empty")

    if "new_text" not in args:
        raise ValueError("missing new_text")

    text = path.read_text(encoding="utf-8")
    count = text.count(old_text)
    if count != 1:
        raise ValueError(f"old_text must occur exactly once, found {count}")

    path.write_text(text.replace(old_text, str(args["new_text"]), 1), encoding="utf-8")
    return f"patched {path.relative_to(context.root)}"


def tool_delegate(context, args):
    """
    delegate 做成只读调查
    其本质上是一个受限子任务。代码里对子 agent 的约束很明确：
        approval_policy="never"
        read_only=True
        max_steps 更小
        深度超过上限时，连 delegate 都不再暴露
    """
    if context.depth >= context.max_depth:
        raise ValueError("delegate depth exceeded")

    task = str(args.get("task", "")).strip()

    if not task:
        raise ValueError("task must not be empty")

    return context.spawn_delegate(args)


_TOOL_RUNNERS = {
    "list_files": tool_list_files,
    "read_file": tool_read_file,
    "search": tool_search,
    "run_shell": tool_run_shell,
    "write_file": tool_write_file,
    "patch_file": tool_patch_file,
}