"""Session JSON persistence."""

import json
from pathlib import Path

class SessionStore:
    """
    把 Agent 的会话状态保存到本地 JSON 文件中，也可以从本地恢复历史会话。

    路径：.lincoder/sessions/<session_id>.json
    内容：
        {
            "id": "session_id",
            "created_at": "2024-06-01T12:00:00Z",
            "workspace_root": "",
            "history": ,
            "memory":
        }
    """
    def __init__(self, root):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def path(self, session_id):
        return self.root / f"{session_id},json"

    def save(self, session):
        path = self.path(session["id"])
        path.write_text(json.dumps(session, indent=2), encoding="utf-8")
        return path

    def load(self, session_id):
        return json.loads(self.path(session_id).read_text(encoding="utf-8"))

    def latest(self):
        """
        找到最近一次保存的 session

        self.root.glob("*.json")：找到最近一次保存的 session：找到 root 目录下所有的 JSON 文件
        sorted(..., key=lambda path: path.stat().st_mtime)：按照文件的修改时间排序
        """
        files = sorted(self.root.glob("*.json"), key=lambda path: path.stat().st_mtime)
        return files[-1].stem if files else None