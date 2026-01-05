from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional, TextIO


@dataclass
class LogPaths:
    log_file: Path


class Tee(TextIO):
    """
    一个简单的 stdout/stderr tee：同时写入原始流与日志文件。
    """

    def __init__(self, original: TextIO, file: TextIO):
        self._original = original
        self._file = file

    def write(self, s: str) -> int:
        n1 = self._original.write(s)
        n2 = self._file.write(s)
        # 尽量实时落盘，适合后台训练观察
        self._original.flush()
        self._file.flush()
        return max(n1, n2)

    def flush(self) -> None:
        self._original.flush()
        self._file.flush()


def build_log_path(
    log_dir: Path,
    mode: str,
    is_train: bool,
    prefix: str = "run",
    ts: Optional[datetime] = None,
) -> LogPaths:
    ts = ts or datetime.now()
    log_dir.mkdir(parents=True, exist_ok=True)
    stamp = ts.strftime("%Y%m%d-%H%M%S")
    kind = "train" if is_train else "infer"
    fname = f"{prefix}_{mode}_{kind}_{stamp}_pid{os.getpid()}.log"
    return LogPaths(log_file=log_dir / fname)


def setup_std_stream_logging(log_file: Path, tee_to_console: bool = True) -> None:
    """
    把 stdout/stderr 重定向到 log_file。
    - tee_to_console=True：同时保留终端输出（但你可以不看）
    - tee_to_console=False：完全静默，只写日志文件
    """
    log_file.parent.mkdir(parents=True, exist_ok=True)
    f = open(log_file, "a", buffering=1, encoding="utf-8")

    if tee_to_console:
        sys.stdout = Tee(sys.__stdout__, f)  # type: ignore
        sys.stderr = Tee(sys.__stderr__, f)  # type: ignore
    else:
        sys.stdout = f  # type: ignore
        sys.stderr = f  # type: ignore


