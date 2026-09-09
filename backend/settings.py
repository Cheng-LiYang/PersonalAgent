"""Persistent application settings with Windows DPAPI secret protection."""
from __future__ import annotations

import base64
import ctypes
import json
import os
from ctypes import wintypes
from pathlib import Path
from typing import Any


class DataBlob(ctypes.Structure):
    _fields_ = [("cbData", wintypes.DWORD), ("pbData", ctypes.POINTER(ctypes.c_byte))]


def _blob(data: bytes) -> tuple[DataBlob, ctypes.Array]:
    buffer = ctypes.create_string_buffer(data)
    return DataBlob(len(data), ctypes.cast(buffer, ctypes.POINTER(ctypes.c_byte))), buffer


def protect_secret(secret: str) -> str:
    if not secret:
        return ""
    if os.name != "nt":
        return base64.b64encode(secret.encode()).decode()
    source, buffer = _blob(secret.encode("utf-8"))
    output = DataBlob()
    if not ctypes.windll.crypt32.CryptProtectData(ctypes.byref(source), None, None, None, None, 0, ctypes.byref(output)):
        raise ctypes.WinError()
    try:
        encrypted = ctypes.string_at(output.pbData, output.cbData)
        return base64.b64encode(encrypted).decode()
    finally:
        ctypes.windll.kernel32.LocalFree(output.pbData)


def unprotect_secret(value: str) -> str:
    if not value:
        return ""
    encrypted = base64.b64decode(value)
    if os.name != "nt":
        return encrypted.decode()
    source, buffer = _blob(encrypted)
    output = DataBlob()
    if not ctypes.windll.crypt32.CryptUnprotectData(ctypes.byref(source), None, None, None, None, 0, ctypes.byref(output)):
        raise ctypes.WinError()
    try:
        return ctypes.string_at(output.pbData, output.cbData).decode("utf-8")
    finally:
        ctypes.windll.kernel32.LocalFree(output.pbData)


class SettingsStore:
    def __init__(self, data_dir: str | Path) -> None:
        self.path = Path(data_dir) / "model_settings.json"

    def load(self) -> dict[str, Any]:
        if not self.path.is_file():
            return {}
        try:
            return json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}

    def save_model(self, settings: dict[str, Any], api_key: str) -> None:
        data = self.load()
        data["model"] = {key: settings[key] for key in ("provider", "model", "base_url")}
        if api_key:
            data["api_key_encrypted"] = protect_secret(api_key)
        self._write(data)

    def load_model(self) -> dict[str, Any] | None:
        data = self.load()
        model = data.get("model")
        if not model:
            return None
        try:
            return {**model, "api_key": unprotect_secret(data.get("api_key_encrypted", ""))}
        except (ValueError, OSError):
            return None

    def save_knowledge_base(self, path: str) -> None:
        data = self.load()
        data["knowledge_base"] = path
        self._write(data)

    def _write(self, data: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(".tmp")
        temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(self.path)
