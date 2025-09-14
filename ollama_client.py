# -*- coding: utf-8 -*-
import subprocess
from validator import validate_push, autocorrect

def run_ollama(model: str, prompt: str, timeout_sec: int = 120) -> str:
    """
    Кросс-платформенный вызов Ollama:
    - Пишем prompt в stdin в UTF-8, чтобы не падать на символе ₸ в Windows (cp1251).
    """
    try:
        proc = subprocess.Popen(
            ["ollama", "run", model],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",   # <-- ключевое
            errors="replace"    # на всякий случай
        )
        out, err = proc.communicate(prompt, timeout=timeout_sec)
    except subprocess.TimeoutExpired:
        try:
            proc.kill()
        except Exception:
            pass
        return ""
    return (out or "").strip()

def generate_with_guardrails(model: str, system_prompt: str, user_prompt: str) -> str:
    text = run_ollama(model, f"{system_prompt}\n{user_prompt}")
    if text:
        v = validate_push(text)
        if v["ok"]:
            return text
        critique = "Исправьте текст по замечаниям: " + "; ".join(v["issues"]) + ". Перепишите, соблюдая правила."
        text2 = run_ollama(model, f"{system_prompt}\n{user_prompt}\n\n{critique}")
        if text2:
            v2 = validate_push(text2)
            if v2["ok"]:
                return text2
            return autocorrect(text2)
    return ""
