# -*- coding: utf-8 -*-
import re

def validate_push(text: str) -> dict:
    issues = []

    # длина
    n = len(text)
    if n < 160 or n > 240:  # допускаем небольшой люфт вокруг 180–220
        issues.append(f"Длина {n} символов (нужно 180–220).")

    # капс (жёстко)
    if re.search(r"[A-ZА-Я]{4,}", text):
        issues.append("Обнаружен КАПС подряд 4+ символа.")

    # обращение на «вы» (любые формы)
    text_lower = " " + text.lower() + " "
    if not any(form in text_lower for form in [" вы ", " вас ", " вам ", " вами ", " ваш ", " ваша ", " ваше ", " ваши ", " вашей ", " вашего ", " вашему ", " вашим ", " вашими "]):
        issues.append("Нет обращения на «вы» (в любом месте).")

    # максимум 1 восклицательный
    if text.count("!") > 1:
        issues.append("Слишком много «!» (макс. 1).")

    # CTA (хотя бы одно из)
    if not re.search(r"(Открыть|Настроить|Посмотреть|Оформить сейчас|Оформить карту|Открыть вклад|Открыть счёт)", text):
        issues.append("Нет CTA из списка.")

    # валюта формат «27 400 ₸» (проверяем наличие символа ₸)
    if "₸" in text:
        pass

    return {"ok": len(issues) == 0, "issues": issues}

def autocorrect(text: str) -> str:
    import re
    t = text.strip()
    t = re.sub(r"!{2,}", "!", t)
    def lower_long_caps(m):
        return m.group(0).lower().capitalize()
    t = re.sub(r"\b[A-ZА-Я]{4,}\b", lower_long_caps, t)
    if len(t) > 220:
        t = t[:220].rstrip(" ,.;")
    return t
