GROUP_LABELS = {
    "linear": "Линейные",
}

TARGET_LABELS = {
    "SP63": "СП 63.13330",
    "RUK78": "Руководство 1978",
}

MODEL_LABELS = {
    "linear": "Линейная регрессия",
    "ridge": "Ridge-регрессия"
}

Q_UNIT = "кН"

def method_label(name):
    return MODEL_LABELS.get(name)


def target_label(name):
    return TARGET_LABELS.get(name, name)
