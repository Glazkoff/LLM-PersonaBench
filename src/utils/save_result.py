import json
import os


def save_log(log_data, results_dir, name_file):
    """Сохраняет лог в файл"""
    log_file = results_dir / name_file
    # Создаем директорию, если её нет
    os.makedirs(results_dir, exist_ok=True)
    with open(log_file, 'w', encoding='utf-8') as f:
        json.dump(log_data, f, indent=2, ensure_ascii=False, default=str)


def save_jsonl(rows, results_dir, name_file):
    """Сохраняет список объектов в jsonl файл."""
    log_file = results_dir / name_file
    os.makedirs(results_dir, exist_ok=True)
    with open(log_file, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, default=str))
            f.write("\n")
