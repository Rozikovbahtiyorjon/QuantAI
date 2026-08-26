"""
====================================================
QuantAI Professional v5.1
Model Manager
====================================================

Сохранение и загрузка обученных моделей.
"""

from pathlib import Path
import joblib

MODEL_FOLDER = Path("models")
MODEL_FOLDER.mkdir(exist_ok=True)

DEFAULT_MODEL = MODEL_FOLDER / "quantai_v5.pkl"


class ModelManager:

    def __init__(self):
        self.model_path = DEFAULT_MODEL

    def save(self, model, path: Path | str | None = None):
        save_path = Path(path) if path is not None else self.model_path
        save_path.parent.mkdir(parents=True, exist_ok=True)

        joblib.dump(
            model,
            save_path,
        )

        print()
        print("=" * 60)
        print("MODEL SAVED")
        print("=" * 60)
        print(save_path)
        print("=" * 60)

    def load(self, path: Path | str | None = None):
        load_path = Path(path) if path is not None else self.model_path

        if not load_path.exists():
            return None

        print()
        print("=" * 60)
        print("MODEL LOADED")
        print("=" * 60)
        print(load_path)
        print("=" * 60)

        return joblib.load(load_path)