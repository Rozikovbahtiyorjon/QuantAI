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

    def save(self, model):

        joblib.dump(
            model,
            self.model_path,
        )

        print()
        print("=" * 60)
        print("MODEL SAVED")
        print("=" * 60)
        print(self.model_path)
        print("=" * 60)

    def load(self):

        if not self.model_path.exists():
            return None

        print()
        print("=" * 60)
        print("MODEL LOADED")
        print("=" * 60)
        print(self.model_path)
        print("=" * 60)

        return joblib.load(self.model_path)