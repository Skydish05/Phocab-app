import csv
import io
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from PIL import Image, ImageOps

# SigLIP was trained on captions, so a sentence scores better than a bare noun.
# This is the template transformers' zero-shot pipeline uses for SigLIP.
PROMPT_TEMPLATE = "This is a photo of {}."


@dataclass(frozen=True)
class Prediction:
    label_en: str
    score: float


class Classifier(Protocol):
    """What main.py and bench.py depend on, so tests can swap in a fake model."""

    model_id: str

    def classify(self, image: Image.Image, top_k: int = 3) -> list[Prediction]: ...


def load_labels(path: Path) -> list[str]:
    """Read candidate labels from labels_ko.csv, failing fast on bad data.

    A duplicate label would split probability between two identical rows, and a
    blank one would be a meaningless candidate, so both are startup errors.
    """
    with path.open(newline="", encoding="utf-8") as f:
        labels = [row["label_en"].strip() for row in csv.DictReader(f)]
    if not labels:
        raise ValueError(f"{path} has no labels")
    if any(not label for label in labels):
        raise ValueError(f"{path} has a blank label_en")
    duplicates = sorted({label for label in labels if labels.count(label) > 1})
    if duplicates:
        raise ValueError(f"{path} has duplicate labels: {duplicates}")
    return labels


def prepare_image(data: bytes) -> Image.Image:
    """Decode upload bytes into an upright RGB image.

    Phone cameras store rotation in EXIF instead of rotating pixels; without
    exif_transpose the model would see many photos sideways.
    Raises PIL.UnidentifiedImageError / OSError for non-images.
    """
    image = Image.open(io.BytesIO(data))
    image.load()
    return ImageOps.exif_transpose(image).convert("RGB")


def top_k_predictions(labels: Sequence[str], scores: Sequence[float], k: int) -> list[Prediction]:
    ranked = sorted(zip(labels, scores, strict=True), key=lambda pair: pair[1], reverse=True)
    return [Prediction(label, float(score)) for label, score in ranked[:k]]


class SiglipClassifier:
    """Zero-shot classifier over a fixed vocabulary.

    Uses AutoModel directly instead of transformers' pipeline(): the pipeline
    re-encodes every candidate label on every call, but our ~300 labels never
    change, so we encode them once here and each request only encodes the image.
    """

    def __init__(self, model_id: str, labels: Sequence[str]) -> None:
        # Imported here so the API, tests and CI can import this module without
        # installing torch (hundreds of MB).
        import torch
        from transformers import AutoModel, AutoProcessor

        self._torch = torch
        self.model_id = model_id
        self.labels = list(labels)
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModel.from_pretrained(model_id).eval()

        prompts = [PROMPT_TEMPLATE.format(label) for label in self.labels]
        with torch.inference_mode():
            # SigLIP was trained with text padded to 64 tokens; other padding
            # lengths change the embeddings and hurt accuracy.
            text_inputs = self.processor(
                text=prompts, padding="max_length", max_length=64, return_tensors="pt"
            )
            # transformers 5 returns an output object, not a tensor; pooler_output
            # is the embedding, the same field the model's own forward() uses.
            text_embeds = self.model.get_text_features(**text_inputs).pooler_output
            self.text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)

    def classify(self, image: Image.Image, top_k: int = 3) -> list[Prediction]:
        torch = self._torch
        with torch.inference_mode():
            image_inputs = self.processor(images=image, return_tensors="pt")
            image_embeds = self.model.get_image_features(**image_inputs).pooler_output
            image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
            logits = (
                image_embeds @ self.text_embeds.T * self.model.logit_scale.exp()
                + self.model.logit_bias
            )
            # Sigmoid, not softmax: SigLIP is trained so each label's score is
            # an independent yes/no probability. The scores don't sum to 1, so a
            # low top score means "probably none of these", which is the signal
            # the API's VLM fallback threshold needs.
            scores = torch.sigmoid(logits)[0].tolist()
        return top_k_predictions(self.labels, scores, top_k)
