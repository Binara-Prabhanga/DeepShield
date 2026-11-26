"""Model adapter: simulated scores, or real inference from Colab-exported artifacts.

MODEL_MODE=simulation|demo  -> random placeholder scores (never evidence of credibility).
Any other MODEL_MODE        -> real inference. Requires `pip install -r requirements-ml.txt`.

Artifact contract (must match what the Colab notebook exports):

TEXT_MODEL_PATH/                      # output of model.save_pretrained() + tokenizer.save_pretrained()
    config.json                       # id2label must contain a "real"/"true" style label
    tokenizer files ...
    calibration.json   (optional)     # {"temperature": T}  fitted on held-out data, see fit_temperature()

MULTIMODAL_MODEL_PATH/
    fusion_config.json                # {"clip_model": "openai/clip-vit-base-patch32" or a subdir name,
                                      #  "hidden_dim": 512, "dropout": 0.1, "real_index": 1,
                                      #  "temperature": 1.0}
    fusion_head.pt                    # state_dict of FusionHead below (identical architecture!)
    <clip subdir>/     (optional)     # CLIPModel + CLIPProcessor save_pretrained() if fine-tuned

Fusion input = concat[t, i, t*i, |t-i|] of L2-normalised CLIP text/image embeddings.

SINHALA_MODEL_PATH/                   # research/eval_sinhala.py --export
    model.joblib                      # calibrated char n-gram classifier (sklearn)
    config.json                       # {"misleading_index": 1, ...}

Text that is mostly Sinhala script goes to the Sinhala model (mode "model-sinhala"): DistilBERT,
SHAP and CLIP are English-only, so they are skipped for it.
"""
from __future__ import annotations

import base64
import io
import json
import secrets
from dataclasses import dataclass, asdict, field
from functools import lru_cache
from pathlib import Path

from app.config import get_settings

_rng = secrets.SystemRandom()
_REAL_LABEL_NAMES = {'real', 'true', 'authentic', 'reliable', 'legit', 'label_1'}


@dataclass
class ModelOutput:
    text_probability_real: float
    image_text_alignment: float | None
    fused_probability_real: float
    mode: str
    explanation_tokens: list[dict]
    gradcam_available: bool = False
    shap_available: bool = False
    gradcam_image: str | None = None          # data:image/png;base64 heatmap overlay
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


# --------------------------------------------------------------------------- calibration

def fit_temperature(logits, labels, max_iter: int = 200) -> float:
    """Fit temperature scaling on HELD-OUT logits (N, C) and int labels (N,). Run in Colab,
    then save {"temperature": T} to calibration.json / fusion_config.json."""
    import torch
    logits = torch.as_tensor(logits, dtype=torch.float32)
    labels = torch.as_tensor(labels, dtype=torch.long)
    log_t = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.05, max_iter=max_iter)
    nll = torch.nn.CrossEntropyLoss()

    def closure():
        opt.zero_grad()
        loss = nll(logits / log_t.exp(), labels)
        loss.backward()
        return loss

    opt.step(closure)
    return float(log_t.exp().item())


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding='utf-8')) if path.is_file() else {}


# --------------------------------------------------------------------------- loaders

def _device():
    import torch
    return 'cuda' if torch.cuda.is_available() else 'cpu'


@lru_cache
def _text_bundle():
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    path = get_settings().text_model_path
    if not path or not Path(path).is_dir():
        raise RuntimeError(f'TEXT_MODEL_PATH is not a directory: {path!r}')
    tokenizer = AutoTokenizer.from_pretrained(path)
    model = AutoModelForSequenceClassification.from_pretrained(path).to(_device()).eval()
    id2label = {int(k): str(v).lower() for k, v in model.config.id2label.items()}
    real_idx = next((i for i, name in id2label.items() if name in _REAL_LABEL_NAMES), None)
    if real_idx is None:
        raise RuntimeError(f'Cannot find a "real" label in id2label={id2label}. Fix config.json in Colab.')
    temperature = float(_read_json(Path(path) / 'calibration.json').get('temperature', 1.0))
    return tokenizer, model, real_idx, temperature


def _fusion_head_cls():
    import torch
    import torch.nn as nn

    class FusionHead(nn.Module):
        def __init__(self, embed_dim: int, hidden_dim: int, dropout: float):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(embed_dim * 4, hidden_dim), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden_dim, 2),
            )

        def forward(self, t, i):
            return self.net(torch.cat([t, i, t * i, (t - i).abs()], dim=-1))

    return FusionHead


@lru_cache
def _multimodal_bundle():
    import torch
    from transformers import CLIPModel, CLIPProcessor
    root = Path(get_settings().multimodal_model_path or '')
    cfg = _read_json(root / 'fusion_config.json')
    if not cfg or not (root / 'fusion_head.pt').is_file():
        raise RuntimeError(f'MULTIMODAL_MODEL_PATH must contain fusion_config.json and fusion_head.pt: {str(root)!r}')
    clip_ref = cfg.get('clip_model', 'openai/clip-vit-base-patch32')
    clip_src = str(root / clip_ref) if (root / clip_ref).is_dir() else clip_ref
    device = _device()
    clip = CLIPModel.from_pretrained(clip_src).to(device).eval()
    processor = CLIPProcessor.from_pretrained(clip_src)
    head = _fusion_head_cls()(clip.config.projection_dim, int(cfg.get('hidden_dim', 512)), float(cfg.get('dropout', 0.0)))
    head.load_state_dict(torch.load(root / 'fusion_head.pt', map_location=device))
    head.to(device).eval()
    return clip, processor, head, int(cfg.get('real_index', 1)), float(cfg.get('temperature', 1.0))


# --------------------------------------------------------------------------- inference

def _text_probs(texts: list[str]):
    import torch
    tokenizer, model, real_idx, temperature = _text_bundle()
    enc = tokenizer(texts, truncation=True, max_length=512, padding=True, return_tensors='pt').to(model.device)
    with torch.no_grad():
        logits = model(**enc).logits / temperature
    return torch.softmax(logits, dim=-1)[:, real_idx].cpu().numpy()


def _shap_tokens(text: str, top_k: int = 12) -> list[dict]:
    import shap
    tokenizer = _text_bundle()[0]
    explainer = shap.Explainer(lambda xs: _text_probs([str(x) for x in xs]), shap.maskers.Text(tokenizer))
    sv = explainer([text[:2000]], max_evals=300, silent=True)[0]
    pairs = [(str(tok).strip(), float(val)) for tok, val in zip(sv.data, sv.values) if str(tok).strip()]
    pairs.sort(key=lambda p: abs(p[1]), reverse=True)
    # positive value -> pushes toward "real", negative -> toward "fake"
    return [{'token': tok, 'value': round(val, 4)} for tok, val in pairs[:top_k]]


def _gradcam_png(cam, image) -> str:
    import numpy as np
    from PIL import Image
    cam = cam - cam.min()
    cam = cam / (cam.max() + 1e-8)
    heat = Image.fromarray(np.uint8(cam * 255)).resize(image.size, Image.BILINEAR)
    h = np.asarray(heat, dtype=np.float32) / 255.0
    rgb = np.stack([h, np.zeros_like(h), 1.0 - h], axis=-1) * 255  # red = high attribution
    overlay = Image.blend(image.convert('RGB'), Image.fromarray(np.uint8(rgb)), 0.45)
    overlay.thumbnail((512, 512))
    buf = io.BytesIO()
    overlay.save(buf, format='PNG')
    return 'data:image/png;base64,' + base64.b64encode(buf.getvalue()).decode()


def _multimodal(text: str, image_bytes: bytes):
    """Returns (fused_p_real, clip_cosine, gradcam_data_url)."""
    from PIL import Image
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    fused_p, cosine, cam = _multimodal_scores(text, image)
    gradcam = _gradcam_png(cam, image) if cam is not None else None
    return fused_p, cosine, gradcam


def _multimodal_scores(text: str, image):
    """Fused P(real), CLIP cosine and the raw Grad-CAM map (or None) for a PIL RGB image."""
    import torch
    clip, processor, head, real_idx, temperature = _multimodal_bundle()
    inputs = processor(text=[text], images=image, return_tensors='pt', truncation=True,
                       max_length=77, padding=True).to(clip.device)

    # Grad-CAM on the second-to-last ViT block: the last block's patch tokens never reach the
    # pooled CLS output, so their gradients are zero.
    acts, grads = {}, {}
    layer = clip.vision_model.encoder.layers[-2]
    fwd = layer.register_forward_hook(lambda m, i, o: acts.__setitem__('a', o[0] if isinstance(o, tuple) else o))
    bwd = layer.register_full_backward_hook(lambda m, gi, go: grads.__setitem__('g', go[0]))
    try:
        with torch.enable_grad():
            t = clip.get_text_features(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'])
            i = clip.get_image_features(pixel_values=inputs['pixel_values'].requires_grad_(True))
            t, i = t / t.norm(dim=-1, keepdim=True), i / i.norm(dim=-1, keepdim=True)
            logits = head(t, i) / temperature
            probs = torch.softmax(logits, dim=-1)
            pred = int(probs.argmax(dim=-1))
            clip.zero_grad(); head.zero_grad()
            logits[0, pred].backward()
    finally:
        fwd.remove(); bwd.remove()

    fused_p = float(probs[0, real_idx].detach())
    cosine = float((t * i).sum().detach())
    return fused_p, cosine, _gradcam_map(acts, grads)


def _gradcam_map(acts: dict, grads: dict):
    """Raw Grad-CAM map (side x side numpy array over ViT patches) from the hooked activations
    and gradients, or None if they are missing or the patches do not form a square grid."""
    import torch
    if 'a' not in acts or 'g' not in grads:
        return None
    a, g = acts['a'][0, 1:].detach(), grads['g'][0, 1:].detach()   # drop CLS token
    side = int(round(a.shape[0] ** 0.5))
    if side * side != a.shape[0]:
        return None
    return torch.relu((a * g.mean(dim=0, keepdim=True)).sum(dim=-1)).reshape(side, side).cpu().numpy()


def is_sinhala(text: str, min_share: float = 0.5) -> bool:
    """True when at least min_share of the letters are Sinhala script (U+0D80-U+0DFF)."""
    letters = [c for c in text if c.isalpha() or '඀' <= c <= '෿']
    return bool(letters) and sum('඀' <= c <= '෿' for c in letters) / len(letters) >= min_share


@lru_cache
def _sinhala_bundle():
    import joblib
    root = Path(get_settings().sinhala_model_path or '')
    if not (root / 'model.joblib').is_file():
        raise RuntimeError(f'SINHALA_MODEL_PATH must contain model.joblib: {str(root)!r}')
    cfg = _read_json(root / 'config.json')
    return joblib.load(root / 'model.joblib'), int(cfg.get('misleading_index', 1))


def _sinhala_p_real(text: str) -> float:
    model, misleading_idx = _sinhala_bundle()
    return 1.0 - float(model.predict_proba([text])[0, misleading_idx])


def _run_sinhala(text: str, has_image: bool) -> ModelOutput:
    text_p = round(_sinhala_p_real(text), 3)
    notes = ['Sinhala text: scored by the Sinhala character n-gram model (LIRNEasia corpus, cross-validated '
             'AUROC 0.75). Word-level explanations are not available for it.']
    if has_image:
        notes.append('Image not analysed: the image-text model reads English only.')
    return ModelOutput(text_p, None, text_p, 'model-sinhala', [], notes=notes)


def run_models(text: str, has_image: bool, image_bytes: bytes | None = None) -> ModelOutput:
    settings = get_settings()
    if settings.model_mode not in ('simulation', 'demo'):
        sinhala = is_sinhala(text)
        if sinhala and settings.sinhala_model_path:
            return _run_sinhala(text, has_image)
        notes: list[str] = []
        if sinhala:
            notes.append('Sinhala text detected but SINHALA_MODEL_PATH is not set: the English model cannot '
                         'read Sinhala script, so this score is unreliable.')
        text_p = round(float(_text_probs([text])[0]), 3)

        tokens, shap_ok = [], False
        try:
            tokens, shap_ok = _shap_tokens(text), True
        except Exception as exc:  # SHAP is best-effort; never fail the prediction because of it
            notes.append(f'SHAP unavailable: {type(exc).__name__}: {exc}')

        align, fused_p, gradcam = None, text_p, None
        if has_image and image_bytes:
            fused, cosine, gradcam = _multimodal(text, image_bytes)
            fused_p = round(fused, 3)
            align = round(min(1.0, max(0.0, cosine)), 3)   # raw CLIP cosine similarity, not a probability
        elif has_image:
            notes.append('Image flagged but no bytes passed; multimodal model skipped.')

        return ModelOutput(text_p, align, fused_p, 'model', tokens,
                           gradcam_available=gradcam is not None, shap_available=shap_ok,
                           gradcam_image=gradcam, notes=notes)

    # The classifier probabilities below are simulated.
    # They must NEVER be represented as evidence of news credibility.
    text_p = round(_rng.uniform(0.17, 0.88), 3)
    align_p = round(_rng.uniform(0.19, 0.95), 3) if has_image else None
    fused_p = round(min(0.96, max(0.05, text_p * 0.72 + (align_p if align_p is not None else text_p) * 0.28 + _rng.uniform(-0.10, 0.10))), 3)
    return ModelOutput(text_p, align_p, fused_p, 'simulated', [])
