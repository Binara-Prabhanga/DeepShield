"""_gradcam_map: the raw Grad-CAM array split out of _multimodal (C3 image deletion test)."""
import pytest

torch = pytest.importorskip('torch')

from app.services.model_adapter import _gradcam_map


def test_gradcam_map_shape_and_relu():
    acts = {'a': torch.ones(1, 1 + 4, 3)}                       # CLS + 2x2 patches, 3 channels
    grads = {'g': torch.tensor([[[9., 9., 9.], [1., 1., 1.], [1., 1., 1.], [-1., -1., -1.], [-1., -1., -1.]]])}
    cam = _gradcam_map(acts, grads)
    assert cam.shape == (2, 2) and (cam >= 0).all()
    # mean gradient over patches is 0 per channel -> all zeros after relu
    assert cam.sum() == 0


def test_gradcam_map_none_when_missing_or_not_square():
    assert _gradcam_map({}, {}) is None
    assert _gradcam_map({'a': torch.ones(1, 1 + 3, 2)}, {'g': torch.ones(1, 1 + 3, 2)}) is None
