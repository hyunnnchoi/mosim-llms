"""Model registry for PyTorch CNN benchmarks."""

from __future__ import annotations

from functools import partial
from typing import Any, Callable, Dict, Iterable

try:
    import torch.nn as nn
    from torchvision import models as tv_models
except ModuleNotFoundError as err:  # pragma: no cover - dependency guard
    raise ModuleNotFoundError(
        "PyTorch 및 TorchVision이 필요합니다. 'pip install torch torchvision'으로 설치하세요."
    ) from err

from .densenet import DENSENET_FACTORIES
from .resnet_cifar import RESNET_CIFAR_FACTORIES


def _override_classifier(module: nn.Module, in_features_attr: str, num_classes: int) -> nn.Module:
    in_features = getattr(module, in_features_attr).in_features
    setattr(module, in_features_attr, nn.Linear(in_features, num_classes))
    return module


def _build_resnet50(num_classes: int, **kwargs: Any) -> nn.Module:
    model = tv_models.resnet50(weights=None, **{k: v for k, v in kwargs.items() if k != "aux_logits"})
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def _build_vgg16(num_classes: int, **kwargs: Any) -> nn.Module:
    # [NOTE, hyunnnchoi, 2025.11.04] aux_logits 인자 필터링 추가 (vgg16은 aux_logits를 지원하지 않음)
    model = tv_models.vgg16(weights=None, **{k: v for k, v in kwargs.items() if k != "aux_logits"})
    model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
    return model


def _build_googlenet(num_classes: int, aux_logits: bool = True, **kwargs: Any) -> nn.Module:
    model = tv_models.googlenet(weights=None, aux_logits=aux_logits, **kwargs)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    if aux_logits and model.aux_logits:  # type: ignore[attr-defined]
        model.aux1.fc2 = nn.Linear(model.aux1.fc2.in_features, num_classes)  # type: ignore[assignment]
        model.aux2.fc2 = nn.Linear(model.aux2.fc2.in_features, num_classes)  # type: ignore[assignment]
    return model


def _build_inception3(num_classes: int, aux_logits: bool = True, **kwargs: Any) -> nn.Module:
    model = tv_models.inception_v3(weights=None, aux_logits=aux_logits, **kwargs)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    if aux_logits and model.aux_logits:  # type: ignore[attr-defined]
        model.AuxLogits.fc = nn.Linear(model.AuxLogits.fc.in_features, num_classes)  # type: ignore[assignment]
    return model


MODEL_FACTORIES: Dict[str, Callable[..., nn.Module]] = {
    **DENSENET_FACTORIES,
    **RESNET_CIFAR_FACTORIES,
    "resnet50": _build_resnet50,
    "vgg16": _build_vgg16,
    "googlenet": _build_googlenet,
    "inception3": _build_inception3,
}


MODEL_DEFAULTS: Dict[str, Dict[str, Any]] = {
    "densenet40_k12": {"dataset": "cifar10", "num_classes": 10, "image_size": 32},
    "densenet100_k12": {"dataset": "cifar10", "num_classes": 10, "image_size": 32},
    "resnet44": {"dataset": "cifar10", "num_classes": 10, "image_size": 32},
    "resnet110": {"dataset": "cifar10", "num_classes": 10, "image_size": 32},
    "resnet50": {"dataset": "imagenet", "num_classes": 1000, "image_size": 224},
    "vgg16": {"dataset": "imagenet", "num_classes": 1000, "image_size": 224},
    "googlenet": {"dataset": "imagenet", "num_classes": 1000, "image_size": 224},
    "inception3": {"dataset": "imagenet", "num_classes": 1000, "image_size": 299},
}


def list_models() -> Iterable[str]:
    return sorted(MODEL_FACTORIES.keys())


def get_model_defaults(name: str) -> Dict[str, Any]:
    if name not in MODEL_DEFAULTS:
        raise KeyError(f"Unknown model '{name}'. Available models: {', '.join(list_models())}")
    return MODEL_DEFAULTS[name].copy()


def create_model(name: str, num_classes: int, **kwargs: Any) -> nn.Module:
    if name not in MODEL_FACTORIES:
        raise KeyError(f"Unknown model '{name}'. Available models: {', '.join(list_models())}")
    
    # [NOTE, hyunnnchoi, 2025.11.04] aux_logits는 googlenet과 inception3만 지원하므로 다른 모델에서는 필터링
    factory = MODEL_FACTORIES[name]
    if name not in ("googlenet", "inception3"):
        kwargs = {k: v for k, v in kwargs.items() if k != "aux_logits"}
    
    return factory(num_classes=num_classes, **kwargs)


__all__ = ["create_model", "list_models", "get_model_defaults", "MODEL_FACTORIES", "MODEL_DEFAULTS"]

