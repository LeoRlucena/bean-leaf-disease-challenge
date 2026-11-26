"""Modelos de Transfer Learning (etapa 5 / Entrega 2).

Este módulo reúne o estudo de arquiteturas candidatas, o construtor
`build_transfer_model` e helpers de fine-tuning usados na Parte C do notebook 05.
"""

from __future__ import annotations

import numpy as np
from tensorflow import keras
from tensorflow.keras import layers

from bean_leaf_disease.data.beans import DEFAULT_IMAGE_SIZE

# Especificações das arquiteturas candidatas de Transfer Learning.
#
# Fonte: documentação oficial do Keras Applications (keras.io/api/applications),
# referência ImageNet (entrada 224 x 224, top-1 single-crop). Os valores alimentam
# a tabela comparativa e o gráfico do notebook 05 e embasam a justificativa da escolha.
#
# Campos:
#   keras_app       nome da fábrica em tf.keras.applications
#   params_millions parâmetros totais do backbone (milhões)
#   size_mb         tamanho do arquivo de pesos ImageNet (MB)
#   imagenet_top1   acurácia top-1 em ImageNet (fração)
#   expects         faixa de pixels esperada pela função preprocess_input do backbone
#   note            leitura rápida para o estudo de viabilidade
BACKBONES: dict[str, dict] = {
    "MobileNetV2": {
        "keras_app": "MobileNetV2",
        "params_millions": 3.5,
        "size_mb": 14,
        "imagenet_top1": 0.713,
        "expects": "[-1, 1]",
        "note": (
            "Menor footprint. Projetado para dispositivos móveis/embarcados, "
            "adequado a inferência em campo. Treino rápido e menor risco de "
            "overfitting num dataset pequeno."
        ),
    },
    "EfficientNetB0": {
        "keras_app": "EfficientNetB0",
        "params_millions": 5.3,
        "size_mb": 29,
        "imagenet_top1": 0.771,
        "expects": "[0, 255]",  # a normalização está embutida no próprio backbone
        "note": (
            "Melhor acurácia ImageNet entre as candidatas, ainda compacto. "
            "A normalização é interna ao modelo (espera pixels em [0, 255])."
        ),
    },
    "ResNet50": {
        "keras_app": "ResNet50",
        "params_millions": 25.6,
        "size_mb": 98,
        "imagenet_top1": 0.749,
        "expects": "caffe (BGR, centrado)",
        "note": (
            "Arquitetura robusta e muito usada como referência. Maior e mais "
            "custosa; serve de teto de capacidade na comparação."
        ),
    },
}

# Faixa de pixels produzida pelo pipeline da etapa 1 (make_tf_dataset normaliza para [0, 1]).
PIPELINE_PIXEL_RANGE = (0.0, 1.0)


def _preprocess_input_for(backbone: str):
    """Retorna a função `preprocess_input` do backbone escolhido."""
    module = {
        "MobileNetV2": keras.applications.mobilenet_v2,
        "EfficientNetB0": keras.applications.efficientnet,
        "ResNet50": keras.applications.resnet50,
    }[backbone]
    return module.preprocess_input


def _base_model_factory(backbone: str):
    """Retorna a fábrica do backbone em `tf.keras.applications`."""
    return getattr(keras.applications, BACKBONES[backbone]["keras_app"])


def build_transfer_model(
    backbone: str = "MobileNetV2",
    *,
    num_classes: int = 3,
    image_size: tuple[int, int] = DEFAULT_IMAGE_SIZE,
    trainable_base: bool = False,
    dropout: float = 0.3,
) -> keras.Model:
    """Monta um modelo de Transfer Learning com backbone pré-treinado em ImageNet.

    Topo idêntico ao do baseline (notebook 03): GlobalAveragePooling2D → Dropout →
    Dense(softmax), para que a comparação baseline × transfer learning seja justa.

    O pipeline da etapa 1 entrega pixels em [0, 1], mas as funções `preprocess_input`
    das arquiteturas de ImageNet esperam pixels em [0, 255]. Por isso o modelo começa
    com `Rescaling(255.0)` seguido do `preprocess_input` do backbone — assim ele casa
    com `make_tf_dataset` sem precisar alterar o pré-processamento existente.

    Com `trainable_base=False` (padrão) o backbone fica congelado (feature extraction);
    a Entrega 2 pode liberar as camadas superiores para fine-tuning passando `True`.
    """
    if backbone not in BACKBONES:
        raise ValueError(
            f"Backbone '{backbone}' não suportado. Opções: {sorted(BACKBONES)}."
        )

    base_factory = _base_model_factory(backbone)
    preprocess_input = _preprocess_input_for(backbone)

    base_model = base_factory(
        include_top=False,
        weights="imagenet",
        input_shape=(*image_size, 3),
    )
    base_model.trainable = trainable_base

    inputs = keras.Input(shape=(*image_size, 3))
    # [0, 1] -> [0, 255] -> pré-processamento específico do backbone.
    x = layers.Rescaling(255.0)(inputs)
    x = preprocess_input(x)
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(dropout)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = keras.Model(inputs, outputs, name=f"transfer_{backbone.lower()}")
    # Referência explícita ao backbone aninhado (fine-tune / Grad-CAM).
    model.backbone = base_model
    return model


def get_backbone(model: keras.Model) -> keras.Model:
    """Devolve o backbone aninhado de um modelo criado por `build_transfer_model`."""
    backbone = getattr(model, "backbone", None)
    if isinstance(backbone, keras.Model):
        return backbone

    for layer in model.layers:
        if isinstance(layer, keras.Model) and len(layer.layers) > 1:
            return layer
    raise ValueError(
        "Não encontrei um backbone aninhado. Use um modelo de build_transfer_model."
    )


def enable_finetune(
    model: keras.Model,
    *,
    unfreeze_last: int = 40,
) -> keras.Model:
    """Libera as últimas `unfreeze_last` camadas do backbone para fine-tuning.

    Mantém as camadas iniciais congeladas (features genéricas de ImageNet) e
    treina só o topo do extrator + o classificador — padrão seguro em datasets
    pequenos como o Beans (~1k imagens de treino).
    """
    if unfreeze_last < 1:
        raise ValueError("unfreeze_last deve ser >= 1.")

    backbone = get_backbone(model)
    backbone.trainable = True
    freeze_until = max(0, len(backbone.layers) - unfreeze_last)
    for layer in backbone.layers[:freeze_until]:
        layer.trainable = False
    for layer in backbone.layers[freeze_until:]:
        layer.trainable = True
    return model


def count_trainable_weights(model: keras.Model) -> tuple[int, int]:
    """Retorna (parâmetros treináveis, parâmetros não treináveis)."""
    trainable = int(sum(int(np.prod(w.shape)) for w in model.trainable_weights))
    non_trainable = int(
        sum(int(np.prod(w.shape)) for w in model.non_trainable_weights)
    )
    return trainable, non_trainable
