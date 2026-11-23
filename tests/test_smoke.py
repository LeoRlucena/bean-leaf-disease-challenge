def test_import_package():
    import bean_leaf_disease  # noqa: F401


def test_backbones_catalog():
    """O catálogo de arquiteturas do estudo tem as candidatas e os campos esperados."""
    from bean_leaf_disease.models import BACKBONES

    assert {"MobileNetV2", "EfficientNetB0", "ResNet50"} <= set(BACKBONES)
    for spec in BACKBONES.values():
        assert {"params_millions", "size_mb", "imagenet_top1", "note"} <= set(spec)
        assert 0.0 < spec["imagenet_top1"] < 1.0


def test_build_transfer_model_frozen_head():
    """build_transfer_model monta a saída de 3 classes com o backbone congelado."""
    from bean_leaf_disease.models import build_transfer_model

    model = build_transfer_model("MobileNetV2", num_classes=3, image_size=(96, 96))
    assert model.output_shape == (None, 3)
    # Backbone congelado por padrão (feature extraction): há pesos não treináveis.
    assert len(model.non_trainable_weights) > 0


def test_gradcam_heatmap_shape_and_range():
    """Grad-CAM retorna heatmap no shape do mapa convolucional, normalizado em [0, 1]."""
    import numpy as np
    from tensorflow import keras

    from bean_leaf_disease.evaluation import find_last_conv_layer, make_gradcam_heatmap

    inputs = keras.Input(shape=(16, 16, 3))
    x = keras.layers.Conv2D(4, 3, padding="same", activation="relu", name="conv")(inputs)
    x = keras.layers.GlobalAveragePooling2D()(x)
    outputs = keras.layers.Dense(3, activation="softmax")(x)
    model = keras.Model(inputs, outputs)

    assert find_last_conv_layer(model) == "conv"

    img = np.random.rand(1, 16, 16, 3).astype("float32")
    heatmap = make_gradcam_heatmap(img, model, "conv")
    assert heatmap.shape == (16, 16)
    assert 0.0 <= float(heatmap.min()) and float(heatmap.max()) <= 1.0
