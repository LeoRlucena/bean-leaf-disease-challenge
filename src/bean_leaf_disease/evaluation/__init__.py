"""Grad-CAM para explicabilidade (etapa 4, bônus).

Grad-CAM (Gradient-weighted Class Activation Mapping) destaca as regiões da imagem
que mais influenciaram a predição de uma classe. Serve para verificar a hipótese
levantada na EDA (notebook 02): o modelo olha para a folha e as lesões, ou se apoia
no fundo (solo, mão, outras folhas)?

As funções são genéricas: valem para a CNN baseline do notebook 03 e para os modelos
de Transfer Learning da Entrega 2.
"""

from __future__ import annotations

import matplotlib as mpl
import numpy as np
import tensorflow as tf
from tensorflow import keras


def find_last_conv_layer(model: keras.Model) -> str:
    """Retorna o nome da última camada convolucional do modelo.

    Percorre as camadas de trás para frente procurando uma saída 4D (lote, altura,
    largura, canais) — o mapa de ativação sobre o qual o Grad-CAM opera. No baseline
    do notebook 03 isso corresponde a `conv2d_2`.
    """
    for layer in reversed(model.layers):
        output_shape = getattr(layer, "output_shape", None)
        if output_shape is None:
            # Keras 3: a forma fica em layer.output.shape.
            try:
                output_shape = layer.output.shape
            except AttributeError:
                continue
        if isinstance(output_shape, (list, tuple)) and len(output_shape) == 4:
            return layer.name
    raise ValueError("Nenhuma camada convolucional (saída 4D) encontrada no modelo.")


def make_gradcam_heatmap(
    img_array: np.ndarray,
    model: keras.Model,
    last_conv_layer_name: str,
    pred_index: int | None = None,
) -> np.ndarray:
    """Calcula o mapa de calor Grad-CAM para uma imagem.

    Args:
        img_array: imagem já pré-processada, com dimensão de lote — shape (1, H, W, 3).
        model: modelo treinado.
        last_conv_layer_name: nome da camada convolucional alvo (ver `find_last_conv_layer`).
        pred_index: índice da classe a explicar. Se None, usa a classe predita.

    Returns:
        Mapa de calor 2D (float32) normalizado em [0, 1], no tamanho do mapa de ativação.
    """
    img_tensor = tf.convert_to_tensor(img_array, dtype=tf.float32)

    # Chamar o modelo uma vez garante que ele esteja construído (um Sequential carregado
    # do disco só materializa o grafo depois da primeira chamada).
    model(img_tensor, training=False)

    # Monta um modelo auxiliar que devolve (saída da conv alvo, predições). Propagar a
    # entrada pelas camadas em ordem funciona tanto para Sequential (inclusive carregado
    # do disco, onde model.get_layer(...).output não é acessível em Keras 3) quanto para
    # modelos funcionais.
    inputs = keras.Input(shape=model.inputs[0].shape[1:])
    conv_output = None
    x = inputs
    for layer in model.layers:
        # Modelos funcionais expõem o InputLayer em model.layers; ele não é "chamável"
        # como as demais camadas e já está representado por `inputs`, então o pulamos.
        if isinstance(layer, keras.layers.InputLayer):
            continue
        x = layer(x)
        if layer.name == last_conv_layer_name:
            conv_output = x
    if conv_output is None:
        raise ValueError(f"Camada '{last_conv_layer_name}' não encontrada no modelo.")
    grad_model = keras.models.Model(inputs, [conv_output, x])

    with tf.GradientTape() as tape:
        conv_output, predictions = grad_model(img_tensor, training=False)
        if pred_index is None:
            pred_index = int(tf.argmax(predictions[0]))
        class_channel = predictions[:, pred_index]

    # Gradiente da classe alvo em relação ao mapa de ativação.
    grads = tape.gradient(class_channel, conv_output)
    # Importância de cada canal = média espacial do gradiente.
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_output = conv_output[0]
    heatmap = conv_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # Só regiões que empurram a predição para a classe alvo (ReLU) e normaliza.
    heatmap = tf.maximum(heatmap, 0)
    max_value = tf.reduce_max(heatmap)
    if max_value > 0:
        heatmap = heatmap / max_value
    return heatmap.numpy()


def overlay_heatmap(
    image: np.ndarray,
    heatmap: np.ndarray,
    *,
    alpha: float = 0.4,
    colormap: str = "jet",
) -> np.ndarray:
    """Sobrepõe o mapa de calor à imagem original.

    Args:
        image: imagem RGB em [0, 1] ou [0, 255], shape (H, W, 3).
        heatmap: mapa de calor 2D em [0, 1] (ver `make_gradcam_heatmap`).
        alpha: peso do mapa de calor na mistura.
        colormap: mapa de cores do matplotlib (padrão "jet").

    Returns:
        Imagem RGB em [0, 1] (float32) com o mapa de calor sobreposto.
    """
    image = np.asarray(image, dtype=np.float32)
    if image.max() > 1.0:
        image = image / 255.0

    # Redimensiona o heatmap ao tamanho da imagem usando o próprio TF (bilinear).
    heatmap_resized = tf.image.resize(
        heatmap[..., np.newaxis], (image.shape[0], image.shape[1])
    ).numpy()[..., 0]

    colored = mpl.colormaps[colormap](heatmap_resized)[..., :3]  # descarta o canal alfa
    overlay = (1 - alpha) * image + alpha * colored
    return np.clip(overlay, 0.0, 1.0).astype(np.float32)
