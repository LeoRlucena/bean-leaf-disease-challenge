from __future__ import annotations

from collections.abc import Iterable

import pandas as pd
import tensorflow as tf
from datasets import Dataset, DatasetDict, load_dataset

DATASET_ID = "AI-Lab-Makerere/beans"
DEFAULT_IMAGE_SIZE = (224, 224)

def load_beans_dataset(dataset_id: str = DATASET_ID) -> DatasetDict:
    """Carrega o dataset Beans diretamente do Hugging Face."""
    return load_dataset(dataset_id)

def class_names(dataset: DatasetDict) -> list[str]:
    """Retorna os nomes das classes conforme o schema do dataset."""
    return dataset["train"].features["labels"].names

def split_summary(dataset: DatasetDict) -> pd.DataFrame:
    """Resumo simples com quantidade de amostras por split."""
    names = class_names(dataset)
    rows: list[dict[str, object]] = []
    for split, ds in dataset.items():
        counts = pd.Series(ds["labels"]).value_counts().sort_index()
        total = len(ds)
        for label_id, label_name in enumerate(names):
            count = int(counts.get(label_id, 0))
            rows.append(
                {
                    "split": split,
                    "label_id": label_id,
                    "class_name": label_name,
                    "count": count,
                    "proportion": count / total if total > 0 else 0.0,
                }
            )
    return pd.DataFrame(rows)

def _generator(ds: Dataset) -> Iterable[tuple[object, int]]:
    for sample in ds:
        yield sample["image"], sample["labels"]

def make_tf_dataset(
        ds: Dataset,
        image_size: tuple[int, int] = DEFAULT_IMAGE_SIZE,
        batch_size: int = 32,
        shuffle: bool = True,
        seed: int = 42,
) -> tf.data.Dataset:
    """Converte um split Hugging Face em tf.data.Dataset com resize e pixels em [0, 1]."""
    output_signature = (
        tf.TensorSpec(shape=(None, None, 3), dtype=tf.uint8),
        tf.TensorSpec(shape=(), dtype=tf.int32)
    )

    def gen():
        for image, label in _generator(ds):
            yield tf.convert_to_tensor(image.convert("RGB")), label

    tf_ds = tf.Data.Dataset.from_generator(gen, output_signature=output_signature)

    def preprocess(image: tf.Tensor, label: tf.Tensor):
        image = tf.image.resize(image, image_size)
        image = tf.cast(image, tf.float32) / 255.0
        return image, label

    if shuffle:
        tf_ds = tf_ds.shuffle(buffer_size=len(ds), seed=seed, reshuffle_each_iteration=True)

    return (
        tf_ds.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )
