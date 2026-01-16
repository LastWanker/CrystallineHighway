import importlib.util
import math
from pathlib import Path

from src.crystalline_highway.core.word_vectors import WordVectorProvider


def _assert_vector_close(actual, expected, tol=1e-6):
    assert len(actual) == len(expected)
    for left, right in zip(actual, expected):
        assert math.isclose(left, right, rel_tol=tol, abs_tol=tol)


def test_text_vectors_mean(tmp_path: Path):
    text_path = tmp_path / "vectors.txt"
    text_path.write_text("2 3\n你好 0.1 0.2 0.3\n世界 0.4 0.5 0.6\n", encoding="utf-8")
    provider = WordVectorProvider(dim=3, path=str(text_path), lazy=True)
    vec = provider.get_tokens_vector(["你好", "世界"])
    _assert_vector_close(vec, [0.25, 0.35, 0.45])


def test_binary_index_build(tmp_path: Path):
    if importlib.util.find_spec("gensim") is None:
        return
    from gensim.models import KeyedVectors

    vectors = KeyedVectors(vector_size=2)
    vectors.add_vectors(["你好", "世界"], [[0.1, 0.2], [0.3, 0.4]])
    binary_path = tmp_path / "vectors.bin"
    vectors.save_word2vec_format(str(binary_path), binary=True)

    index_path = tmp_path / "vectors.kv"
    provider = WordVectorProvider(
        dim=2,
        path=str(binary_path),
        index_path=str(index_path),
        auto_build_index=True,
    )
    _assert_vector_close(provider.get_vector("你好"), [0.1, 0.2])
    assert index_path.exists()
