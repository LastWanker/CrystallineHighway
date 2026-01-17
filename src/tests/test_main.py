import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from crystalline_highway.core.segmentation import ChineseSegmenter
from crystalline_highway.core.text_utils import normalize_text
from crystalline_highway.core.word_vectors import WordVectorProvider


def run_demo(tmp_path: Path):
    text = "你好，世界！\n再见。"
    segmenter = ChineseSegmenter(backend="simple")

    paragraphs = segmenter.split_paragraphs(text)
    print("paragraphs:", [para.display_text for para in paragraphs])
    assert [para.display_text for para in paragraphs] == ["你好，世界！", "再见。"]
    assert [para.normalized_text for para in paragraphs] == ["你好世界", "再见"]

    long_sentences = segmenter.split_long_sentences([para.display_text for para in paragraphs])
    print("long_sentences:", [unit.display_text for unit in long_sentences])
    assert [unit.display_text for unit in long_sentences] == ["你好，世界！", "再见。"]

    short_sentences = segmenter.split_short_sentences([unit.display_text for unit in long_sentences])
    print("short_sentences:", [unit.display_text for unit in short_sentences])
    assert [unit.display_text for unit in short_sentences] == ["你好，", "世界！", "再见。"]

    normalized = normalize_text("你好，世界！")
    print("normalized:", normalized)
    assert normalized == "你好世界"

    vector_path = tmp_path / "vectors.txt"
    vector_path.write_text("4 2\n你 1 0\n好 1 0\n世 0 1\n界 0 1\n", encoding="utf-8")

    provider = WordVectorProvider(dim=2, path=str(vector_path), lazy=True)
    vector = provider.get_text_vector("你好世界", segmenter=segmenter)

    assert vector == [0.5, 0.5]
    print("test_main 聚合向量:", vector)


if __name__ == "__main__":
    temp_root = Path.cwd() / ".tmp_test_main"
    temp_root.mkdir(parents=True, exist_ok=True)
    run_demo(temp_root)
