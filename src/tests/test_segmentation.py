import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from crystalline_highway.core.segmentation import ChineseSegmenter
from crystalline_highway.core.text_utils import normalize_text


def test_normalize_text_strips_punctuation():
    result = normalize_text("你好，世界。")
    print("normalize_text result:", result)
    assert result == "你好世界"


def test_simple_segmenter_keeps_display_punctuation():
    segmenter = ChineseSegmenter(backend="simple")
    sentences = segmenter.split_long_sentences(["你好。再见！"])
    print("split_long_sentences:", [unit.display_text for unit in sentences])
    assert [unit.display_text for unit in sentences] == ["你好。", "再见！"]


def test_simple_morpheme_segmenter():
    segmenter = ChineseSegmenter(backend="simple")
    tokens = segmenter.segment_morphemes("你好世界")
    print("segment_morphemes tokens:", tokens)
    assert tokens == ["你好", "世界"]
