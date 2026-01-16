from src.crystalline_highway.core.segmentation import ChineseSegmenter
from src.crystalline_highway.core.text_utils import normalize_text


def test_normalize_text_strips_punctuation():
    assert normalize_text("你好，世界。") == "你好世界"


def test_simple_segmenter_keeps_display_punctuation():
    segmenter = ChineseSegmenter(backend="simple")
    sentences = segmenter.split_long_sentences(["你好。再见！"])
    assert [unit.display_text for unit in sentences] == ["你好。", "再见！"]


def test_simple_morpheme_segmenter():
    segmenter = ChineseSegmenter(backend="simple")
    tokens = segmenter.segment_morphemes("你好世界")
    assert tokens == ["你", "好", "世", "界"]
