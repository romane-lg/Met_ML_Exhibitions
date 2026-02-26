from src.features.nlp_utils import tokenize_text


def test_tokenize_text_number_toggle() -> None:
    text = "ancient 1234 art"
    out_no_numbers = tokenize_text(text, keep_numbers=False, min_len=1)
    out_with_numbers = tokenize_text(text, keep_numbers=True, min_len=1)
    assert "1234" not in out_no_numbers
    assert "1234" in out_with_numbers
