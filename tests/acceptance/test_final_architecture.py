# The executable acceptance suite lives in ../final_test.py so students can
# run `python final_test.py` without model downloads or API credentials.

def test_final_test_file_exists():
    from pathlib import Path
    assert (Path(__file__).parents[2] / "final_test.py").exists()
