import pytest
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from app.enums.translit_engine import TransliterateEngine

TransliterationEngine = getattr(TransliterateEngine, os.getenv("TRANSLITERATE_ENGINE", "TRANSLIT_ME_CLOUD")) or ModuleNotFoundError(f"Not Found TRANSLITERATE_ENGINE module named: {os.getenv('TRANSLITERATE_ENGINE')}")

def test_hebrew_to_arabic():
    names = ["תל אביב"]
    expected = ['تل ابيب']
    translit_engines = [TransliterateEngine.TRANSLIT_ME_CLOUD, TransliterateEngine.TRANSLIT_ME_LOCAL, TransliterateEngine.GIMELTRA]
    for engine in translit_engines:
        ts = engine()
        for i in range(len(names)):
            assert ts.transliterate(names[i]) == expected[i], f"Engine {engine} failed to transliterate {names[i]}"


if __name__ == "__main__":
    import subprocess
    retcode = subprocess.call(['pytest', '--tb=short', str(__file__)])
    sys.exit(retcode)