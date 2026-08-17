import sys
from pathlib import Path

# Make src/ importable from tests/ without needing an installed package
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))