from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest


def load_table_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "render_latex_tables.py"
    spec = importlib.util.spec_from_file_location("render_latex_tables", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load render_latex_tables.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class RenderLatexTablesTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.tables = load_table_module()

    def test_render_table_preserves_raw_latex_and_escapes_text(self) -> None:
        with TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "table_2_main_validation_results.csv"
            csv_path.write_text(
                "Condition,Hide rate,Note\n"
                "Q27-D,__RAW__\\textbf{93.2\\%},cost_1 & cost_2\n",
                encoding="utf-8",
            )
            rendered = self.tables.render_table(csv_path, self.tables.TABLE_SPECS[csv_path.stem])

        self.assertIn(r"\textbf{93.2\%}", rendered)
        self.assertIn(r"cost\_1 \& cost\_2", rendered)
        self.assertIn(r"\caption{Main validation results.", rendered)

    def test_escape_tex_handles_percent_and_underscore(self) -> None:
        self.assertEqual(self.tables.escape_tex("Q4_K_M 93%"), r"Q4\_K\_M 93\%")


if __name__ == "__main__":
    unittest.main()
