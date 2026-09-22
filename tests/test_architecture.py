"""Guard the stage boundaries needed for future plate scheduling."""

import ast
from pathlib import Path


SRC = Path(__file__).resolve().parents[1] / "src"


def imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    result = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            if node.level and node.module:
                result.add(node.module.split(".")[0])
            elif node.module and node.module.startswith("src."):
                result.add(node.module.split(".")[1])
    return result


def test_model_and_algorithm_modules_do_not_import_orchestration():
    assert "tracking" not in imports(SRC / "plate_detector.py")
    assert "plate_buffer" not in imports(SRC / "microcharnet_ocr.py")
    assert "ocr_stage" not in imports(SRC / "microcharnet_ocr.py")
    assert "ocr_serialization" not in imports(SRC / "ocr_fusion.py")
    assert "plate_ownership" not in imports(SRC / "plate_ownership_temporal.py")


def test_production_import_graph_has_no_cycle():
    modules = {path.stem: imports(path) for path in SRC.glob("*.py")}
    graph = {name: deps & modules.keys() for name, deps in modules.items()}
    visited = set()
    visiting = set()

    def walk(name: str):
        assert name not in visiting, f"Circular production import through {name}"
        if name in visited:
            return
        visiting.add(name)
        for dependency in graph[name]:
            walk(dependency)
        visiting.remove(name)
        visited.add(name)

    for module in graph:
        walk(module)
