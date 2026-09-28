"""Guard the stage boundaries needed for future plate scheduling."""

import ast
from importlib.util import resolve_name
from pathlib import Path
import subprocess
import sys


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
    assert "tracking" not in imports(SRC / "core" / "plate_detector.py")
    assert "plate_buffer" not in imports(SRC / "core" / "ocr.py")
    assert "ocr_stage" not in imports(SRC / "core" / "ocr.py")
    assert "ocr_serialization" not in imports(SRC / "core" / "ocr_fusion.py")
    assert "video" not in imports(SRC / "image" / "plate_ownership.py")


def test_core_has_no_image_or_video_imports():
    for source in (SRC / "core").glob("*.py"):
        text = source.read_text(encoding="utf-8")
        assert "from ..image" not in text and "from ..video" not in text
        assert "from src.image" not in text and "from src.video" not in text


def test_video_does_not_import_image():
    for source in (SRC / "video").glob("*.py"):
        text = source.read_text(encoding="utf-8")
        assert "from ..image" not in text and "from src.image" not in text


def test_image_does_not_import_video():
    for source in (SRC / "image").glob("*.py"):
        text = source.read_text(encoding="utf-8")
        assert "from ..video" not in text and "from src.video" not in text


def test_root_and_config_do_not_eagerly_import_inference_modules():
    code = (
        "import sys, src, src.config; "
        "assert 'onnxruntime' not in sys.modules; "
        "assert 'src.core.ocr' not in sys.modules; "
        "assert 'src.core.vehicle_detector' not in sys.modules; "
        "assert 'src.image.pipeline' not in sys.modules; "
        "assert 'src.video.pipeline' not in sys.modules"
    )
    subprocess.run([sys.executable, "-c", code], cwd=SRC.parent, check=True)


def test_video_ownership_does_not_import_detector_runtime():
    code = (
        "import sys, src.video.plate_ownership; "
        "assert 'onnxruntime' not in sys.modules; "
        "assert 'src.core.plate_detector' not in sys.modules; "
        "assert 'src.video.tracking_runtime' not in sys.modules"
    )
    subprocess.run([sys.executable, "-c", code], cwd=SRC.parent, check=True)


def test_video_ownership_is_not_implemented_in_plate_detection_stage():
    stage = (SRC / "video" / "plate_stage.py").read_text(encoding="utf-8")
    ownership = (SRC / "video" / "plate_ownership.py").read_text(encoding="utf-8")
    assert "def _ownership_score" not in stage
    assert "def resolve_frame_plate_ownership" not in stage
    assert ownership.count("def resolve_frame_plate_ownership") == 1


def test_shared_algorithms_have_one_production_implementation():
    expected = {
        "crop_vehicle_roi": SRC / "core" / "plate_geometry.py",
        "local_bbox_to_global": SRC / "core" / "plate_geometry.py",
        "score_plate_quality": SRC / "core" / "plate_quality.py",
        "resolve_frame_plate_ownership": SRC / "video" / "plate_ownership.py",
        "_ownership_score": SRC / "video" / "plate_ownership.py",
    }
    locations = {name: [] for name in expected}
    for source in SRC.rglob("*.py"):
        tree = ast.parse(source.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name in locations:
                locations[node.name].append(source)
    assert locations == {name: [path] for name, path in expected.items()}


def test_root_package_has_no_eager_imports():
    tree = ast.parse((SRC / "__init__.py").read_text(encoding="utf-8"))
    assert not any(isinstance(node, (ast.Import, ast.ImportFrom)) for node in tree.body)


def test_central_config_does_not_import_mode_specific_modules():
    source = (SRC / "config.py").read_text(encoding="utf-8")
    assert "from .image" not in source and "from .video" not in source


def test_image_and_video_config_names_and_defaults_are_explicit():
    from src.config import PipelineConfig

    config = PipelineConfig()
    assert config.image.quality.sharpness_mode == "normalized"
    assert config.video.quality.sharpness_mode == "original"
    assert config.image.ownership.ownership_conflict_iou_threshold == 0.70
    assert config.video.ownership.conflict_iou_threshold == 0.70
    assert config.video.ownership.overlap_over_smaller_threshold == 0.85
    assert config.video.validation.min_confidence == config.video.tracking.high_confidence == 0.50
    assert config.video.topk.top_k == 4


def test_production_import_graph_has_no_cycle():
    modules = {}
    for path in SRC.rglob("*.py"):
        pieces = path.relative_to(SRC).with_suffix("").parts
        if pieces[-1] == "__init__":
            pieces = pieces[:-1]
        modules[".".join(("src", *pieces))] = path
    graph = {}
    for name, path in modules.items():
        package = name if path.stem == "__init__" else name.rpartition(".")[0]
        dependencies = set()
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.ImportFrom):
                target = (resolve_name("." * node.level + (node.module or ""), package)
                          if node.level else node.module)
                if target in modules and target != name:
                    dependencies.add(target)
                for alias in node.names:
                    member = f"{target}.{alias.name}"
                    if member in modules and member != name:
                        dependencies.add(member)
            elif isinstance(node, ast.Import):
                dependencies.update(alias.name for alias in node.names if alias.name in modules)
        graph[name] = dependencies
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


def test_tests_only_import_production_and_src_has_no_tool_dependency():
    for path in (SRC, SRC.parent / "tests"):
        for source in path.rglob("*.py"):
            for node in ast.walk(ast.parse(source.read_text(encoding="utf-8"))):
                if isinstance(node, ast.ImportFrom):
                    assert not (node.module or "").startswith("tools")
                elif isinstance(node, ast.Import):
                    assert all(not alias.name.startswith("tools") for alias in node.names)
