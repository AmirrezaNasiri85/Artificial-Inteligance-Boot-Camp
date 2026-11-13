"""Schema and integrity validation for generated HW01 evidence."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

from PIL import Image

REQUIRED_FIGURES = {
    "causal_mask.png",
    "training_loss.png",
    "error_propagation.png",
}
REQUIRED_JSON_SCHEMAS = {
    "tokenizer_examples.json": {"text", "token_ids", "decoded_without_specials"},
    "training_step_metrics.json": {
        "loss",
        "gradient_norm",
        "supervised_positions",
        "forward_passes",
        "batch_shape",
        "logit_shape",
        "label_shape",
        "parameters_changed",
        "model_training_mode",
    },
    "training_metrics.json": {
        "initial_loss",
        "final_training_loss",
        "validation_loss",
        "training_steps",
        "wall_clock_seconds",
        "parameter_count",
        "fp32_parameter_memory_bytes",
    },
}
REQUIRED_CSV_SCHEMAS = {
    "token_shift_table.csv": {
        "position",
        "input_token_id",
        "input_token_display",
        "target_token_id",
        "target_token_display",
    },
    "generation_trace.csv": {
        "step",
        "input_length",
        "effective_context_length",
        "selected_token_id",
        "selected_token_display",
        "selected_probability",
        "forward_pass_count",
        "stopped_by_eos",
    },
    "error_propagation.csv": {
        "position",
        "phase",
        "reference_token_id",
        "reference_token_display",
        "free_prefix_token_id",
        "free_prefix_token_display",
        "injected_token_id",
        "injected_token_display",
        "teacher_reference_probability",
        "free_reference_probability",
        "total_variation",
    },
    "training_vs_inference.csv": {"dimension", "training", "inference"},
    "eval_vs_inference_mode.csv": {
        "configuration",
        "model.training",
        "torch.is_grad_enabled",
        "output.requires_grad",
        "dropout_active_or_repeat_outputs_equal",
    },
}


def _assert_finite(value: Any, path: str = "root") -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"non-finite numeric value at {path}")
    if isinstance(value, dict):
        for key, item in value.items():
            _assert_finite(item, f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _assert_finite(item, f"{path}[{index}]")


def validate_artifacts(project_root: str | Path) -> dict[str, int]:
    """Parse and validate every required result and PNG under a project root."""
    root = Path(project_root)
    results = root / "results"
    figures = root / "figures"
    summary: dict[str, int] = {}
    parsed_json: dict[str, Any] = {}

    for name, required_keys in REQUIRED_JSON_SCHEMAS.items():
        path = results / name
        value = json.loads(path.read_text(encoding="utf-8"))
        parsed_json[name] = value
        _assert_finite(value)
        if name == "tokenizer_examples.json":
            if not isinstance(value, list) or not value:
                raise ValueError(f"{name} must contain a nonempty list")
            if any(not required_keys.issubset(item) for item in value):
                raise ValueError(f"{name} rows do not match the required schema")
            if any(item["decoded_without_specials"] != item["text"] for item in value):
                raise ValueError("tokenizer round-trip evidence is inconsistent")
            summary[name] = len(value)
        else:
            if not isinstance(value, dict) or not required_keys.issubset(value):
                raise ValueError(f"{name} does not match the required schema")
            summary[name] = len(value)

    step_metrics = parsed_json["training_step_metrics.json"]
    if step_metrics["forward_passes"] != 1.0 or not step_metrics["parameters_changed"]:
        raise ValueError("training-step evidence is logically inconsistent")
    if step_metrics["batch_shape"] != step_metrics["label_shape"]:
        raise ValueError("training inputs and labels must have equal shapes")
    expected_supervision = math.prod(step_metrics["label_shape"])
    if step_metrics["supervised_positions"] != expected_supervision:
        raise ValueError("supervised-position count does not match label shape")
    if step_metrics["logit_shape"][:2] != step_metrics["batch_shape"]:
        raise ValueError("logit batch/sequence shape does not match inputs")

    training_metrics = parsed_json["training_metrics.json"]
    if training_metrics["training_steps"] <= 0 or training_metrics["parameter_count"] <= 0:
        raise ValueError("training metrics contain invalid counts")
    if training_metrics["fp32_parameter_memory_bytes"] != training_metrics["parameter_count"] * 4:
        raise ValueError("FP32 memory does not match parameter count")

    parsed_csv: dict[str, list[dict[str, str]]] = {}
    for name, required_columns in REQUIRED_CSV_SCHEMAS.items():
        path = results / name
        with path.open(encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream)
            rows = list(reader)
            columns = set(reader.fieldnames or [])
        if not rows or not required_columns.issubset(columns):
            raise ValueError(f"{name} is empty or has an invalid schema")
        parsed_csv[name] = rows
        summary[name] = len(rows)

    generation_rows = parsed_csv["generation_trace.csv"]
    for index, row in enumerate(generation_rows):
        if int(row["step"]) != index or int(row["forward_pass_count"]) != index + 1:
            raise ValueError("generation trace forward-pass sequence is inconsistent")
        probability = float(row["selected_probability"])
        if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
            raise ValueError("generation probability is invalid")

    shift_rows = parsed_csv["token_shift_table.csv"]
    for index, row in enumerate(shift_rows):
        if int(row["position"]) != index:
            raise ValueError("token-shift positions must be consecutive")
        if (
            index + 1 < len(shift_rows)
            and row["target_token_id"] != shift_rows[index + 1]["input_token_id"]
        ):
            raise ValueError("token-shift targets must equal the following input token")

    error_rows = parsed_csv["error_propagation.csv"]
    injection_rows = [row for row in error_rows if row["phase"] == "injection"]
    if len(injection_rows) != 1:
        raise ValueError("error propagation must contain exactly one injection row")
    before_or_injection = [row for row in error_rows if row["phase"] != "after"]
    if any(abs(float(row["total_variation"])) > 1e-6 for row in before_or_injection):
        raise ValueError("pre-alteration distributions must be aligned")
    after = [float(row["total_variation"]) for row in error_rows if row["phase"] == "after"]
    if not after or not any(value > 1e-6 for value in after):
        raise ValueError("error propagation must contain post-injection divergence")
    injected = injection_rows[0]
    if injected["injected_token_id"] == injected["reference_token_id"]:
        raise ValueError("injected token must differ from the reference token")
    for index, row in enumerate(error_rows, start=1):
        if int(row["position"]) != index:
            raise ValueError("error-propagation positions must be consecutive")
        for key in [
            "teacher_reference_probability",
            "free_reference_probability",
            "total_variation",
        ]:
            value = float(row[key])
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"invalid error-propagation probability: {key}")

    eval_rows = parsed_csv["eval_vs_inference_mode.csv"]
    if {row["configuration"] for row in eval_rows} != {
        "train + grad",
        "eval + grad",
        "train + inference_mode",
        "eval + inference_mode",
    }:
        raise ValueError("eval/inference experiment configurations are incomplete")
    eval_by_name = {row["configuration"]: row for row in eval_rows}
    expected_eval = {
        "train + grad": ("True", "True", "True", "False"),
        "eval + grad": ("False", "True", "True", "True"),
        "train + inference_mode": ("True", "False", "False", "False"),
        "eval + inference_mode": ("False", "False", "False", "True"),
    }
    for configuration, expected in expected_eval.items():
        row = eval_by_name[configuration]
        observed = (
            row["model.training"],
            row["torch.is_grad_enabled"],
            row["output.requires_grad"],
            row["dropout_active_or_repeat_outputs_equal"],
        )
        if observed != expected:
            raise ValueError(f"eval/inference evidence is inconsistent: {configuration}")

    for name in REQUIRED_FIGURES:
        path = figures / name
        if path.stat().st_size <= 0 or path.stat().st_size >= 500_000:
            raise ValueError(f"invalid figure size: {name}")
        with Image.open(path) as image:
            if image.format != "PNG" or image.width <= 0 or image.height <= 0:
                raise ValueError(f"invalid PNG figure: {name}")
            image.verify()
        summary[name] = path.stat().st_size
    return summary
