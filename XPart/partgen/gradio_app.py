import argparse
import os
from pathlib import Path

import gradio as gr
import pytorch_lightning as pl
import torch

from partgen.partformer_pipeline import PartFormerPipeline
from partgen.utils.misc import get_config_from_file

_PIPELINE = None


def _load_pipeline(config_path, device):
    """Load the PartFormer pipeline from checkpoint and config."""
    pl.seed_everything(2026, workers=True)
    cfg_path = (
        Path(config_path)
        if config_path
        else Path(__file__).resolve().parent / "config" / "infer.yaml"
    )
    config = get_config_from_file(str(cfg_path))
    assert hasattr(config, "ckpt") or hasattr(
        config, "ckpt_path"
    ), "ckpt or ckpt_path must be specified in config"
    pipeline = PartFormerPipeline.from_pretrained(
        model_path="tencent/Hunyuan3D-Part",
        verbose=True,
    )

    device = device if device in ["cuda", "cpu"] else "cuda"
    pipeline.to(device=device, dtype=torch.float32)
    return pipeline


def _ensure_pipeline(config_path, device):
    global _PIPELINE
    if _PIPELINE is None:
        _PIPELINE = _load_pipeline(config_path, device)
    return _PIPELINE


def run_infer(
    mesh_file_name,
    seed,
    bbox_point_num,
    bbox_prompt_num,
    bbox_threshold,
    bbox_post_process,
    bbox_clean_mesh_flag,
    obj_pc_size,
    part_pc_size,
    cond_batch_size,
    num_inference_steps,
    guidance_scale,
    octree_resolution,
    add_assembly_pins,
    pin_diameter,
    pin_length,
    pin_clearance,
    pin_area_per_pin,
    pin_interface_distance,
    pin_edge_distance,
    pin_sample_count,
    pin_boolean_engine,
):
    """Run inference on a single mesh file."""
    pipeline = _PIPELINE
    if pipeline is None:
        raise RuntimeError("Pipeline is not initialized. Call build_demo() first.")
    seed = int(seed)
    bbox_point_num = int(bbox_point_num)
    bbox_prompt_num = int(bbox_prompt_num)
    bbox_threshold = float(bbox_threshold)
    obj_pc_size = int(obj_pc_size)
    part_pc_size = int(part_pc_size)
    cond_batch_size = int(cond_batch_size)
    num_inference_steps = int(num_inference_steps)
    guidance_scale = float(guidance_scale)
    octree_resolution = int(octree_resolution)
    add_assembly_pins = bool(add_assembly_pins)
    pin_diameter = float(pin_diameter)
    pin_length = float(pin_length)
    pin_clearance = float(pin_clearance)
    pin_area_per_pin = float(pin_area_per_pin)
    pin_interface_distance = float(pin_interface_distance)
    pin_edge_distance = float(pin_edge_distance)
    pin_sample_count = int(pin_sample_count)
    pin_boolean_engine = str(pin_boolean_engine).strip()
    if pin_boolean_engine == "":
        pin_boolean_engine = None
    print(f"Running inference on {mesh_file_name} with seed {seed}")
    # Ensure deterministic behavior per request
    try:
        pl.seed_everything(int(seed), workers=True)
    except Exception:
        pl.seed_everything(2026, workers=True)
    additional_params = {"output_type": "trimesh"}
    obj_mesh, (out_bbox, mesh_gt_bbox, explode_object) = pipeline(
        mesh_path=mesh_file_name,
        octree_resolution=octree_resolution,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        bbox_point_num=bbox_point_num,
        bbox_prompt_num=bbox_prompt_num,
        bbox_threshold=bbox_threshold,
        bbox_post_process=bbox_post_process,
        bbox_clean_mesh_flag=bbox_clean_mesh_flag,
        obj_pc_size=obj_pc_size,
        part_pc_size=part_pc_size,
        cond_batch_size=cond_batch_size,
        add_assembly_pins=add_assembly_pins,
        pin_diameter=pin_diameter,
        pin_length=pin_length,
        pin_clearance=pin_clearance,
        pin_area_per_pin=pin_area_per_pin,
        pin_interface_distance=pin_interface_distance,
        pin_edge_distance=pin_edge_distance,
        pin_sample_count=pin_sample_count,
        pin_boolean_engine=pin_boolean_engine,
        **additional_params,
    )
    obj_path = "tmp_obj.glb"
    out_bbox_path = "tmp_out_bbox.glb"
    gt_bbox_path = "tmp_gt_bbox.glb"
    explode_path = "tmp_explode.glb"
    obj_mesh.export(obj_path)
    out_bbox.export(out_bbox_path)
    mesh_gt_bbox.export(gt_bbox_path)
    explode_object.export(explode_path)
    return obj_path, out_bbox_path, gt_bbox_path, explode_path


def _build_examples(enable_examples: bool):
    if not enable_examples:
        return []
    data_dir = Path(__file__).resolve().parents[1] / "data"
    if not data_dir.exists():
        return []
    default_params = [
        30000,
        400,
        0.95,
        True,
        True,
        10240,
        10240,
        4,
        50,
        -1.0,
        512,
        False,
        3.0,
        6.0,
        0.2,
        2000.0,
        0.5,
        6.0,
        20000,
        "",
    ]
    candidates = [
        ("000.glb", 42),
        ("001.glb", 42),
        ("002.glb", 42),
        ("003.glb", 42),
        ("004.glb", 2025),
    ]
    examples = []
    for filename, seed in candidates:
        path = data_dir / filename
        if not path.exists():
            continue
        examples.append([str(path), seed, *default_params])
    return examples


def build_demo(config_path=None, device="cuda"):
    """Create the Gradio demo interface."""
    _ensure_pipeline(config_path, device)
    enable_examples = os.getenv("XPART_ENABLE_EXAMPLES", "").strip().lower() in {
        "1",
        "true",
        "yes",
    }
    examples = _build_examples(enable_examples)
    if not examples:
        examples = None
    return gr.Interface(
        description="""
# XPart: PartFormer Inference Demo

Upload a mesh to run XPart's PartFormer pipeline. The demo returns:
- Predicted object mesh
- Predicted bbox
- Input bbox
- Exploded object
""",
        fn=run_infer,
        inputs=[
            gr.Model3D(clear_color=[0.0, 0.0, 0.0, 0.0], label="Input Mesh"),
            gr.Number(value=42, label="Random Seed"),
            gr.Number(value=30000, label="BBox Point Count", precision=0),
            gr.Number(value=400, label="BBox Prompt Count", precision=0),
            gr.Number(value=0.95, label="BBox Threshold"),
            gr.Checkbox(value=True, label="BBox Post-process"),
            gr.Checkbox(value=True, label="BBox Clean Mesh"),
            gr.Number(value=10240, label="Object Point Count", precision=0),
            gr.Number(value=10240, label="Part Point Count", precision=0),
            gr.Number(value=4, label="Conditioning Batch Size", precision=0),
            gr.Number(value=50, label="Inference Steps", precision=0),
            gr.Number(value=-1.0, label="Guidance Scale"),
            gr.Number(value=512, label="Octree Resolution", precision=0),
            gr.Checkbox(value=False, label="Add Assembly Pins"),
            gr.Number(value=3.0, label="Pin Diameter (mm)"),
            gr.Number(value=6.0, label="Pin Length (mm)"),
            gr.Number(value=0.2, label="Pin Clearance (mm)"),
            gr.Number(value=2000.0, label="Pin Area per Pin (mm²)"),
            gr.Number(value=0.5, label="Pin Interface Distance (mm)"),
            gr.Number(value=6.0, label="Pin Edge Distance (mm)"),
            gr.Number(value=20000, label="Pin Sample Count", precision=0),
            gr.Textbox(value="", label="Pin Boolean Engine"),
        ],
        outputs=[
            gr.Model3D(
                clear_color=[0.0, 0.0, 0.0, 0.0], label="Predicted Object Mesh"
            ),
            gr.Model3D(clear_color=[0.0, 0.0, 0.0, 0.0], label="Predicted BBox"),
            gr.Model3D(clear_color=[0.0, 0.0, 0.0, 0.0], label="Input BBox"),
            gr.Model3D(clear_color=[0.0, 0.0, 0.0, 0.0], label="Exploded Object"),
        ],
        examples=examples,
        cache_examples=False,
        examples_per_page=8,
        flagging_mode="never",
    )


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help="Path to infer.yaml")
    parser.add_argument("--device", type=str, default="cuda", help="cuda or cpu")
    parser.add_argument("--server-name", type=str, default="0.0.0.0")
    parser.add_argument("--server-port", type=int, default=8080)
    args = parser.parse_args(argv)

    demo = build_demo(config_path=args.config, device=args.device)
    demo.launch(server_name=args.server_name, server_port=args.server_port)


if __name__ == "__main__":
    main()
