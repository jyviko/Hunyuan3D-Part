import argparse
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
    num_inference_steps,
    guidance_scale,
    octree_resolution,
):
    """Run inference on a single mesh file."""
    pipeline = _PIPELINE
    if pipeline is None:
        raise RuntimeError("Pipeline is not initialized. Call build_demo() first.")
    seed = int(seed)
    bbox_point_num = int(bbox_point_num)
    bbox_prompt_num = int(bbox_prompt_num)
    bbox_threshold = float(bbox_threshold)
    num_inference_steps = int(num_inference_steps)
    guidance_scale = float(guidance_scale)
    octree_resolution = int(octree_resolution)
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


def _build_examples():
    data_dir = Path(__file__).resolve().parents[1] / "data"
    if not data_dir.exists():
        return []
    default_params = [30000, 400, 0.95, True, True, 50, -1.0, 512]
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
    examples = _build_examples()
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
            gr.Number(value=50, label="Inference Steps", precision=0),
            gr.Number(value=-1.0, label="Guidance Scale"),
            gr.Number(value=512, label="Octree Resolution", precision=0),
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
        cache_examples=bool(examples),
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
