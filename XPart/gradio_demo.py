import gradio as gr
import os
import argparse
from pathlib import Path

import torch
import pytorch_lightning as pl

from partgen.partformer_pipeline import PartFormerPipeline
from partgen.utils.misc import get_config_from_file

argparser = argparse.ArgumentParser()
argparser.add_argument("--config", type=str, help="Path to infer.yaml")
argparser.add_argument("--device", type=str, default="cuda", help="cuda or cpu")
args, _ = argparser.parse_known_args()


def _load_pipeline():
    """Load the PartFormer pipeline from checkpoint and config.

    Returns:
        PartFormerPipeline: The loaded pipeline.
    """
    pl.seed_everything(2026, workers=True)
    cfg_path = (
        args.config
        if args.config
        else str(Path(__file__).parent / "partgen/config" / "infer.yaml")
    )
    config = get_config_from_file(cfg_path)
    assert hasattr(config, "ckpt") or hasattr(
        config, "ckpt_path"
    ), "ckpt or ckpt_path must be specified in config"
    pipeline = PartFormerPipeline.from_pretrained(
        model_path="tencent/Hunyuan3D-Part",
        verbose=True,
    )

    device = args.device if args.device in ["cuda", "cpu"] else "cuda"
    pipeline.to(device=device, dtype=torch.float32)
    return pipeline


_PIPELINE = _load_pipeline()


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
    """Run inference on a single mesh file.
    Args:
        mesh_file_name (str): Path to the input mesh file.
        seed (int): Random seed for deterministic behavior.
        bbox_point_num (int): Point count for bbox prediction.
        bbox_prompt_num (int): Prompt count for bbox prediction.
        bbox_threshold (float): Post-process threshold for bbox prediction.
        bbox_post_process (bool): Whether to run bbox post-processing.
        bbox_clean_mesh_flag (bool): Whether to clean the mesh for bbox prediction.
        num_inference_steps (int): Diffusion sampling steps.
        guidance_scale (float): Guidance scale for classifier-free guidance.
        octree_resolution (int): Octree resolution for mesh extraction.
    Returns:
        Tuple[str, str, str, str]: Paths to the output meshes (obj_mesh, out_bbox, mesh_gt_bbox, explode_object).
    """
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
    uid = Path(mesh_file_name).stem
    additional_params = {"output_type": "trimesh"}
    obj_mesh, (out_bbox, mesh_gt_bbox, explode_object) = _PIPELINE(
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
    # Export all results to temporary files for Gradio Model3D
    # _, obj_path = mkstemp(suffix=".glb")
    # _, out_bbox_path = mkstemp(suffix=".glb")
    # _, gt_bbox_path = mkstemp(suffix=".glb")
    # _, explode_path = mkstemp(suffix=".glb")
    obj_path = f"tmp_obj.glb"
    out_bbox_path = f"tmp_out_bbox.glb"
    gt_bbox_path = f"tmp_gt_bbox.glb"
    explode_path = f"tmp_explode.glb"
    obj_mesh.export(obj_path)
    out_bbox.export(out_bbox_path)
    mesh_gt_bbox.export(gt_bbox_path)
    explode_object.export(explode_path)
    return obj_path, out_bbox_path, gt_bbox_path, explode_path


demo = gr.Interface(
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
        gr.Number(value=100000, label="BBox Point Count", precision=0),
        gr.Number(value=400, label="BBox Prompt Count", precision=0),
        gr.Number(value=0.95, label="BBox Threshold"),
        gr.Checkbox(value=True, label="BBox Post-process"),
        gr.Checkbox(value=True, label="BBox Clean Mesh"),
        gr.Number(value=50, label="Inference Steps", precision=0),
        gr.Number(value=-1.0, label="Guidance Scale"),
        gr.Number(value=512, label="Octree Resolution", precision=0),
    ],
    outputs=[
        gr.Model3D(clear_color=[0.0, 0.0, 0.0, 0.0], label="Predicted Object Mesh"),
        gr.Model3D(clear_color=[0.0, 0.0, 0.0, 0.0], label="Predicted BBox"),
        gr.Model3D(clear_color=[0.0, 0.0, 0.0, 0.0], label="Input BBox"),
        gr.Model3D(clear_color=[0.0, 0.0, 0.0, 0.0], label="Exploded Object"),
    ],
    examples=[
        [
            os.path.join(os.path.dirname(__file__), "data/000.glb"),
            42,
            100000,
            400,
            0.95,
            True,
            True,
            50,
            -1.0,
            512,
        ],
        [
            os.path.join(os.path.dirname(__file__), "data/001.glb"),
            42,
            100000,
            400,
            0.95,
            True,
            True,
            50,
            -1.0,
            512,
        ],
        [
            os.path.join(os.path.dirname(__file__), "data/002.glb"),
            42,
            100000,
            400,
            0.95,
            True,
            True,
            50,
            -1.0,
            512,
        ],
        [
            os.path.join(os.path.dirname(__file__), "data/003.glb"),
            42,
            100000,
            400,
            0.95,
            True,
            True,
            50,
            -1.0,
            512,
        ],
        [
            os.path.join(os.path.dirname(__file__), "data/004.glb"),
            2025,
            100000,
            400,
            0.95,
            True,
            True,
            50,
            -1.0,
            512,
        ],
    ],
    cache_examples=True,
    examples_per_page=8,
    flagging_mode="never",
)


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=8080)
