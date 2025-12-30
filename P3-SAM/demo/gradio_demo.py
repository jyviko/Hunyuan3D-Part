import argparse
import os

import gradio as gr
import numpy as np
import trimesh

from p3sam.demo.auto_mask import AutoMask
from p3sam.demo.auto_mask_no_postprocess import AutoMask as AutoMaskNoPostProcess


def build_demo(ckpt_path=None):
    automask = AutoMask(ckpt_path)
    automask_no_postprocess = AutoMaskNoPostProcess(
        ckpt_path, automask_instance=automask
    )

    def load_mesh(
        mesh_file_name,
        post_process,
        seed,
        point_num,
        prompt_num,
        threshold,
        prompt_bs,
        clean_mesh_flag,
    ):
        seed = int(seed)
        point_num = int(point_num)
        prompt_num = int(prompt_num)
        prompt_bs = int(prompt_bs)
        threshold = float(threshold)
        np.random.seed(seed)
        mesh = trimesh.load(mesh_file_name, force="mesh", process=False)
        if post_process:
            _, face_ids, mesh = automask.predict_aabb(
                mesh,
                seed=seed,
                is_parallel=False,
                post_process=True,
                point_num=point_num,
                prompt_num=prompt_num,
                threshold=threshold,
                prompt_bs=prompt_bs,
                clean_mesh_flag=clean_mesh_flag,
                show_info=False,
            )
        else:
            _, face_ids, mesh = automask_no_postprocess.predict_aabb(
                mesh,
                seed=seed,
                is_parallel=False,
                post_process=False,
                point_num=point_num,
                prompt_num=prompt_num,
                threshold=threshold,
                prompt_bs=prompt_bs,
                clean_mesh_flag=clean_mesh_flag,
                show_info=False,
            )
        color_map = {}
        unique_ids = np.unique(face_ids)
        for i in unique_ids:
            if i == -1:
                continue
            part_color = np.random.rand(3) * 255
            color_map[i] = part_color
        face_colors = []
        for i in face_ids:
            if i == -1:
                face_colors.append([0, 0, 0])
            else:
                face_colors.append(color_map[i])
        face_colors = np.array(face_colors).astype(np.uint8)
        mesh_save = mesh.copy()
        mesh_save.visual.face_colors = face_colors

        file_path = "segment_result.glb"
        mesh_save.export(file_path)
        return file_path

    enable_examples = os.getenv("P3SAM_ENABLE_EXAMPLES", "").strip().lower() in {
        "1",
        "true",
        "yes",
    }
    examples = []
    if enable_examples:
        examples = [
            [
                os.path.join(os.path.dirname(__file__), "assets/1.glb"),
                True,
                42,
                30000,
                400,
                0.95,
                32,
                True,
            ],
            [
                os.path.join(os.path.dirname(__file__), "assets/2.glb"),
                True,
                42,
                30000,
                400,
                0.95,
                32,
                True,
            ],
            [
                os.path.join(os.path.dirname(__file__), "assets/3.glb"),
                True,
                42,
                30000,
                400,
                0.95,
                32,
                True,
            ],
            [
                os.path.join(os.path.dirname(__file__), "assets/4.glb"),
                True,
                42,
                30000,
                400,
                0.95,
                32,
                True,
            ],
        ]
    if not examples:
        examples = None

    return gr.Interface(
        description="""
## P3-SAM: Native 3D Part Segmentation

[Paper](https://arxiv.org/abs/2509.06784) | [Project Page](https://murcherful.github.io/P3-SAM/) | [Code](https://github.com/Tencent-Hunyuan/Hunyuan3D-Part/P3-SAM/) | [Model](https://huggingface.co/tencent/Hunyuan3D-Part)

This is a demo of P3-SAM, a native 3D part segmentation method that can segment a mesh into different parts.
Input a mesh and push the "submit" button to get the segmentation results.
""",
        fn=load_mesh,
        inputs=[
            gr.Model3D(clear_color=[0.0, 0.0, 0.0, 0.0], label="Input Mesh"),
            gr.Checkbox(value=True, label="Post-process"),
            gr.Number(value=42, label="Random Seed"),
            gr.Number(value=30000, label="Point Count", precision=0),
            gr.Number(value=400, label="Prompt Count", precision=0),
            gr.Number(value=0.95, label="Post-process Threshold"),
            gr.Number(value=32, label="Prompt Batch Size", precision=0),
            gr.Checkbox(value=True, label="Clean Mesh"),
        ],
        outputs=gr.Model3D(
            clear_color=[0.0, 0.0, 0.0, 0.0], label="Segmentation Results"
        ),
        examples=examples,
        cache_examples=False,
        flagging_mode="never",
    )


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_path", type=str, default=None, help="Model path")
    parser.add_argument("--server-name", type=str, default="0.0.0.0")
    parser.add_argument("--server-port", type=int, default=8080)
    args = parser.parse_args(argv)

    demo = build_demo(ckpt_path=args.ckpt_path)
    demo.launch(server_name=args.server_name, server_port=args.server_port)


if __name__ == "__main__":
    main()
