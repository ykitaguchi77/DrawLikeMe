"""
DrawLikeMe - Style Transfer App

Upload your illustrations to define your style, then transform any image
into your unique artistic style while preserving its structure.

Usage:
    python app.py                    # Launch Gradio UI
    python app.py --share            # Launch with public share link
    python app.py --server-port 8080 # Custom port
"""

import argparse
import traceback

import gradio as gr
import torch
from PIL import Image

from config import REPLICATE_API_TOKEN, DEVICE
from pipelines.preprocessors import extract_canny, extract_lineart
from utils.image_utils import prepare_image


# ---------------------------------------------------------------------------
# Device detection
# ---------------------------------------------------------------------------
def get_device() -> str:
    if DEVICE != "auto":
        return DEVICE
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


CURRENT_DEVICE = get_device()
HAS_GPU = CURRENT_DEVICE in ("cuda", "mps")
HAS_REPLICATE = bool(REPLICATE_API_TOKEN)

# Cache loaded pipelines to avoid reloading
_pipeline_cache: dict = {}


# ---------------------------------------------------------------------------
# Pipeline factory
# ---------------------------------------------------------------------------
def get_available_methods() -> list[str]:
    methods = []
    if HAS_GPU:
        methods.append("ControlNet + IP-Adapter (Local GPU)")
        methods.append("InstantStyle (Local GPU)")
    if HAS_REPLICATE:
        methods.append("Replicate API (Cloud)")
    if not methods:
        methods.append("Preview Only (CPU - preprocessors only)")
    return methods


def get_pipeline(method: str):
    if method in _pipeline_cache:
        return _pipeline_cache[method]

    pipe = None
    if method == "ControlNet + IP-Adapter (Local GPU)":
        from pipelines.controlnet_ipadapter import ControlNetIPAdapterPipeline
        pipe = ControlNetIPAdapterPipeline(device=CURRENT_DEVICE)
    elif method == "InstantStyle (Local GPU)":
        from pipelines.instantstyle import InstantStylePipeline
        pipe = InstantStylePipeline(device=CURRENT_DEVICE)
    elif method == "Replicate API (Cloud)":
        from pipelines.replicate_api import ReplicateStyleTransferPipeline
        pipe = ReplicateStyleTransferPipeline()

    if pipe is not None:
        _pipeline_cache[method] = pipe
    return pipe


# ---------------------------------------------------------------------------
# Core processing functions
# ---------------------------------------------------------------------------
def preview_preprocessing(content_image: Image.Image):
    """Show Canny and Lineart extraction results for the content image."""
    if content_image is None:
        return None, None
    content = prepare_image(content_image)
    canny = extract_canny(content)
    lineart = extract_lineart(content)
    return canny, lineart


def run_style_transfer(
    content_image: Image.Image,
    style_image: Image.Image,
    method: str,
    controlnet_scale: float,
    style_strength: float,
    num_steps: int,
    guidance_scale: float,
    prompt: str,
    negative_prompt: str,
    progress=gr.Progress(),
):
    """Run style transfer with the selected method."""
    if content_image is None or style_image is None:
        raise gr.Error("コンテンツ画像とスタイル参照画像の両方をアップロードしてください。")

    # Preview-only mode (no GPU, no API)
    if method == "Preview Only (CPU - preprocessors only)":
        content = prepare_image(content_image)
        canny = extract_canny(content)
        lineart = extract_lineart(content)
        gr.Info(
            "GPU/APIが利用できないため、前処理結果のみ表示しています。"
            "GPUマシンで実行するか、REPLICATE_API_TOKEN を設定してください。"
        )
        return canny, lineart, "Preview only - no style transfer performed"

    pipeline = get_pipeline(method)
    if pipeline is None:
        raise gr.Error(f"パイプライン '{method}' を初期化できませんでした。")

    progress(0.1, desc="画像を前処理中...")
    progress(0.3, desc=f"{method} でスタイル変換中...")

    try:
        result = pipeline.transfer_style(
            content_image=content_image,
            style_image=style_image,
            controlnet_scale=controlnet_scale,
            style_strength=style_strength,
            num_steps=int(num_steps),
            guidance_scale=guidance_scale,
            prompt=prompt,
            negative_prompt=negative_prompt,
        )
    except Exception as e:
        raise gr.Error(f"スタイル変換中にエラーが発生しました: {e}\n{traceback.format_exc()}")

    progress(0.9, desc="完了！")

    # Get preprocessing image if available
    preprocess_img = None
    if result.preprocessing_images:
        preprocess_img = list(result.preprocessing_images.values())[0]

    param_str = "\n".join(f"  {k}: {v}" for k, v in result.parameters.items())
    info = f"Method: {result.method_name}\nParameters:\n{param_str}"

    return result.output_image, preprocess_img, info


def run_comparison(
    content_image: Image.Image,
    style_image: Image.Image,
    controlnet_scale: float,
    style_strength: float,
    num_steps: int,
    guidance_scale: float,
    prompt: str,
    negative_prompt: str,
    progress=gr.Progress(),
):
    """Run all available methods and return results side-by-side."""
    if content_image is None or style_image is None:
        raise gr.Error("コンテンツ画像とスタイル参照画像の両方をアップロードしてください。")

    methods = get_available_methods()
    results = []

    for i, method in enumerate(methods):
        if method == "Preview Only (CPU - preprocessors only)":
            continue
        progress(i / len(methods), desc=f"{method} で処理中...")
        pipeline = get_pipeline(method)
        if pipeline is None:
            continue
        try:
            result = pipeline.transfer_style(
                content_image=content_image,
                style_image=style_image,
                controlnet_scale=controlnet_scale,
                style_strength=style_strength,
                num_steps=int(num_steps),
                guidance_scale=guidance_scale,
                prompt=prompt,
                negative_prompt=negative_prompt,
            )
            results.append((result.output_image, result.method_name))
        except Exception as e:
            results.append((None, f"{method} (error: {e})"))

    return results


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------
def build_ui():
    available_methods = get_available_methods()

    with gr.Blocks(
        title="DrawLikeMe",
        theme=gr.themes.Soft(),
        css="""
        .header { text-align: center; margin-bottom: 1em; }
        .method-info { padding: 10px; background: #f0f0f0; border-radius: 8px; }
        """,
    ) as app:
        gr.Markdown(
            """
            # DrawLikeMe
            ### あなたのイラストスタイルで画像を描き変えるアプリ

            1. **スタイル参照画像**: あなたのイラストをアップロード（スタイルの元になる画像）
            2. **コンテンツ画像**: 変換したい画像をアップロード（構造を維持する画像）
            3. **手法を選択**して「スタイル変換実行」ボタンをクリック
            """,
            elem_classes=["header"],
        )

        # Environment info
        env_parts = []
        env_parts.append(f"Device: **{CURRENT_DEVICE}**")
        if HAS_GPU:
            if CURRENT_DEVICE == "cuda":
                env_parts.append(f"GPU: **{torch.cuda.get_device_name(0)}**")
            env_parts.append("Local pipelines: **available**")
        else:
            env_parts.append("Local pipelines: **unavailable** (no GPU)")
        env_parts.append(
            f"Replicate API: **{'available' if HAS_REPLICATE else 'not configured'}**"
        )
        gr.Markdown(" | ".join(env_parts))

        with gr.Tabs():
            # ===================== Tab 1: Single Method =====================
            with gr.TabItem("スタイル変換"):
                with gr.Row():
                    with gr.Column(scale=1):
                        style_image = gr.Image(
                            label="スタイル参照画像（あなたのイラスト）",
                            type="pil",
                            height=300,
                        )
                    with gr.Column(scale=1):
                        content_image = gr.Image(
                            label="コンテンツ画像（変換したい画像）",
                            type="pil",
                            height=300,
                        )

                with gr.Row():
                    method_dropdown = gr.Dropdown(
                        choices=available_methods,
                        value=available_methods[0],
                        label="スタイル変換手法",
                    )
                    run_btn = gr.Button("スタイル変換実行", variant="primary", scale=1)

                with gr.Accordion("詳細パラメータ", open=False):
                    with gr.Row():
                        controlnet_scale = gr.Slider(
                            0.0, 1.5, value=0.8, step=0.05,
                            label="ControlNet Scale（構造維持の強さ）",
                            info="高いほど元画像の構造を維持",
                        )
                        style_strength = gr.Slider(
                            0.0, 1.5, value=0.6, step=0.05,
                            label="Style Strength（スタイルの強さ）",
                            info="高いほどスタイル参照画像に近づく",
                        )
                    with gr.Row():
                        num_steps = gr.Slider(
                            10, 50, value=30, step=1,
                            label="Inference Steps（推論ステップ数）",
                            info="多いほど品質が上がるが遅くなる",
                        )
                        guidance_scale = gr.Slider(
                            1.0, 20.0, value=7.5, step=0.5,
                            label="Guidance Scale（CFG）",
                            info="プロンプトへの忠実度",
                        )
                    with gr.Row():
                        prompt = gr.Textbox(
                            label="プロンプト（任意）",
                            placeholder="e.g. best quality, illustration style",
                            value="",
                        )
                        negative_prompt = gr.Textbox(
                            label="ネガティブプロンプト（任意）",
                            placeholder="e.g. lowres, blurry",
                            value="",
                        )

                with gr.Row():
                    with gr.Column(scale=2):
                        output_image = gr.Image(
                            label="変換結果", type="pil", height=400,
                        )
                    with gr.Column(scale=1):
                        preprocess_image = gr.Image(
                            label="構造抽出結果", type="pil", height=200,
                        )
                        result_info = gr.Textbox(
                            label="処理情報", lines=5,
                        )

                run_btn.click(
                    fn=run_style_transfer,
                    inputs=[
                        content_image, style_image, method_dropdown,
                        controlnet_scale, style_strength, num_steps,
                        guidance_scale, prompt, negative_prompt,
                    ],
                    outputs=[output_image, preprocess_image, result_info],
                )

            # ===================== Tab 2: Preprocessing Preview ============
            with gr.TabItem("前処理プレビュー"):
                gr.Markdown(
                    "コンテンツ画像から抽出される構造情報のプレビュー。"
                    "ControlNetはこれらの構造を元に線の位置関係を維持します。"
                )
                preview_input = gr.Image(
                    label="プレビューする画像", type="pil", height=300,
                )
                preview_btn = gr.Button("前処理を実行", variant="secondary")

                with gr.Row():
                    canny_output = gr.Image(label="Canny Edge", type="pil")
                    lineart_output = gr.Image(label="Lineart (DoG)", type="pil")

                preview_btn.click(
                    fn=preview_preprocessing,
                    inputs=[preview_input],
                    outputs=[canny_output, lineart_output],
                )

            # ===================== Tab 3: Method Info =======================
            with gr.TabItem("手法の説明"):
                gr.Markdown("""
                ## 実装されている手法

                ### Method A: ControlNet + IP-Adapter（ローカルGPU）
                - **構造維持**: ControlNet (Canny Edge) で入力画像のエッジを抽出し、生成時の構造制約として使用
                - **スタイル注入**: IP-Adapter Plus がCLIP画像エンコーダでスタイル参照画像の特徴を抽出し、
                  クロスアテンション層に注入
                - **特徴**: 最も確立された組み合わせ。安定性が高い
                - **要件**: VRAM 12GB+

                ### Method B: InstantStyle（ローカルGPU）
                - **構造維持**: ControlNet (Canny Edge)
                - **スタイル注入**: IP-Adapterの改良版。スタイル特徴をスタイル関連のアテンションブロック
                  **のみ**に注入し、コンテンツの漏洩を抑制
                - **特徴**: Method Aより「スタイルのみ」の転写に優れる。コンテンツとスタイルの分離が明確
                - **要件**: VRAM 12GB+

                ### Method C: Replicate API（クラウド）
                - **構造維持 + スタイル注入**: fofr/style-transfer モデル（ControlNet + IP-Adapter内蔵）
                - **特徴**: ローカルGPU不要。APIキー設定のみで利用可能
                - **コスト**: ~$0.02-0.08/画像
                - **要件**: `REPLICATE_API_TOKEN` 環境変数

                ---

                ## パラメータガイド

                | パラメータ | 説明 | 推奨値 |
                |-----------|------|--------|
                | ControlNet Scale | 構造維持の強さ。高いほど元画像の線・構図を忠実に再現 | 0.7-0.9 |
                | Style Strength | スタイル適用の強さ。高いほど参照イラストのスタイルに近づく | 0.4-0.8 |
                | Inference Steps | 推論ステップ数。多いほど品質向上・速度低下 | 25-35 |
                | Guidance Scale | プロンプトへの忠実度 | 5.0-10.0 |
                """)

        gr.Markdown(
            "---\n"
            "*DrawLikeMe v0.1.0 - Style Transfer Prototype*"
        )

    return app


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DrawLikeMe - Style Transfer App")
    parser.add_argument("--share", action="store_true", help="Create public share link")
    parser.add_argument("--server-port", type=int, default=7860)
    parser.add_argument("--server-name", type=str, default="0.0.0.0")
    args = parser.parse_args()

    app = build_ui()
    app.launch(
        server_name=args.server_name,
        server_port=args.server_port,
        share=args.share,
    )
