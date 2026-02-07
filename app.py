"""
DrawLikeMe - Medical Illustration Style Transfer App

Upload your medical illustrations to define your style, then transform
any image into your unique artistic style while preserving anatomical
structure.

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

from config import (
    DEVICE,
    DEFAULT_CONTROLNET_SCALE,
    DEFAULT_IP_ADAPTER_SCALE,
    DEFAULT_NUM_INFERENCE_STEPS,
    DEFAULT_GUIDANCE_SCALE,
    DEFAULT_PROMPT,
    DEFAULT_NEGATIVE_PROMPT,
)
from pipelines.preprocessors import extract_canny, extract_lineart, extract_adaptive_threshold
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

_pipeline_cache: dict = {}


# ---------------------------------------------------------------------------
# Available methods (all local GPU)
# ---------------------------------------------------------------------------
METHODS = {
    "A: ControlNet + IP-Adapter (SDXL)": {
        "loader": "controlnet_ipadapter",
        "class": "ControlNetIPAdapterPipeline",
        "description": (
            "Canny ControlNet でエッジ構造を維持 + IP-Adapter Plus でスタイル注入。"
            "SDXL ベース（1024px）。最も標準的な組み合わせ。"
        ),
        "vram": "~12GB",
    },
    "B: InstantStyle (SDXL)": {
        "loader": "instantstyle",
        "class": "InstantStylePipeline",
        "description": (
            "IP-Adapter のスタイル注入をスタイル専用ブロックのみに制限。"
            "解剖学的構造への影響を最小化。SDXL ベース（1024px）。"
        ),
        "vram": "~12GB",
    },
    "C: Lineart ControlNet + IP-Adapter (SD 1.5)": {
        "loader": "lineart_ipadapter",
        "class": "LineartIPAdapterPipeline",
        "description": (
            "線画専用 ControlNet でイラストの線を高精度に維持。"
            "SD 1.5 ベース（512px）。線画の忠実度が最も高い。"
        ),
        "vram": "~8GB",
    },
}


def get_available_methods() -> list[str]:
    if HAS_GPU:
        return list(METHODS.keys())
    return ["プレビューのみ（GPU未検出）"]


def get_pipeline(method: str):
    if method in _pipeline_cache:
        return _pipeline_cache[method]

    info = METHODS.get(method)
    if info is None:
        return None

    module = __import__(f"pipelines.{info['loader']}", fromlist=[info["class"]])
    cls = getattr(module, info["class"])
    pipe = cls(device=CURRENT_DEVICE)
    _pipeline_cache[method] = pipe
    return pipe


# ---------------------------------------------------------------------------
# Core processing
# ---------------------------------------------------------------------------
def preview_preprocessing(content_image: Image.Image):
    """Show all preprocessing results for the content image."""
    if content_image is None:
        return None, None, None
    content = prepare_image(content_image)
    canny = extract_canny(content)
    lineart = extract_lineart(content)
    adaptive = extract_adaptive_threshold(content)
    return canny, lineart, adaptive


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

    if method == "プレビューのみ（GPU未検出）":
        content = prepare_image(content_image)
        canny = extract_canny(content)
        lineart = extract_lineart(content)
        gr.Info(
            "GPUが検出されません。前処理結果のみ表示しています。\n"
            "CUDAまたはMPS対応GPUのある環境で実行してください。"
        )
        return canny, lineart, "プレビューのみ - スタイル変換にはGPUが必要です"

    pipeline = get_pipeline(method)
    if pipeline is None:
        raise gr.Error(f"パイプライン '{method}' を初期化できませんでした。")

    progress(0.1, desc="画像を前処理中...")
    progress(0.2, desc="モデルを読み込み中（初回は数分かかります）...")

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
        raise gr.Error(f"スタイル変換中にエラーが発生: {e}\n{traceback.format_exc()}")

    progress(1.0, desc="完了")

    preprocess_img = None
    if result.preprocessing_images:
        preprocess_img = list(result.preprocessing_images.values())[0]

    param_str = "\n".join(f"  {k}: {v}" for k, v in result.parameters.items())
    info = f"Method: {result.method_name}\nParameters:\n{param_str}"

    return result.output_image, preprocess_img, info


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------
def build_ui():
    available_methods = get_available_methods()

    with gr.Blocks(
        title="DrawLikeMe",
        theme=gr.themes.Soft(),
    ) as app:
        gr.Markdown(
            """
            # DrawLikeMe
            ### 医学イラストのスタイル変換アプリ

            自分の描いたイラストのスタイル（線画のタッチ・塗り方）で、
            別の画像を描き変えます。解剖学的な構造は維持されます。

            **ワークフロー:**
            1. 自分のイラストを「スタイル参照画像」にアップロード
            2. 変換したい画像を「コンテンツ画像」にアップロード
            3. 手法を選んで実行
            """
        )

        # Environment info
        if HAS_GPU and CURRENT_DEVICE == "cuda":
            gpu_name = torch.cuda.get_device_name(0)
            vram = torch.cuda.get_device_properties(0).total_mem / (1024**3)
            gr.Markdown(f"GPU: **{gpu_name}** ({vram:.1f} GB) | 全手法利用可能")
        elif HAS_GPU:
            gr.Markdown(f"Device: **{CURRENT_DEVICE}** | 全手法利用可能")
        else:
            gr.Markdown(
                "**GPU未検出** - 前処理プレビューのみ利用可能。\n"
                "スタイル変換にはCUDA/MPS対応GPUが必要です。"
            )

        with gr.Tabs():
            # ==================== Tab 1: Style Transfer ====================
            with gr.TabItem("スタイル変換"):
                with gr.Row():
                    with gr.Column(scale=1):
                        style_image = gr.Image(
                            label="スタイル参照画像（自分のイラスト）",
                            type="pil",
                            height=300,
                        )
                    with gr.Column(scale=1):
                        content_image = gr.Image(
                            label="コンテンツ画像（変換対象）",
                            type="pil",
                            height=300,
                        )

                with gr.Row():
                    method_dropdown = gr.Dropdown(
                        choices=available_methods,
                        value=available_methods[0],
                        label="スタイル変換手法",
                        scale=3,
                    )
                    run_btn = gr.Button(
                        "スタイル変換を実行",
                        variant="primary",
                        scale=1,
                    )

                # Method description
                method_desc = gr.Markdown(
                    value=_get_method_description(available_methods[0]),
                )
                method_dropdown.change(
                    fn=_get_method_description,
                    inputs=[method_dropdown],
                    outputs=[method_desc],
                )

                with gr.Accordion("パラメータ調整", open=False):
                    gr.Markdown(
                        "医学イラスト向けに最適化済み。"
                        "構造維持を高め、スタイル強度を控えめにしています。"
                    )
                    with gr.Row():
                        controlnet_scale = gr.Slider(
                            0.3, 1.5, value=DEFAULT_CONTROLNET_SCALE, step=0.05,
                            label="構造維持の強さ（ControlNet Scale）",
                            info="高い＝元画像の線・構造を忠実に維持。医学イラストでは0.85-1.0推奨",
                        )
                        style_strength = gr.Slider(
                            0.1, 1.2, value=DEFAULT_IP_ADAPTER_SCALE, step=0.05,
                            label="スタイルの強さ",
                            info="高い＝参照イラストのスタイルに近づく。0.4-0.6推奨",
                        )
                    with gr.Row():
                        num_steps = gr.Slider(
                            15, 50, value=DEFAULT_NUM_INFERENCE_STEPS, step=1,
                            label="推論ステップ数",
                            info="多い＝高品質だが低速。25-35推奨",
                        )
                        guidance_scale = gr.Slider(
                            1.0, 15.0, value=DEFAULT_GUIDANCE_SCALE, step=0.5,
                            label="Guidance Scale",
                            info="プロンプトへの忠実度。5.0-10.0推奨",
                        )
                    with gr.Row():
                        prompt = gr.Textbox(
                            label="プロンプト",
                            value=DEFAULT_PROMPT,
                            info="生成を誘導するテキスト",
                        )
                        negative_prompt = gr.Textbox(
                            label="ネガティブプロンプト",
                            value=DEFAULT_NEGATIVE_PROMPT,
                            info="避けたい要素",
                        )

                with gr.Row():
                    with gr.Column(scale=2):
                        output_image = gr.Image(
                            label="変換結果", type="pil", height=400,
                        )
                    with gr.Column(scale=1):
                        preprocess_image = gr.Image(
                            label="構造抽出結果（ControlNet入力）",
                            type="pil",
                            height=200,
                        )
                        result_info = gr.Textbox(
                            label="処理情報", lines=6,
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

            # ==================== Tab 2: Preprocessing Preview =============
            with gr.TabItem("前処理プレビュー"):
                gr.Markdown(
                    """
                    コンテンツ画像からどのように構造が抽出されるかをプレビューします。
                    GPUなしでも動作します。

                    - **Canny Edge**: エッジ検出。シャープな境界線を抽出
                    - **Lineart (DoG)**: 線画抽出。イラストの線を再現
                    - **Adaptive Threshold**: 適応的二値化。コントラストが不均一な画像に有効
                    """
                )
                preview_input = gr.Image(
                    label="プレビューする画像", type="pil", height=300,
                )
                preview_btn = gr.Button("前処理を実行", variant="secondary")

                with gr.Row():
                    canny_output = gr.Image(label="Canny Edge", type="pil")
                    lineart_output = gr.Image(label="Lineart (DoG)", type="pil")
                    adaptive_output = gr.Image(label="Adaptive Threshold", type="pil")

                preview_btn.click(
                    fn=preview_preprocessing,
                    inputs=[preview_input],
                    outputs=[canny_output, lineart_output, adaptive_output],
                )

            # ==================== Tab 3: Method Details ====================
            with gr.TabItem("手法の詳細"):
                gr.Markdown(
                    """
                    ## 3つの手法の比較

                    | | Method A | Method B | Method C |
                    |---|---|---|---|
                    | **手法** | ControlNet + IP-Adapter | InstantStyle | Lineart ControlNet + IP-Adapter |
                    | **ベースモデル** | SDXL (1024px) | SDXL (1024px) | SD 1.5 (512px) |
                    | **構造抽出** | Canny Edge | Canny Edge | Lineart (線画特化) |
                    | **スタイル注入** | IP-Adapter Plus (全ブロック) | IP-Adapter Plus (スタイルブロックのみ) | IP-Adapter Plus (全ブロック) |
                    | **VRAM** | ~12GB | ~12GB | ~8GB |
                    | **特徴** | 標準的・安定 | スタイルとコンテンツの分離が明確 | 線画の忠実度が最高 |

                    ---

                    ### Method A: ControlNet + IP-Adapter (SDXL)
                    最も標準的な組み合わせ。Canny エッジ検出でコンテンツ画像の構造を抽出し、
                    ControlNet で構造を維持しながら、IP-Adapter Plus がスタイル参照画像の
                    CLIP 特徴量をクロスアテンション層に注入してスタイルを適用します。

                    **医学イラストでの利点**: 高解像度出力(1024px)。全般的に安定した結果。

                    ### Method B: InstantStyle (SDXL)
                    IP-Adapter のスタイル特徴を、SDXL の**スタイル関連アテンションブロック
                    (up_blocks.0.attentions.1)のみ**に注入します。これにより、スタイル
                    参照画像のコンテンツがリークして解剖学的構造を歪めるリスクを低減します。

                    **医学イラストでの利点**: 構造の歪みが最も少ない。解剖学的正確性が重要な場合に最適。

                    ### Method C: Lineart ControlNet + IP-Adapter (SD 1.5)
                    線画に特化した ControlNet モデル (`control_v11p_sd15_lineart`) を使用。
                    Canny よりもイラストの線をより正確に捉えます。SD 1.5 ベースのため
                    解像度は 512px ですが、線画の忠実度は最も高くなります。

                    **医学イラストでの利点**: 線画のスタイル変換に最も適している。VRAMが少なくても動作。

                    ---

                    ## パラメータの意味

                    - **構造維持の強さ (ControlNet Scale)**: 0.9がデフォルト。
                      医学イラストでは解剖学的正確性のため高めに設定。
                      1.0以上にするとほぼ構造そのままでスタイルのみ変化
                    - **スタイルの強さ**: 0.5がデフォルト。
                      高くしすぎると構造が崩れる可能性あり。0.3-0.7の範囲で調整推奨
                    """
                )

        gr.Markdown("---\n*DrawLikeMe v0.2.0 - Medical Illustration Style Transfer*")

    return app


def _get_method_description(method: str) -> str:
    info = METHODS.get(method)
    if info is None:
        return ""
    return f"> {info['description']}  \n> VRAM: {info['vram']}"


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DrawLikeMe")
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
