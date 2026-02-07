# DrawLikeMe - Windows セットアップガイド

## 動作環境

- **OS**: Windows 10 / 11
- **GPU**: NVIDIA GPU（VRAM 12GB）
  例: RTX 3060 12GB, RTX 4070, RTX 3080 など
- **Python**: 3.10 または 3.11（3.12 も可）
- **ストレージ**: 約 25GB の空き容量（モデルのダウンロードに必要）

---

## Step 1: 前提ソフトウェアの確認

### 1-1. Python のインストール

Python がインストールされていない場合:

1. https://www.python.org/downloads/ から **Python 3.11** をダウンロード
2. インストーラを実行し、**「Add python.exe to PATH」にチェック**を入れてからインストール

確認:
```
python --version
```
`Python 3.10.x` ～ `3.12.x` が表示されれば OK。

### 1-2. Git のインストール

Git がインストールされていない場合:

1. https://git-scm.com/download/win からダウンロードしてインストール

確認:
```
git --version
```

### 1-3. NVIDIA ドライバの確認

```
nvidia-smi
```

- **Driver Version** と **CUDA Version** が表示されれば OK
- CUDA Version が **12.1 以上**であることを確認
- 表示されない場合は https://www.nvidia.com/drivers/ から最新ドライバをインストール

---

## Step 2: プロジェクトの取得

PowerShell またはコマンドプロンプトで:

```
cd %USERPROFILE%\Documents
git clone https://github.com/ykitaguchi77/DrawLikeMe.git
cd DrawLikeMe
```

---

## Step 3: Python 仮想環境の作成

```
python -m venv venv
venv\Scripts\activate
```

プロンプトの先頭に `(venv)` が表示されれば OK。

> **以降の全てのコマンドは仮想環境が有効な状態で実行してください。**
> 新しいターミナルを開いた場合は、再度 `venv\Scripts\activate` を実行してください。

---

## Step 4: PyTorch（CUDA版）のインストール

**重要: PyTorch は pip install だけでは CPU 版がインストールされます。**
必ず以下のコマンドで CUDA 版をインストールしてください。

### CUDA 12.1 の場合（推奨）:
```
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### CUDA 12.4 の場合:
```
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

> **どちらを使うべきか？**
> `nvidia-smi` で表示された CUDA Version を確認してください。
> - 12.1 ～ 12.3 → `cu121` を使用
> - 12.4 以上 → `cu124` を使用

### インストール確認:
```
python -c "import torch; print(f'PyTorch {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0)}' if torch.cuda.is_available() else 'No GPU')"
```

以下のように表示されれば OK:
```
PyTorch 2.x.x+cu121
CUDA available: True
GPU: NVIDIA GeForce RTX XXXX
```

**「CUDA available: False」の場合:**
- PyTorch の CPU 版がインストールされています
- `pip uninstall torch torchvision` で削除してから再度 CUDA 版をインストール

---

## Step 5: その他の依存パッケージのインストール

```
pip install diffusers transformers accelerate safetensors
pip install gradio
pip install Pillow opencv-python-headless numpy
```

---

## Step 6: アプリの起動

```
python app.py
```

初回起動時の出力例:
```
GPU: NVIDIA GeForce RTX 3060 (12.0 GB) | 全手法利用可能
Running on local URL: http://0.0.0.0:7860
```

**ブラウザで http://localhost:7860 を開いてください。**

---

## Step 7: 初回のスタイル変換（モデルの自動ダウンロード）

初めてスタイル変換を実行すると、以下のモデルが自動ダウンロードされます:

| モデル | サイズ | 用途 |
|--------|--------|------|
| SDXL Base 1.0 | ~6.5GB | ベースモデル (Method A, B) |
| ControlNet Canny SDXL | ~2.5GB | 構造維持 (Method A, B) |
| SDXL VAE (fp16-fix) | ~335MB | 画像デコーダ |
| IP-Adapter Plus SDXL | ~100MB | スタイル注入 |
| CLIP ViT-H Image Encoder | ~2.5GB | スタイル特徴抽出 |
| SD 1.5 | ~4GB | ベースモデル (Method C) |
| ControlNet Lineart SD1.5 | ~1.4GB | 線画構造維持 (Method C) |
| IP-Adapter Plus SD1.5 | ~100MB | スタイル注入 (Method C) |

**初回は合計約 15-20GB のダウンロードが必要です。**
ダウンロードしたモデルは `~/.cache/huggingface/` に保存され、2回目以降は再ダウンロード不要です。

> ダウンロードが途中で止まった場合は、再度実行すれば途中から再開されます。

---

## 使い方

### 基本的な流れ

1. **「スタイル参照画像」**に自分の医学イラストをアップロード
2. **「コンテンツ画像」**に変換したい画像をアップロード
3. **手法を選択**（下記参照）
4. **「スタイル変換を実行」**をクリック

### どの手法を選ぶべきか

**12GB VRAM 環境での推奨:**

| 手法 | VRAM使用量 | 推奨度 | 特徴 |
|------|-----------|--------|------|
| **C: Lineart ControlNet (SD 1.5)** | ~8GB | 最初に試す | 線画の忠実度が最高。VRAM に余裕あり |
| **B: InstantStyle (SDXL)** | ~12GB | 次に試す | 構造の歪みが最も少ない。VRAM ギリギリ |
| **A: ControlNet + IP-Adapter (SDXL)** | ~12GB | 最後に試す | 高解像度で安定。VRAM ギリギリ |

> **12GB VRAM では Method A, B はメモリが厳しい場合があります。**
> もし CUDA out of memory エラーが出たら:
> 1. まず Method C を使う（確実に動作）
> 2. 他のGPUを使うアプリを閉じる（ブラウザ、ゲーム等）
> 3. 入力画像のサイズを小さくする

### 前処理プレビュー

「前処理プレビュー」タブで、画像からどのように構造が抽出されるかを確認できます。
GPUなしでも動作するので、まずここで試すと安心です。

---

## トラブルシューティング

### CUDA out of memory

SDXL 系（Method A, B）で発生する場合:

```
# 環境変数で VRAM 節約モードを有効化
set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python app.py
```

それでもダメな場合は **Method C (SD 1.5)** を使ってください（~8GB で動作）。

### モデルのダウンロードが遅い / 失敗する

Hugging Face のミラーを使う:
```
set HF_ENDPOINT=https://hf-mirror.com
python app.py
```

### 「torch not compiled with CUDA enabled」エラー

CPU版 PyTorch がインストールされています:
```
pip uninstall torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### Gradio の画面が表示されない

ファイアウォールがブロックしている可能性:
```
python app.py --server-name 127.0.0.1
```

---

## ストレージ使用量の確認

モデルキャッシュの場所と容量:
```
# Windows
dir /s "%USERPROFILE%\.cache\huggingface\hub"
```

キャッシュを削除したい場合:
```
# 全モデルキャッシュを削除（次回実行時に再ダウンロード）
rmdir /s /q "%USERPROFILE%\.cache\huggingface\hub"
```
