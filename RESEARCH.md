# DrawLikeMe - スタイル変換技術 調査レポート

## 概要

**結論: このアプリの開発は十分に実現可能です。**

ユーザーのイラストスタイルを学習し、アップロードされた画像をそのスタイルに変換しつつ、構造（線の位置関係・構図）を維持するアプリは、現在の技術で実現可能です。以下にその根拠と推奨アーキテクチャを示します。

---

## 1. 要件の整理

| 要件 | 説明 |
|------|------|
| スタイル学習 | ユーザーがアップロードしたイラスト（少数枚）からスタイルを抽出 |
| スタイル変換 | 新たにアップロードされた画像を、そのスタイルで描き変える |
| 構造維持 | 線の位置関係・構図・形状は元画像を保持する |
| スタイル要素 | 線画のタッチ、塗り方、色彩、テクスチャを変換対象とする |

---

## 2. 主要技術の調査結果

### 2.1 ControlNet（構造維持 - コア技術）

ControlNetは、Stable Diffusionに空間的な制約を追加するアーキテクチャ。入力画像からエッジ・線画・深度などの構造情報を抽出し、生成過程に注入する。

| プリプロセッサ | 構造忠実度 | 用途 |
|--------------|-----------|------|
| **Canny Edge** | 最高（~94%） | 全エッジの正確な保持 |
| **Lineart** | 高 | イラスト線画の保持（スタイル自由度あり） |
| **Lineart Anime** | 高 | アニメ・イラスト特化 |
| **Depth** | 中高（~92%） | 3D空間関係の保持 |
| **Tile** | 最高 | 全体的なコンテンツ忠実度 |

**DrawLikeMeへの推奨**: Lineart（またはLineart Anime）をメインに使用。Cannyを補助的に組み合わせることで、線の位置関係を高精度に維持可能。

**主要リポジトリ**:
- [lllyasviel/ControlNet](https://github.com/lllyasviel/ControlNet)
- HuggingFace: `lllyasviel/control_v11p_sd15_lineart`, `control_v11p_sd15s2_lineart_anime`
- FLUX: `black-forest-labs/FLUX.1-Canny-dev-lora`

### 2.2 IP-Adapter（スタイル注入 - 即時モード）

Tencent AI Lab開発。CLIP画像エンコーダで参照画像のスタイル特徴を抽出し、拡散モデルのクロスアテンション層に注入する。

| 特徴 | 詳細 |
|------|------|
| 必要画像数 | **1枚**（ゼロショット、学習不要） |
| 追加パラメータ | ~22M（軽量） |
| スタイル制御 | `ip_adapter_scale`パラメータ（0.0〜1.0） |
| ControlNet併用 | 可能（構造維持+スタイル注入の組み合わせ） |

**主要バリアント**:
- **IP-Adapter Plus**: パッチトークンを使用、より詳細なスタイル把握
- **IPAdapter-Instruct (2024)**: 「スタイルのみ」転写を明示的に指定可能
- **InstantStyle (2024)**: コンテンツCLIP埋め込みをスタイルCLIP埋め込みから差し引くことで、純粋なスタイル特徴を抽出

**主要リポジトリ**:
- [tencent-ailab/IP-Adapter](https://github.com/tencent-ailab/IP-Adapter)
- [instantX-research/InstantStyle](https://github.com/instantX-research/InstantStyle)
- HuggingFace: `h94/IP-Adapter`

### 2.3 LoRA微調整（スタイル学習 - 深層モード）

Low-Rank Adaptationにより、少数のイラストからスタイルを学習する小さなアダプター（~5MB）を訓練する。

| 手法 | 必要画像数 | 学習時間 | 特徴 |
|------|-----------|---------|------|
| **B-LoRA** (ECCV 2024) | **1枚** | ~15分（RTX 4090） | Block4=コンテンツ、Block5=スタイルと明示的に分離 |
| **標準LoRA** | 5-10枚 | 20-60分 | 汎用的で安定 |
| **FLUX LoRA** | 5枚〜 | 500-1500ステップ | 最新モデルベース |

**クラウド学習コスト**:
- fal.ai: FLUX LoRA学習 ~$2/回
- Replicate: ~$1.85/回（1000ステップ）

**主要リポジトリ**:
- [yardenfren1996/B-LoRA](https://github.com/yardenfren1996/B-LoRA)
- [kohya-ss/sd-scripts](https://github.com/kohya-ss/sd-scripts)

### 2.4 その他の注目手法（2024-2025）

| 手法 | 学習不要 | 構造維持 | 特徴 |
|------|---------|---------|------|
| **InstantStyle-Plus** | ○ | ○（Tile CN） | InstantStyle + Tile ControlNet + 反転コンテンツノイズ |
| **StyleShot** (OpenMMLab) | ○ | ○（lineart版あり） | 専用スタイルエンコーダ。lineart版がDrawLikeMeに直接適用可能 |
| **StyleID** (CVPR 2024) | ○ | △〜○ | アテンション操作+AdaINによるスタイル注入 |
| **CSGO** | ○ | ○ | コンテンツとスタイルの明示的分離 |
| **StyleStudio** (CVPR 2025) | ○ | △ | テキスト駆動スタイル変換 |

### 2.5 古典的Neural Style Transfer（比較用）

Gatys et al. (2015) のGram行列ベースのアプローチ。テクスチャ・色彩統計は捕捉できるが、「このアーティストならどう描くか」という抽象的なスタイルは学習できない。構造維持も中程度。**DrawLikeMeの主要手法としては非推奨**だが、クイックプレビュー用途には使える。

---

## 3. 比較サマリー

| アプローチ | 構造維持 | スタイル学習 | 学習不要 | 推論時間 | 成熟度 |
|-----------|---------|------------|---------|---------|--------|
| ControlNet + IP-Adapter | ★★★★★ | ★★★★ (1枚) | ○ | 5-15秒 | 成熟 |
| ControlNet + LoRA | ★★★★★ | ★★★★★ (1-5枚) | × | 5-15秒 | 成熟 |
| B-LoRA + ControlNet | ★★★★★ | ★★★★ (1枚) | ×(15分学習) | 5-15秒 | ECCV 2024 |
| InstantStyle-Plus | ★★★★★ | ★★★★ (1枚) | ○ | 5-15秒 | 2024 |
| StyleShot (lineart) | ★★★★ | ★★★★ (1枚) | ○ | 5-15秒 | 2024 |
| 古典的NST | ★★★ | ★★ (テクスチャのみ) | ○ | 5-60秒 | 非常に成熟 |

---

## 4. 推奨アーキテクチャ

### 2段階アプローチ

#### Tier 1: インスタントモード（学習不要 - IP-Adapter + ControlNet）

```
入力画像 → Lineart/Canny ControlNet（構造抽出）
                +
スタイル参照画像 → IP-Adapter/InstantStyle（スタイル注入）
                ↓
            スタイル変換済み画像
```

- **用途**: クイックプレビュー、初回利用、スタイル参照1枚
- **技術スタック**: SDXL or FLUX + ControlNet (Lineart + Canny) + IP-Adapter Plus / InstantStyle
- **推論時間**: 5-15秒/画像
- **コスト**: ~$0.02-0.08/画像（クラウドAPI）

#### Tier 2: ディープスタイルモード（LoRA学習 + ControlNet）

```
ユーザーイラスト（3-10枚）→ B-LoRA / 標準LoRA学習（2-20分）
                                    ↓
入力画像 → ControlNet（構造抽出）+ 学習済みLoRA（スタイル）→ スタイル変換済み画像
```

- **用途**: 細かなスタイル再現、プロイラストレーター向け、繰り返し利用
- **技術スタック**: FLUX or SDXL + LoRA (fal.ai or Replicate API) + ControlNet
- **学習時間**: 2-20分（初回のみ）
- **推論時間**: 5-15秒/画像
- **学習コスト**: $1.85-8.00/スタイル

---

## 5. 技術スタック

### バックエンド

| ライブラリ | 用途 |
|-----------|------|
| `diffusers` (HuggingFace) | ControlNet, IP-Adapter, SD/FLUXパイプライン |
| `transformers` | CLIPモデル（IP-Adapter用） |
| `torch` / `torchvision` | PyTorchバックエンド |
| `controlnet_aux` | ControlNetプリプロセッサ（canny, lineart, depth等） |
| `accelerate` | メモリ最適化、CPUオフロード |
| `Pillow` / `opencv-python` | 画像前処理 |

### フロントエンド候補

- **Next.js / React** - Webアプリ
- **Gradio** - 迅速なプロトタイプ
- **Streamlit** - データサイエンス向けUI

### デプロイ先候補

| プラットフォーム | 特徴 | コスト |
|---------------|------|--------|
| **Replicate** | APIファースト、事前ホスト済みモデル | ~$0.014/推論 |
| **fal.ai** | 高速LoRA学習、FLUX推論 | ステップ/メガピクセル課金 |
| **RunPod** | カスタムComfyUIワークフロー、スケール | $0.34/hr (RTX 4090) |
| **Modal** | サーバーレス、Python-first | RunPodと同等 |

---

## 6. GPU要件

| 構成 | VRAM | 備考 |
|------|------|------|
| SD 1.5 + ControlNet + IP-Adapter | 8-10 GB | 最低限の推論構成 |
| SDXL + ControlNet + IP-Adapter | 12-18 GB | 推奨構成 |
| FLUX + ControlNet | 24 GB+ | 高品質だが要求が高い |

**CPU推論**: 技術的には可能だが実用的ではない（GPU比10-50倍遅い）。**クラウドGPUの利用を強く推奨。**

---

## 7. 実現可能性の評価

### 結論: **開発は十分に実現可能**

| 観点 | 評価 | 理由 |
|------|------|------|
| 技術的実現性 | ★★★★★ | ControlNet + IP-Adapter/LoRAの組み合わせで全要件を満たせる |
| 少数画像対応 | ★★★★★ | IP-Adapterは1枚、B-LoRAも1枚から対応 |
| 構造維持 | ★★★★★ | Lineart/Canny ControlNetで高精度に維持可能 |
| スタイル再現 | ★★★★ | IP-Adapterで表面的スタイル、LoRAで深いスタイルを再現 |
| 開発コスト | ★★★★ | OSS中心、クラウドAPIで低コスト開発可能 |
| 運用コスト | ★★★★ | 推論$0.01-0.08/画像、LoRA学習$2-8/スタイル |
| レイテンシ | ★★★★ | 推論5-15秒/画像（GPU使用時） |

### リスクと注意点

1. **スタイル再現精度**: IP-Adapterは表面的なスタイル（色彩・テクスチャ）は得意だが、アーティスト固有の「描き方の癖」まで完全に再現するにはLoRA学習が必要
2. **GPU依存**: 実用的な速度にはGPUが必須。クラウドGPUサービスで解決可能
3. **モデル選択**: SDXL vs FLUXの選択で品質・コスト・速度のトレードオフがある
4. **著作権**: ユーザー自身のイラストスタイルを使用する前提なので、著作権リスクは低い

---

## 8. 主要参考リポジトリ

| プロジェクト | URL | 用途 |
|------------|-----|------|
| ControlNet | [lllyasviel/ControlNet](https://github.com/lllyasviel/ControlNet) | 構造維持 |
| IP-Adapter | [tencent-ailab/IP-Adapter](https://github.com/tencent-ailab/IP-Adapter) | スタイル注入 |
| InstantStyle | [instantX-research/InstantStyle](https://github.com/instantX-research/InstantStyle) | 改良型スタイル注入 |
| InstantStyle-Plus | [instantX-research/InstantStyle-Plus](https://github.com/instantX-research/InstantStyle-Plus) | スタイル+構造維持 |
| StyleShot | [open-mmlab/StyleShot](https://github.com/open-mmlab/StyleShot) | 単一画像スタイル変換 |
| B-LoRA | [yardenfren1996/B-LoRA](https://github.com/yardenfren1996/B-LoRA) | 1枚からのスタイル学習 |
| StyleID | [jiwoogit/StyleID](https://github.com/jiwoogit/StyleID) | 学習不要スタイル注入 |
| Kohya SD Scripts | [kohya-ss/sd-scripts](https://github.com/kohya-ss/sd-scripts) | LoRA学習ツールキット |
| Diffusers | [huggingface/diffusers](https://github.com/huggingface/diffusers) | メインパイプライン |
| Awesome Style Transfer | [Westlake-AGI-Lab/Awesome-Style-Transfer-with-Diffusion-Models](https://github.com/Westlake-AGI-Lab/Awesome-Style-Transfer-with-Diffusion-Models) | 包括的な論文リスト |

---

## 9. 次のステップ

1. **プロトタイプ開発**: Gradioで簡易UIを構築し、ControlNet + IP-Adapterのパイプラインを実装
2. **API選定**: Replicate or fal.aiのAPIを利用してクラウド推論を実装
3. **品質評価**: 実際のイラストでスタイル変換品質を検証
4. **LoRA統合**: Tier 2のディープスタイルモード実装
5. **本番UI**: Next.js等でWebアプリとして構築
