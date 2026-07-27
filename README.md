# AcademiaSD LoRAlab-Krea2

---

## 🌟 Overview

**AcademiaSD LoRAlab-Krea2** is an advanced, high-performance training suite and web interface designed for training high-quality LoRA models on **Krea-2 (Rectified Flow / Diffusion Transformer)** architectures.

It features an interactive web GUI with real-time VRAM/RAM monitoring, face-identity dataset curation, progressive multi-phase resolution training, background session persistence via tmux, advanced training dynamics, dataset caption inspection, batch search-and-replace tag editing, and automated headless CLI execution.

---

## ✨ Key Features

- **🌐 Interactive Web Interface**: Sleek dark-mode web GUI built for configuring parameters, inspecting datasets, tracking console logs, and previewing generated step outputs.
- **🎯 Multi-Criteria Dataset Curation & Priority Weighting (`0_curate_dataset.py`)**:
  - **4 Specialized Curation Modes**: Supports **Face** (ArcFace identity), **Clothes** (CLIP style), **Body-Type** (pose/shape), and **Tattoo** (zero-shot CLIP prompt anchoring + HSV saliency + caption keyword synergy).
  - **⭐ High-Priority Baseline Weighting ($1.5\times$)**: Starred baseline images (★) automatically receive a **$1.5\times$ priority loss weight** during training to accelerate learning on target features.
  - **Non-Destructive Dataset Weighting**: Partitions remaining images into `weight_good` ($1.0\times$) and `weight_bad` ($0.5\times$) groups to modulate loss contribution without deleting files.
  - **Mode-Specific Baseline Persistence**: Saves independent baseline selections per mode (`face`, `clothes`, `body-type`, `tattoo`).
  - **Unified vs Split View**: The **Dataset** tab displays a single flat grid for fast inspection, while the **Curation** tab displays quality group splits and loss weights.
  - **Instant Dark Tooltips**: Space-saving inline tooltips (`ⓘ`) on titles provide instant context without cluttering the screen.
- **🔄 Session Persistence & Resilient Execution**:
  - Auto-wrapped background `tmux` sessions (`loralab` web server and `loralab-batch` CLI runner) to protect active runs against SSH drops or terminal exits.
  - Centralized multi-process state tracking (`logs/process_state.json`) and isolated per-process log files.
  - Automatic Web UI state recovery and live log streaming upon browser refresh or reconnection.
- **📈 Progressive Multi-Resolution & Constant-Resolution Modes**:
  - Train sequentially across resolution phases (e.g., $512^2 \rightarrow 768^2 \rightarrow 1024^2$) with cross-phase weight persistence, global progress tracking, and per-phase learning rate re-warmups.
  - Flexible single-resolution modes (`512`, `768`, `1024` or custom arbitrary resolutions like `640`, `896`).
- **🔍 Batch Caption Search & Replace (`F2` Shortcut)**:
  - Interactive modal triggered by pressing **`F2`** or clicking **`🔍 Reemplazar (F2)`** in the dataset toolbar.
  - Real-time match counter displaying exact occurrences and affected image counts as you type.
  - Options for **Case Sensitivity**, **Whole Word Matching**, and **Automatic Comma/Whitespace Cleaning** when removing tags.
  - REST API endpoint (`POST /api/batch-replace-caption`) for instant server-side updates.
- **🛡️ Anomaly Guards & Reliability**:
  - **NaN Guard**: Detects NaN loss/gradients and halts training to protect optimizer moments from corruption.
  - **OOM Guard**: Automatically catches Out-of-Memory exceptions.
  - **Strict Weight Validation**: Verifies $\|\text{LoRA}_B\| > 0$ and target module compatibility upon loading checkpoints, preventing silent training from random initialization.
  - **Corrupt Checkpoint Protection**: Prevents silent resets to step 0 on invalid state files.
- **⚡ Advanced Optimization & Precision**:
  - Paged 8-bit AdamW (`adamw8bit_paged`), AdamW, or 8-bit AdamW.
  - Opt-in High-Precision FP32 target generation (`high_precision_targets`) & FP32 LoRA weight precision (`lora_dtype`).
  - Exponential Moving Average (EMA) support (`use_ema`, `ema_decay`, `ema_device`).
  - Flexible warmup units (`updates`, `micro_steps`, `ratio`).
  - Timestep weighting & Rectified Flow scheduling (`logit_normal_mu`, `sigma_min`, `sigma_max`, `content_or_style`).
- **📊 Observability & Safetensors Metadata**:
  - Step-by-step CSV logging (`train_log.csv`).
  - Safetensors training metadata export (`ss_network_dim`, `ss_network_alpha`, `ss_tag_frequency`, `modelspec`).
  - Optional `.alpha` tensor exporting for third-party loader compatibility.

---

## 📌 Configuration Files Setup

> ⚠️ **IMPORTANT:** Real configuration files (`pre_cache_settings.json`, `train_settings.json`, `train_advanced.json`) store local paths and Hugging Face authentication tokens (`hf_token`). They are excluded from git tracking for security.
> 
> To set up your configuration for Web UI or CLI batch execution, copy the template files:
> 
> ```bash
> cp pre_cache_settings.json.example pre_cache_settings.json
> cp train_settings.json.example train_settings.json
> cp train_advanced.json.example train_advanced.json
> ```
> 
> Edit `pre_cache_settings.json` and `train_settings.json` to define your dataset location, project name, training parameters, and Hugging Face token.
>
> 🔑 **Default Web UI Credentials:**
> - **Username:** `admin`
> - **Password:** `admin`

---

## 🚀 Execution Guide

### Linux / macOS

1. **Environment Installation:**
   ```bash
   ./install_LoRAlab-Krea2.sh
   ```
2. **Launch Web Interface:**
   ```bash
   ./run_LoRAlab-Krea2.sh
   ```
   *Opens the Web GUI at `http://127.0.0.1:5000` (auto-managed inside `tmux` session `loralab`).*

3. **Headless Batch Execution (CLI):**
   ```bash
   ./run_batch_cli.sh all
   ```
   *Runs inside `tmux` session `loralab-batch`. Arguments: `precache`, `train`, `all`.*

4. **Dataset Curation CLI (Optional):**
   ```bash
   python scripts/python/0_curate_dataset.py
   ```

### Windows

1. **Environment Installation:**
   ```cmd
   Install_LoRAlab-Krea2.bat
   ```
2. **Launch Web Interface:**
   ```cmd
   Run_LoRAlab-Krea2.bat
   ```

---

## 📁 Project Directory Structure

```
├── run_LoRAlab-Krea2.sh              # Web GUI launcher (Linux/macOS)
├── install_LoRAlab-Krea2.sh          # Installation script (Linux/macOS)
├── Run_LoRAlab-Krea2.bat             # Web GUI launcher (Windows)
├── Install_LoRAlab-Krea2.bat         # Installation script (Windows)
├── run_batch_cli.sh                  # Batch execution script (CLI / Headless)
├── pre_cache_settings.json.example   # Template for pre-cache & curation settings
├── train_settings.json.example       # Template for basic training settings
├── train_advanced.json.example       # Template for advanced settings & presets
├── assets/                           # Media & project banners
├── web/
│   └── trainer_ui.html               # Main Web GUI interface
├── scripts/
│   ├── python/
│   │   ├── 0_curate_dataset.py       # Face-identity dataset curation engine (CPU)
│   │   ├── 1_pre_cache_krea2.py      # Latent & text embedding pre-caching engine
│   │   ├── 2_train_lora_krea2.py     # Core LoRA training loop
│   │   ├── run_progressive.py        # Multi-phase & single-res orchestrator
│   │   └── server.py                 # Flask web backend & REST API server
│   ├── shell/                        # Additional Shell utilities
│   └── batch/                        # Additional Windows batch scripts
├── cached_data_local/                # Generated latents & embeddings cache
└── output_local/                     # Output LoRA weights, checkpoints, & logs
```

> **Note:** Scripts in `scripts/python/` automatically resolve their relative paths against the project root, so they can be invoked from any working directory.

---

## 📄 License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
