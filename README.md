<h1 align="center">Quick Mflux on ComfyUI</h1>

<p align="center">
  <font size=5>English</font>
</p>

Simple use of [mflux](https://github.com/filipstrand/mflux) in ComfyUI, suitable for users who are not familiar with terminal usage.
**Only for macOS (Apple Silicon).** This is a fork of [raysers/Mflux-ComfyUI](https://github.com/raysers/Mflux-ComfyUI) updated to mflux 0.16.6 with support for all current model families and new generation modes made with claude.ai.

> NOTE: A MLX port of FLUX and other state-of-the-art diffusion models based on the Huggingface Diffusers implementation.

---

## ✨ What's new in this fork (mflux 0.16.6)

### New model families
In addition to the classic FLUX.1-schnell and FLUX.1-dev, the following model families are now supported:

| Model | Family | Notes |
|---|---|---|
| `schnell`, `dev` | FLUX.1 | Classic, unchanged |
| `krea-dev`, `kontext-dev` | FLUX.1 | New variants |
| `flux2-klein-4b`, `flux2-klein-9b` | FLUX.2 | Distilled, fast — guidance locked to 1.0 |
| `flux2-base-4b`, `flux2-base-9b`, `flux2-dev` | FLUX.2 | Full FLUX.2 variants |
| `z-image-turbo`, `z-image-base` | Z-Image | Tongyi Lab models |
| `qwen-image` | Qwen | Multilingual prompts + negative prompt support |

### Extended quantization
Quantization now supports **3, 4, 5, 6 and 8-bit** (previously only 4 and 8-bit).

### New generation nodes
Six new generation nodes for modes that were not previously available:

| Node | Function |
|---|---|
| **MFlux Fill (Inpainting)** | Fill masked regions of an image using a text prompt |
| **MFlux Depth Conditioning** | Generate images guided by the depth structure of a reference image |
| **MFlux Redux (Image Variation)** | Create image variations without a text prompt |
| **MFlux Kontext (Image Editing)** | Edit images with natural language instructions (e.g. "change the background to a sunset") |
| **MFlux Qwen Image** | Text-to-image with multilingual prompts and negative prompt support |
| **MFlux Qwen Image Edit** | Semantic image editing with 1–4 reference images |

### New input helper nodes
| Node | Function |
|---|---|
| **MFlux Fill Loader** | Loads image + mask (file upload or ComfyUI MASK tensor) for the Fill node |
| **MFlux Image Ref Loader** | Loads 1–4 reference images (upload or IMAGE tensor) for Kontext, Depth, Redux and Qwen Edit |
| **MFlux Scale Factor** | Calculates output dimensions from input image × scale factor (e.g. 2× for upscaling) |

### MFlux LoRAs Loader extended
The LoRA loader now has an optional `hf_lora` text input for loading LoRAs directly from HuggingFace without manual download — just enter `author/repo` or `author/repo:filename.safetensors`.

---

## Installation

```bash
cd /path/to/your_ComfyUI
# Activate your virtual environment
cd custom_nodes
git clone https://github.com/Kuwe93/Mflux-ComfyUI.git
pip install mflux==0.16.6
# Restart ComfyUI
```

---

## Node overview

All nodes can be found by double-clicking the canvas and searching for **"MFlux"** or **"Mflux"**.

### MFlux/Air — Generation nodes

| Node | Category | Description |
|---|---|---|
| Quick MFlux Generation | Air | Standard txt2img / img2img / ControlNet for all model families |
| MFlux Models Loader | Air | Load a locally saved model from `models/Mflux/` |
| MFlux Models Downloader | Air | Download pre-quantized models from HuggingFace |
| MFlux Custom Models | Air | Quantize and save any supported model locally |
| MFlux Fill (Inpainting) | Air | FLUX.1 Fill — inpaint masked image regions |
| MFlux Depth Conditioning | Air | FLUX.1 Depth — depth-guided image generation |
| MFlux Redux (Image Variation) | Air | FLUX.1 Redux — image variations without a text prompt |
| MFlux Kontext (Image Editing) | Air | FLUX.1 Kontext — natural language image editing |
| MFlux Qwen Image | Air | Qwen txt2img with negative prompt |
| MFlux Qwen Image Edit | Air | Qwen semantic image editing with reference images |

### MFlux/Pro — Input helper nodes

| Node | Category | Description |
|---|---|---|
| MFlux Img2Img | Pro | Load an image + strength for img2img use in Quick MFlux Generation |
| MFlux LoRAs Loader | Pro | Load up to 3 LoRAs (chainable), now also supports HuggingFace LoRA strings |
| MFlux ControlNet Loader | Pro | Load a Canny control image for FLUX.1 ControlNet |
| MFlux Fill Loader | Pro | Load image + mask for Fill/Inpainting |
| MFlux Image Ref Loader | Pro | Load 1–4 reference images for Kontext / Depth / Redux / Qwen Edit |
| MFlux Scale Factor | Pro | Compute scaled width/height for upscaling workflows |

---

## Models

### Downloading pre-quantized models
Use the **MFlux Models Downloader** node. Available models include FLUX.1-schnell 4-bit, FLUX.1-dev 4-bit, FLUX.1-Kontext-dev 4-bit, FLUX.2-Klein 4-bit and Z-Image-Turbo 4-bit.

### Saving custom quantized models
Use **MFlux Custom Models** to quantize any supported model (3/4/5/6/8-bit) and save it to `models/Mflux/`. Once saved, load it with **MFlux Models Loader** for all future generations.

### LoRA usage note
LoRAs and pre-quantized local models are **mutually exclusive** — you cannot load a LoRA on top of an already-quantized local model. To use both, quantize the model with the LoRA baked in using **MFlux Custom Models**.
I usually keep a `models/loras/Mflux/` subfolder for mflux-compatible LoRAs.

---

## Metadata
The **metadata** option (default: True) automatically saves generated images to `ComfyUI/output/MFlux/` alongside a JSON sidecar file with all generation parameters. The JSON now also includes `model_family` and `model_alias` fields for the new model families.

---

## Tips

- **Guidance for FLUX.2-Klein**: The distilled Klein models (4b/9b) automatically lock guidance to 1.0 — setting a different value has no effect.
- **Qwen**: 8-bit quantization is recommended for best quality. Multilingual prompts are supported natively.
- **Kontext guidance**: Values between 2.0 and 4.0 work best for image editing tasks.
- **Fill guidance**: High guidance values (20–50) are recommended for inpainting.
- **Scale Factor**: Use the **MFlux Scale Factor** node to chain upscaling into any img2img workflow without manually calculating dimensions.
- If nodes show in red after installation, use ComfyUI-Manager's **"One-click Install Missing Nodes"**.
- Preview nodes do not auto-save — replace with a Save Image node or right-click to save manually.

---

## Credits

Special thanks to:
- [@filipstrand](https://github.com/filipstrand) and [@anthonywu](https://github.com/anthonywu) for the [mflux](https://github.com/filipstrand/mflux) project
- [@raysers](https://github.com/raysers) for the original [Mflux-ComfyUI](https://github.com/raysers/Mflux-ComfyUI) plugin this fork is based on
- [@CharafChnioune](https://github.com/CharafChnioune) for [MFLUX-WEBUI](https://github.com/CharafChnioune/MFLUX-WEBUI) — portions of the original code were referenced under the Apache 2.0 license
- [@InformEthics](https://github.com/InformEthics) for the original ControlNet node contribution
