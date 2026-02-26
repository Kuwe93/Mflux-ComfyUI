import os
from pathlib import Path
from huggingface_hub import snapshot_download
from folder_paths import models_dir
from mflux.models.common.config import ModelConfig
from mflux.models.flux.variants.txt2img.flux import Flux1

from .Mflux_Core import (
    MODEL_FAMILY_MAP,
    ALL_QUANTIZE_OPTIONS,
    FLUX2_DISTILLED_MODELS,
    HAS_FILL, HAS_DEPTH, HAS_REDUX, HAS_KONTEXT, HAS_QWEN, HAS_FLUX2, HAS_ZIMAGE,
    resolve_model_alias,
    get_lora_info,
    generate_image,
    save_images_with_metadata,
)
from .Mflux_Pro import (
    MfluxFillPipeline,
    MfluxImageRefPipeline,
    _tensor_to_temp_path,
)

# Nur importieren wenn verfügbar
if HAS_FILL:
    from .Mflux_Core import generate_fill
if HAS_DEPTH:
    from .Mflux_Core import generate_depth
if HAS_REDUX:
    from .Mflux_Core import generate_redux
if HAS_KONTEXT:
    from .Mflux_Core import generate_kontext
if HAS_QWEN:
    from .Mflux_Core import generate_qwen, generate_qwen_edit
if HAS_FLUX2:
    from mflux.models.flux2.variants.txt2img.flux2 import Flux2
if HAS_ZIMAGE:
    from mflux.models.z_image.variants.z_image import ZImage

# ---------------------------------------------------------------------------
# Konstanten
# ---------------------------------------------------------------------------
ALL_MODEL_ALIASES = list(MODEL_FAMILY_MAP.keys())

DOWNLOADER_MODELS = {
    "flux.1-schnell-mflux-4bit":  "madroid/flux.1-schnell-mflux-4bit",
    "flux.1-dev-mflux-4bit":      "madroid/flux.1-dev-mflux-4bit",
    "MFLUX.1-schnell-8-bit":      "AITRADER/MFLUX.1-schnell-8-bit",
    "MFLUX.1-dev-8-bit":          "AITRADER/MFLUX.1-dev-8-bit",
}
if HAS_KONTEXT:
    DOWNLOADER_MODELS["FLUX.1-Kontext-dev-4bit"] = "akx/FLUX.1-Kontext-dev-mflux-4bit"
if HAS_FLUX2 or True:  # Alias funktioniert auch über Flux1
    DOWNLOADER_MODELS["flux2-klein-4b-4bit"] = "madroid/flux2-klein-4b-mflux-4bit"
    DOWNLOADER_MODELS["flux2-klein-9b-4bit"] = "madroid/flux2-klein-9b-mflux-4bit"
if HAS_ZIMAGE:
    DOWNLOADER_MODELS["Z-Image-Turbo-4bit"] = "filipstrand/Z-Image-Turbo-mflux-4bit"


def create_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
    return directory


mflux_dir = os.path.join(models_dir, "Mflux")
create_directory(mflux_dir)


def get_full_model_path(model_dir, model_name):
    return os.path.join(model_dir, model_name)


def download_hg_model(model_key):
    repo_id = DOWNLOADER_MODELS.get(model_key)
    if not repo_id:
        print(f"Unknown model key: {model_key}")
        return None
    model_checkpoint = get_full_model_path(mflux_dir, model_key)
    if not os.path.exists(model_checkpoint):
        print(f"Downloading {model_key} from {repo_id} → {model_checkpoint} …")
        try:
            snapshot_download(repo_id=repo_id, local_dir=model_checkpoint)
        except Exception as e:
            print(f"Error downloading {model_key}: {e}")
            return None
    else:
        print(f"Model {model_key} already exists. Skipping.")
    return model_checkpoint


# ---------------------------------------------------------------------------
# Node: MfluxModelsDownloader
# ---------------------------------------------------------------------------
class MfluxModelsDownloader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_version": (list(DOWNLOADER_MODELS.keys()),
                                  {"default": "flux.1-schnell-mflux-4bit"}),
            }
        }
    RETURN_TYPES = ("PATH",)
    RETURN_NAMES = ("Downloaded_model",)
    CATEGORY = "MFlux/Air"
    FUNCTION = "download_model"

    def download_model(self, model_version):
        model_path = download_hg_model(model_version)
        return (model_path,) if model_path else (None,)


# ---------------------------------------------------------------------------
# Node: MfluxCustomModels
# ---------------------------------------------------------------------------
class MfluxCustomModels:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model":    (ALL_MODEL_ALIASES, {"default": "schnell"}),
                "quantize": (["3", "4", "5", "6", "8"], {"default": "4"}),
            },
            "optional": {
                "Loras": ("MfluxLorasPipeline",),
                "custom_identifier": ("STRING", {"default": ""}),
            },
        }
    RETURN_TYPES = ("PATH",)
    RETURN_NAMES = ("Custom_model",)
    CATEGORY = "MFlux/Air"
    FUNCTION = "save_model"

    def save_model(self, model, quantize, Loras=None, custom_identifier=""):
        identifier = custom_identifier if custom_identifier else "default"
        save_dir = get_full_model_path(mflux_dir, f"Mflux-{model}-{quantize}bit-{identifier}")
        create_directory(save_dir)
        lora_paths, lora_scales = get_lora_info(Loras)
        family, alias = resolve_model_alias(model, "")
        model_config = ModelConfig.from_alias(alias)
        q = int(quantize)

        if family == "flux2" and HAS_FLUX2:
            inst = Flux2(model_config=model_config, quantize=q,
                         lora_paths=lora_paths, lora_scales=lora_scales)
        elif family == "zimage" and HAS_ZIMAGE:
            inst = ZImage(model_config=model_config, quantize=q,
                          lora_paths=lora_paths, lora_scales=lora_scales)
        else:
            inst = Flux1(model_config=model_config, quantize=q,
                         lora_paths=lora_paths, lora_scales=lora_scales)

        inst.save_model(save_dir)
        print(f"Model saved to {save_dir}")
        return (save_dir,)


# ---------------------------------------------------------------------------
# Node: MfluxModelsLoader
# ---------------------------------------------------------------------------
class MfluxModelsLoader:
    @classmethod
    def INPUT_TYPES(cls):
        available = cls.get_sorted_model_paths()
        default = available[0] if available else "None"
        return {
            "required": {
                "model_name": (available or ["None"], {"default": default}),
            },
            "optional": {
                "free_path": ("STRING", {"default": ""}),
            },
        }
    RETURN_TYPES = ("PATH",)
    RETURN_NAMES = ("Local_model",)
    CATEGORY = "MFlux/Air"
    FUNCTION = "load"

    @classmethod
    def get_sorted_model_paths(cls):
        return sorted(
            [p.name for p in Path(mflux_dir).iterdir() if p.is_dir()],
            key=lambda x: x.lower(),
        )

    def load(self, model_name="", free_path=""):
        if free_path:
            if not os.path.exists(free_path):
                raise ValueError(f"Path does not exist: {free_path}")
            return (free_path,)
        if model_name and model_name != "None":
            return (get_full_model_path(mflux_dir, model_name),)
        raise ValueError("Either 'model_name' or 'free_path' must be provided.")


# ---------------------------------------------------------------------------
# Node: QuickMfluxNode
# ---------------------------------------------------------------------------
class QuickMfluxNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt":   ("STRING", {"multiline": True, "dynamicPrompts": True,
                                        "default": "Luxury food photograph"}),
                "model":    (ALL_MODEL_ALIASES, {"default": "schnell"}),
                "quantize": (ALL_QUANTIZE_OPTIONS, {"default": "4"}),
                "seed":     ("INT", {"default": -1, "min": -1, "max": 0xFFFFFFFFFFFFFFFF}),
                "width":    ("INT", {"default": 512}),
                "height":   ("INT", {"default": 512}),
                "steps":    ("INT", {"default": 2, "min": 1}),
                "guidance": ("FLOAT", {"default": 3.5, "min": 0.0}),
                "metadata": ("BOOLEAN", {"default": True, "label_on": "True", "label_off": "False"}),
            },
            "optional": {
                "Local_model": ("PATH",),
                "Loras":       ("MfluxLorasPipeline",),
                "img2img":     ("MfluxImg2ImgPipeline",),
                "ControlNet":  ("MfluxControlNetPipeline",),
            },
            "hidden": {
                "full_prompt":   "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO",
            },
        }
    RETURN_TYPES = ("IMAGE",)
    CATEGORY = "MFlux/Air"
    FUNCTION = "generate"

    def generate(self, prompt, model, seed, width, height, steps, guidance,
                 quantize="None", metadata=True, Local_model="",
                 img2img=None, Loras=None, ControlNet=None,
                 full_prompt=None, extra_pnginfo=None):
        generated = generate_image(
            prompt, model, seed, width, height, steps, guidance,
            quantize, metadata, Local_model, img2img, Loras, ControlNet,
        )
        if metadata:
            image_path     = img2img.image_path     if img2img else None
            image_strength = img2img.image_strength if img2img else None
            lora_paths, lora_scales = get_lora_info(Loras)
            _, alias = resolve_model_alias(model, Local_model)
            save_images_with_metadata(
                images=generated, prompt=prompt, model=alias, quantize=quantize,
                Local_model=Local_model, seed=seed, height=height, width=width,
                steps=steps, guidance=guidance, image_path=image_path,
                image_strength=image_strength, lora_paths=lora_paths,
                lora_scales=lora_scales, filename_prefix="Mflux",
                full_prompt=full_prompt, extra_pnginfo=extra_pnginfo,
            )
        return generated


# ---------------------------------------------------------------------------
# Nodes für neue Varianten – nur definieren wenn Abhängigkeit vorhanden
# ---------------------------------------------------------------------------

if HAS_FILL:
    class MfluxFillNode:
        @classmethod
        def INPUT_TYPES(cls):
            return {
                "required": {
                    "prompt":   ("STRING", {"multiline": True, "default": ""}),
                    "fill":     ("MfluxFillPipeline",),
                    "quantize": (ALL_QUANTIZE_OPTIONS, {"default": "4"}),
                    "seed":     ("INT", {"default": -1, "min": -1, "max": 0xFFFFFFFFFFFFFFFF}),
                    "width":    ("INT", {"default": 512}),
                    "height":   ("INT", {"default": 512}),
                    "steps":    ("INT", {"default": 30, "min": 1}),
                    "guidance": ("FLOAT", {"default": 30.0, "min": 0.0}),
                    "metadata": ("BOOLEAN", {"default": True, "label_on": "True", "label_off": "False"}),
                },
                "optional": {
                    "Local_model": ("PATH",),
                    "Loras":       ("MfluxLorasPipeline",),
                },
                "hidden": {"full_prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
            }
        RETURN_TYPES = ("IMAGE",)
        CATEGORY = "MFlux/Air"
        FUNCTION = "run"

        def run(self, prompt, fill, quantize, seed, width, height, steps, guidance,
                metadata=True, Local_model="", Loras=None,
                full_prompt=None, extra_pnginfo=None):
            generated = generate_fill(
                prompt, seed, width, height, steps, guidance, quantize,
                fill.image_path, fill.mask_path, Local_model, Loras,
            )
            if metadata:
                lora_paths, lora_scales = get_lora_info(Loras)
                save_images_with_metadata(
                    images=generated, prompt=prompt, model="dev", quantize=quantize,
                    Local_model=Local_model, seed=seed, height=height, width=width,
                    steps=steps, guidance=guidance, image_path=fill.image_path,
                    image_strength=None, lora_paths=lora_paths, lora_scales=lora_scales,
                    filename_prefix="Mflux_Fill", full_prompt=full_prompt,
                    extra_pnginfo=extra_pnginfo,
                    extra_meta={"mask_path": fill.mask_path, "variant": "fill"},
                )
            return generated

if HAS_DEPTH:
    class MfluxDepthNode:
        @classmethod
        def INPUT_TYPES(cls):
            return {
                "required": {
                    "prompt":    ("STRING", {"multiline": True, "default": ""}),
                    "image_ref": ("MfluxImageRefPipeline",),
                    "quantize":  (ALL_QUANTIZE_OPTIONS, {"default": "4"}),
                    "seed":      ("INT", {"default": -1, "min": -1, "max": 0xFFFFFFFFFFFFFFFF}),
                    "width":     ("INT", {"default": 512}),
                    "height":    ("INT", {"default": 512}),
                    "steps":     ("INT", {"default": 25, "min": 1}),
                    "guidance":  ("FLOAT", {"default": 10.0, "min": 0.0}),
                    "metadata":  ("BOOLEAN", {"default": True, "label_on": "True", "label_off": "False"}),
                },
                "optional": {
                    "Local_model": ("PATH",),
                    "Loras":       ("MfluxLorasPipeline",),
                },
                "hidden": {"full_prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
            }
        RETURN_TYPES = ("IMAGE",)
        CATEGORY = "MFlux/Air"
        FUNCTION = "run"

        def run(self, prompt, image_ref, quantize, seed, width, height, steps, guidance,
                metadata=True, Local_model="", Loras=None,
                full_prompt=None, extra_pnginfo=None):
            generated = generate_depth(
                prompt, seed, width, height, steps, guidance, quantize,
                image_ref.image_path, Local_model, Loras,
            )
            if metadata:
                lora_paths, lora_scales = get_lora_info(Loras)
                save_images_with_metadata(
                    images=generated, prompt=prompt, model="dev", quantize=quantize,
                    Local_model=Local_model, seed=seed, height=height, width=width,
                    steps=steps, guidance=guidance, image_path=image_ref.image_path,
                    image_strength=None, lora_paths=lora_paths, lora_scales=lora_scales,
                    filename_prefix="Mflux_Depth", full_prompt=full_prompt,
                    extra_pnginfo=extra_pnginfo, extra_meta={"variant": "depth"},
                )
            return generated

if HAS_REDUX:
    class MfluxReduxNode:
        @classmethod
        def INPUT_TYPES(cls):
            return {
                "required": {
                    "image_ref": ("MfluxImageRefPipeline",),
                    "quantize":  (ALL_QUANTIZE_OPTIONS, {"default": "4"}),
                    "seed":      ("INT", {"default": -1, "min": -1, "max": 0xFFFFFFFFFFFFFFFF}),
                    "width":     ("INT", {"default": 512}),
                    "height":    ("INT", {"default": 512}),
                    "steps":     ("INT", {"default": 25, "min": 1}),
                    "guidance":  ("FLOAT", {"default": 3.5, "min": 0.0}),
                    "metadata":  ("BOOLEAN", {"default": True, "label_on": "True", "label_off": "False"}),
                },
                "optional": {"Local_model": ("PATH",)},
                "hidden": {"full_prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
            }
        RETURN_TYPES = ("IMAGE",)
        CATEGORY = "MFlux/Air"
        FUNCTION = "run"

        def run(self, image_ref, quantize, seed, width, height, steps, guidance,
                metadata=True, Local_model="", full_prompt=None, extra_pnginfo=None):
            generated = generate_redux(
                seed, width, height, steps, guidance, quantize,
                image_ref.image_path, Local_model,
            )
            if metadata:
                save_images_with_metadata(
                    images=generated, prompt="[redux]", model="dev", quantize=quantize,
                    Local_model=Local_model, seed=seed, height=height, width=width,
                    steps=steps, guidance=guidance, image_path=image_ref.image_path,
                    image_strength=None, lora_paths=[], lora_scales=[],
                    filename_prefix="Mflux_Redux", full_prompt=full_prompt,
                    extra_pnginfo=extra_pnginfo, extra_meta={"variant": "redux"},
                )
            return generated

if HAS_KONTEXT:
    class MfluxKontextNode:
        @classmethod
        def INPUT_TYPES(cls):
            return {
                "required": {
                    "prompt":    ("STRING", {"multiline": True,
                                             "default": "Make the background a sunset beach"}),
                    "image_ref": ("MfluxImageRefPipeline",),
                    "quantize":  (ALL_QUANTIZE_OPTIONS, {"default": "4"}),
                    "seed":      ("INT", {"default": -1, "min": -1, "max": 0xFFFFFFFFFFFFFFFF}),
                    "width":     ("INT", {"default": 512}),
                    "height":    ("INT", {"default": 512}),
                    "steps":     ("INT", {"default": 25, "min": 1}),
                    "guidance":  ("FLOAT", {"default": 2.5, "min": 0.0}),
                    "metadata":  ("BOOLEAN", {"default": True, "label_on": "True", "label_off": "False"}),
                },
                "optional": {
                    "Local_model": ("PATH",),
                    "Loras":       ("MfluxLorasPipeline",),
                },
                "hidden": {"full_prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
            }
        RETURN_TYPES = ("IMAGE",)
        CATEGORY = "MFlux/Air"
        FUNCTION = "run"

        def run(self, prompt, image_ref, quantize, seed, width, height, steps, guidance,
                metadata=True, Local_model="", Loras=None,
                full_prompt=None, extra_pnginfo=None):
            generated = generate_kontext(
                prompt, seed, width, height, steps, guidance, quantize,
                image_ref.image_path, Local_model, Loras,
            )
            if metadata:
                lora_paths, lora_scales = get_lora_info(Loras)
                save_images_with_metadata(
                    images=generated, prompt=prompt, model="kontext-dev", quantize=quantize,
                    Local_model=Local_model, seed=seed, height=height, width=width,
                    steps=steps, guidance=guidance, image_path=image_ref.image_path,
                    image_strength=None, lora_paths=lora_paths, lora_scales=lora_scales,
                    filename_prefix="Mflux_Kontext", full_prompt=full_prompt,
                    extra_pnginfo=extra_pnginfo, extra_meta={"variant": "kontext"},
                )
            return generated

if HAS_QWEN:
    class MfluxQwenNode:
        @classmethod
        def INPUT_TYPES(cls):
            return {
                "required": {
                    "prompt":          ("STRING", {"multiline": True,
                                                   "default": "A majestic tiger, photorealistic"}),
                    "negative_prompt": ("STRING", {"multiline": True,
                                                   "default": "blurry, low quality"}),
                    "quantize": (ALL_QUANTIZE_OPTIONS, {"default": "8"}),
                    "seed":     ("INT", {"default": -1, "min": -1, "max": 0xFFFFFFFFFFFFFFFF}),
                    "width":    ("INT", {"default": 1024}),
                    "height":   ("INT", {"default": 1024}),
                    "steps":    ("INT", {"default": 30, "min": 1}),
                    "guidance": ("FLOAT", {"default": 5.0, "min": 0.0}),
                    "metadata": ("BOOLEAN", {"default": True, "label_on": "True", "label_off": "False"}),
                },
                "optional": {
                    "Local_model": ("PATH",),
                    "Loras":       ("MfluxLorasPipeline",),
                },
                "hidden": {"full_prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
            }
        RETURN_TYPES = ("IMAGE",)
        CATEGORY = "MFlux/Air"
        FUNCTION = "run"

        def run(self, prompt, negative_prompt, quantize, seed, width, height, steps,
                guidance, metadata=True, Local_model="", Loras=None,
                full_prompt=None, extra_pnginfo=None):
            generated = generate_qwen(
                prompt, negative_prompt, seed, width, height, steps, guidance,
                quantize, Local_model, Loras,
            )
            if metadata:
                lora_paths, lora_scales = get_lora_info(Loras)
                save_images_with_metadata(
                    images=generated, prompt=prompt, model="qwen-image", quantize=quantize,
                    Local_model=Local_model, seed=seed, height=height, width=width,
                    steps=steps, guidance=guidance, image_path=None, image_strength=None,
                    lora_paths=lora_paths, lora_scales=lora_scales,
                    filename_prefix="Mflux_Qwen", full_prompt=full_prompt,
                    extra_pnginfo=extra_pnginfo,
                    extra_meta={"negative_prompt": negative_prompt, "variant": "qwen"},
                )
            return generated

    class MfluxQwenEditNode:
        @classmethod
        def INPUT_TYPES(cls):
            return {
                "required": {
                    "prompt":    ("STRING", {"multiline": True,
                                             "default": "Change the background to a snowy mountain"}),
                    "image_ref": ("MfluxImageRefPipeline",),
                    "quantize":  (ALL_QUANTIZE_OPTIONS, {"default": "8"}),
                    "seed":      ("INT", {"default": -1, "min": -1, "max": 0xFFFFFFFFFFFFFFFF}),
                    "width":     ("INT", {"default": 1024}),
                    "height":    ("INT", {"default": 1024}),
                    "steps":     ("INT", {"default": 30, "min": 1}),
                    "guidance":  ("FLOAT", {"default": 5.0, "min": 0.0}),
                    "metadata":  ("BOOLEAN", {"default": True, "label_on": "True", "label_off": "False"}),
                },
                "optional": {
                    "Local_model": ("PATH",),
                    "Loras":       ("MfluxLorasPipeline",),
                },
                "hidden": {"full_prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
            }
        RETURN_TYPES = ("IMAGE",)
        CATEGORY = "MFlux/Air"
        FUNCTION = "run"

        def run(self, prompt, image_ref, quantize, seed, width, height, steps, guidance,
                metadata=True, Local_model="", Loras=None,
                full_prompt=None, extra_pnginfo=None):
            generated = generate_qwen_edit(
                prompt, seed, width, height, steps, guidance, quantize,
                image_ref.image_paths, Local_model, Loras,
            )
            if metadata:
                lora_paths, lora_scales = get_lora_info(Loras)
                save_images_with_metadata(
                    images=generated, prompt=prompt, model="qwen-image", quantize=quantize,
                    Local_model=Local_model, seed=seed, height=height, width=width,
                    steps=steps, guidance=guidance, image_path=image_ref.image_path,
                    image_strength=None, lora_paths=lora_paths, lora_scales=lora_scales,
                    filename_prefix="Mflux_QwenEdit", full_prompt=full_prompt,
                    extra_pnginfo=extra_pnginfo,
                    extra_meta={"image_paths": image_ref.image_paths, "variant": "qwen-edit"},
                )
            return generated
