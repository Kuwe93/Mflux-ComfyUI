import random
import json
import os
import numpy as np
import torch
from PIL import Image
from PIL.PngImagePlugin import PngInfo
import folder_paths

# ---------------------------------------------------------------------------
# mflux 0.16.x Imports
# ---------------------------------------------------------------------------

from mflux.models.common.config import ModelConfig

# FLUX.1-Familie
from mflux.models.flux.variants.txt2img.flux import Flux1
from mflux.models.flux.variants.controlnet.flux_controlnet import Flux1Controlnet
from mflux.models.flux.variants.fill.flux_fill import Flux1Fill
from mflux.models.flux.variants.depth.flux_depth import Flux1Depth
from mflux.models.flux.variants.redux.flux_redux import Flux1Redux
from mflux.models.flux.variants.kontext.flux_kontext import Flux1Kontext

# FLUX.2-Familie
from mflux.models.flux2.variants.txt2img.flux2 import Flux2

# Z-Image-Familie
from mflux.models.z_image.variants.z_image import ZImage

# Qwen-Familie
from mflux.models.qwen.variants.txt2img.qwen_image import QwenImage
from mflux.models.qwen.variants.edit.qwen_image_edit import QwenImageEdit

# Post-Processing
from mflux.post_processing.image_util import ImageUtil
from mflux.controlnet.controlnet_util import ControlnetUtil

from .Mflux_Pro import MfluxControlNetPipeline

# ---------------------------------------------------------------------------
# Modell-Familien-Zuordnung
# ---------------------------------------------------------------------------
MODEL_FAMILY_MAP = {
    # FLUX.1
    "schnell":        ("flux1", "schnell"),
    "dev":            ("flux1", "dev"),
    "krea-dev":       ("flux1", "krea-dev"),
    "kontext-dev":    ("flux1", "kontext-dev"),
    # FLUX.2
    "flux2-klein-4b": ("flux2", "flux2-klein-4b"),
    "flux2-klein-9b": ("flux2", "flux2-klein-9b"),
    "flux2-base-4b":  ("flux2", "flux2-base-4b"),
    "flux2-base-9b":  ("flux2", "flux2-base-9b"),
    "flux2-dev":      ("flux2", "flux2-dev"),
    # Z-Image
    "z-image-turbo":  ("zimage", "z-image-turbo"),
    "z-image-base":   ("zimage", "z-image-base"),
    # Qwen
    "qwen-image":     ("qwen", "qwen-image"),
}

# Distilled FLUX.2-Modelle benötigen guidance=1.0
FLUX2_DISTILLED_MODELS = {"flux2-klein-4b", "flux2-klein-9b"}

# Alle Quantisierungsoptionen
ALL_QUANTIZE_OPTIONS = ["None", "3", "4", "5", "6", "8"]

# ---------------------------------------------------------------------------
# Model-Cache
# ---------------------------------------------------------------------------
_model_cache: dict = {}


def _evict_and_store(key, instance):
    _model_cache.clear()
    _model_cache[key] = instance
    return instance


# ---------------------------------------------------------------------------
# Hilfsfunktionen
# ---------------------------------------------------------------------------

def get_lora_info(Loras):
    if Loras:
        return Loras.lora_paths, Loras.lora_scales
    return [], []


def resolve_model_alias(model: str, local_path: str) -> tuple[str, str]:
    """Gibt (familie, alias) zurück."""
    if local_path:
        name_lower = local_path.lower()
        for alias, (family, _) in MODEL_FAMILY_MAP.items():
            if alias in name_lower or alias.replace("-", "_") in name_lower:
                return family, alias
        name = os.path.basename(local_path).lower()
        if "flux2" in name or "klein" in name:
            return "flux2", model
        if "z-image" in name or "zimage" in name:
            return "zimage", model
        if "qwen" in name:
            return "qwen", model
        return "flux1", model
    if model in MODEL_FAMILY_MAP:
        return MODEL_FAMILY_MAP[model]
    return "flux1", model


def _tensor_from_image(generated) -> torch.Tensor:
    """Wandelt ein mflux-Ergebnis in einen ComfyUI (B,H,W,C) Float32-Tensor um."""
    if hasattr(generated, "image"):
        arr = np.array(generated.image).astype(np.float32) / 255.0
    elif isinstance(generated, np.ndarray):
        arr = generated.astype(np.float32)
        if arr.max() > 1.0:
            arr /= 255.0
    else:
        arr = np.array(generated).astype(np.float32)
        if arr.max() > 1.0:
            arr /= 255.0
    t = torch.from_numpy(arr)
    if t.dim() == 3:
        t = t.unsqueeze(0)
    return t


def _save_temp_image(tensor: torch.Tensor) -> str:
    """Speichert einen ComfyUI-Tensor als temporäre PNG und gibt den Pfad zurück."""
    import tempfile
    arr = (tensor.squeeze(0).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
    img = Image.fromarray(arr)
    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    img.save(tmp.name)
    return tmp.name


# ---------------------------------------------------------------------------
# Modell laden / aus Cache holen
# ---------------------------------------------------------------------------

def load_or_create_model(model, quantize, local_path, lora_paths, lora_scales,
                         variant=""):
    family, alias = resolve_model_alias(model, local_path)
    q = None if quantize == "None" else int(quantize)
    path = local_path if local_path else None
    key = (family, alias, q, path or "", tuple(lora_paths), tuple(lora_scales), variant)

    if key in _model_cache:
        return _model_cache[key]

    std_kwargs = dict(
        model_config=ModelConfig.from_alias(alias),
        quantize=q,
        local_path=path,
        lora_paths=lora_paths,
        lora_scales=lora_scales,
    )
    # Varianten, die model_path statt local_path erwarten
    path_kwargs = dict(quantize=q, model_path=path,
                       lora_paths=lora_paths, lora_scales=lora_scales)

    if family == "flux1":
        if variant == "controlnet":
            inst = Flux1Controlnet(**std_kwargs)
        elif variant == "fill":
            inst = Flux1Fill(**path_kwargs)
        elif variant == "depth":
            inst = Flux1Depth(**path_kwargs)
        elif variant == "redux":
            inst = Flux1Redux(quantize=q, model_path=path)
        elif variant == "kontext":
            inst = Flux1Kontext(**path_kwargs)
        else:
            inst = Flux1(**std_kwargs)
    elif family == "flux2":
        inst = Flux2(**std_kwargs)
    elif family == "zimage":
        inst = ZImage(**std_kwargs)
    elif family == "qwen":
        if variant == "edit":
            inst = QwenImageEdit(**path_kwargs)
        else:
            inst = QwenImage(**path_kwargs)
    else:
        inst = Flux1(**std_kwargs)

    return _evict_and_store(key, inst)


# ---------------------------------------------------------------------------
# Generierungsfunktionen
# ---------------------------------------------------------------------------

def generate_image(prompt, model, seed, width, height, steps, guidance,
                   quantize="None", metadata=True, Local_model="",
                   image=None, Loras=None, ControlNet=None):
    """Standard txt2img / img2img / ControlNet für FLUX.1, FLUX.2, Z-Image."""
    seed = random.randint(0, 0xFFFFFFFFFFFFFFFF) if seed == -1 else int(seed)
    print(f"[Mflux] seed={seed}")

    lora_paths, lora_scales = get_lora_info(Loras)
    family, alias = resolve_model_alias(model, Local_model)
    image_path = image.image_path if image else None
    image_strength = image.image_strength if image else None
    use_controlnet = ControlNet is not None and isinstance(ControlNet, MfluxControlNetPipeline)

    if alias in FLUX2_DISTILLED_MODELS:
        guidance = 1.0

    if use_controlnet:
        from mflux.post_processing.array_util import ArrayUtil
        from mflux.models.flux.latent_creator.flux_latent_creator import FluxLatentCreator
        import mlx.core as mx
        from tqdm import tqdm
        import comfy.utils as utils

        inst = load_or_create_model(model, quantize, Local_model, lora_paths, lora_scales,
                                    variant="controlnet")
        from mflux.config.config import ConfigControlnet
        from mflux.config.runtime_config import RuntimeConfig
        config = RuntimeConfig(
            config=ConfigControlnet(
                num_inference_steps=steps, height=height, width=width,
                guidance=guidance, controlnet_strength=ControlNet.control_strength,
            ),
            model_config=inst.model_config,
        )
        time_steps = tqdm(range(config.num_inference_steps))
        control_image = ImageUtil.load_image(ControlNet.control_image_path)
        control_image = ControlnetUtil.scale_image(config.height, config.width, control_image)
        control_image = ControlnetUtil.preprocess_canny(control_image)
        controlnet_cond = ImageUtil.to_array(control_image)
        controlnet_cond = inst.vae.encode(controlnet_cond)
        controlnet_cond = (controlnet_cond / inst.vae.scaling_factor) + inst.vae.shift_factor
        controlnet_cond = ArrayUtil.pack_latents(controlnet_cond, config.height, config.width)
        latents = FluxLatentCreator.create(seed=seed, height=config.height, width=config.width)
        t5_tokens = inst.t5_tokenizer.tokenize(prompt)
        clip_tokens = inst.clip_tokenizer.tokenize(prompt)
        prompt_embeds = inst.t5_text_encoder.forward(t5_tokens)
        pooled_prompt_embeds = inst.clip_text_encoder.forward(clip_tokens)
        pbar = None
        for gen_step, t in enumerate(time_steps, 1):
            if gen_step == 2:
                pbar = utils.ProgressBar(total=steps)
            cb_samples, cb_single = inst.transformer_controlnet.forward(
                t=t, prompt_embeds=prompt_embeds,
                pooled_prompt_embeds=pooled_prompt_embeds,
                hidden_states=latents, controlnet_cond=controlnet_cond, config=config,
            )
            noise = inst.transformer.predict(
                t=t, prompt_embeds=prompt_embeds,
                pooled_prompt_embeds=pooled_prompt_embeds,
                hidden_states=latents, config=config,
                controlnet_block_samples=cb_samples,
                controlnet_single_block_samples=cb_single,
            )
            latents += noise * (config.sigmas[t + 1] - config.sigmas[t])
            if pbar:
                pbar.update(1)
            mx.eval(latents)
        if pbar:
            pbar.update(1)
        latents = ArrayUtil.unpack_latents(latents, config.height, config.width)
        decoded = inst.vae.decode(latents)
        arr = ImageUtil._to_numpy(ImageUtil._denormalize(decoded))
        tensor = torch.from_numpy(arr)
        if tensor.dim() == 3:
            tensor = tensor.unsqueeze(0)
        return (tensor,)

    inst = load_or_create_model(model, quantize, Local_model, lora_paths, lora_scales)
    generated = inst.generate_image(
        prompt=prompt, seed=seed, num_inference_steps=steps,
        width=width, height=height, guidance=guidance,
        init_image_path=image_path, init_image_strength=image_strength,
    )
    return (_tensor_from_image(generated),)


def generate_fill(prompt, seed, width, height, steps, guidance, quantize,
                  image_path, mask_path, Local_model="", Loras=None):
    """FLUX.1 Fill / Inpainting."""
    seed = random.randint(0, 0xFFFFFFFFFFFFFFFF) if seed == -1 else int(seed)
    print(f"[Mflux Fill] seed={seed}")
    lora_paths, lora_scales = get_lora_info(Loras)
    inst = load_or_create_model("dev", quantize, Local_model, lora_paths, lora_scales,
                                variant="fill")
    generated = inst.generate_image(
        prompt=prompt, seed=seed, num_inference_steps=steps,
        width=width, height=height, guidance=guidance,
        image_path=image_path, mask_path=mask_path,
    )
    return (_tensor_from_image(generated),)


def generate_depth(prompt, seed, width, height, steps, guidance, quantize,
                   image_path, Local_model="", Loras=None):
    """FLUX.1 Depth Conditioning."""
    seed = random.randint(0, 0xFFFFFFFFFFFFFFFF) if seed == -1 else int(seed)
    print(f"[Mflux Depth] seed={seed}")
    lora_paths, lora_scales = get_lora_info(Loras)
    inst = load_or_create_model("dev", quantize, Local_model, lora_paths, lora_scales,
                                variant="depth")
    generated = inst.generate_image(
        prompt=prompt, seed=seed, num_inference_steps=steps,
        width=width, height=height, guidance=guidance,
        image_path=image_path,
    )
    return (_tensor_from_image(generated),)


def generate_redux(seed, width, height, steps, guidance, quantize,
                   image_path, Local_model=""):
    """FLUX.1 Redux – Image Variation ohne Text-Prompt."""
    seed = random.randint(0, 0xFFFFFFFFFFFFFFFF) if seed == -1 else int(seed)
    print(f"[Mflux Redux] seed={seed}")
    inst = load_or_create_model("dev", quantize, Local_model, [], [], variant="redux")
    generated = inst.generate_image(
        seed=seed, num_inference_steps=steps,
        width=width, height=height, guidance=guidance,
        image_path=image_path,
    )
    return (_tensor_from_image(generated),)


def generate_kontext(prompt, seed, width, height, steps, guidance, quantize,
                     image_path, Local_model="", Loras=None):
    """FLUX.1 Kontext – image-guided Text-Editing."""
    seed = random.randint(0, 0xFFFFFFFFFFFFFFFF) if seed == -1 else int(seed)
    print(f"[Mflux Kontext] seed={seed}")
    lora_paths, lora_scales = get_lora_info(Loras)
    inst = load_or_create_model("kontext-dev", quantize, Local_model, lora_paths, lora_scales,
                                variant="kontext")
    generated = inst.generate_image(
        prompt=prompt, seed=seed, num_inference_steps=steps,
        width=width, height=height, guidance=guidance,
        image_path=image_path,
    )
    return (_tensor_from_image(generated),)


def generate_qwen(prompt, negative_prompt, seed, width, height, steps, guidance,
                  quantize, Local_model="", Loras=None):
    """Qwen Image txt2img mit Negativprompt."""
    seed = random.randint(0, 0xFFFFFFFFFFFFFFFF) if seed == -1 else int(seed)
    print(f"[Mflux Qwen] seed={seed}")
    lora_paths, lora_scales = get_lora_info(Loras)
    inst = load_or_create_model("qwen-image", quantize, Local_model, lora_paths, lora_scales)
    generated = inst.generate_image(
        prompt=prompt,
        negative_prompt=negative_prompt if negative_prompt.strip() else " ",
        seed=seed, num_inference_steps=steps,
        width=width, height=height, guidance=guidance,
    )
    return (_tensor_from_image(generated),)


def generate_qwen_edit(prompt, seed, width, height, steps, guidance, quantize,
                       image_paths: list, Local_model="", Loras=None):
    """Qwen Image Edit – semantisches Editing mit Referenzbildern."""
    seed = random.randint(0, 0xFFFFFFFFFFFFFFFF) if seed == -1 else int(seed)
    print(f"[Mflux Qwen Edit] seed={seed}, images={len(image_paths)}")
    lora_paths, lora_scales = get_lora_info(Loras)
    inst = load_or_create_model("qwen-image", quantize, Local_model, lora_paths, lora_scales,
                                variant="edit")
    generated = inst.generate_image(
        prompt=prompt, seed=seed, num_inference_steps=steps,
        width=width, height=height, guidance=guidance,
        image_paths=image_paths,
    )
    return (_tensor_from_image(generated),)


# ---------------------------------------------------------------------------
# Metadaten speichern
# ---------------------------------------------------------------------------

def save_images_with_metadata(
    images, prompt, model, quantize, Local_model,
    seed, height, width, steps, guidance,
    lora_paths, lora_scales, image_path, image_strength,
    filename_prefix="Mflux", full_prompt=None, extra_pnginfo=None,
    extra_meta=None,
):
    output_dir = folder_paths.get_output_directory()
    full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(
        filename_prefix, output_dir, images[0].shape[1], images[0].shape[0]
    )
    mflux_output_folder = os.path.join(full_output_folder, "MFlux")
    os.makedirs(mflux_output_folder, exist_ok=True)

    existing_counters = [
        int(f.split("_")[-1].split(".")[0])
        for f in os.listdir(mflux_output_folder)
        if f.startswith(filename_prefix) and f.endswith(".png")
    ]
    counter = max(existing_counters, default=0) + 1

    results = []
    for image in images:
        i = 255.0 * image.cpu().numpy().squeeze()
        img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))

        png_meta = None
        if full_prompt is not None or extra_pnginfo is not None:
            png_meta = PngInfo()
            if full_prompt is not None:
                png_meta.add_text("full_prompt", json.dumps(full_prompt))
            if extra_pnginfo is not None:
                for x in extra_pnginfo:
                    png_meta.add_text(x, json.dumps(extra_pnginfo[x]))

        image_file = f"{filename_prefix}_{counter:05}.png"
        img.save(os.path.join(mflux_output_folder, image_file), pnginfo=png_meta, compress_level=4)
        results.append({"filename": image_file, "subfolder": subfolder, "type": "output"})

        family, alias = resolve_model_alias(model, Local_model)
        json_dict = {
            "prompt": prompt, "model": model, "model_family": family,
            "model_alias": alias, "quantize": quantize, "seed": seed,
            "height": height, "width": width, "steps": steps,
            "guidance": 1.0 if alias in FLUX2_DISTILLED_MODELS else guidance,
            "Local_model": Local_model, "init_image_path": image_path,
            "init_image_strength": image_strength,
            "lora_paths": lora_paths, "lora_scales": lora_scales,
        }
        if extra_meta:
            json_dict.update(extra_meta)

        with open(os.path.join(mflux_output_folder, f"{filename_prefix}_{counter:05}.json"), "w") as f:
            json.dump(json_dict, f, indent=4)
        counter += 1

    return {"ui": {"images": results}, "counter": counter}
