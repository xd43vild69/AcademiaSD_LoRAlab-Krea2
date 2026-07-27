# -*- coding: utf-8 -*-
"""
caption_qwen3vl.py — Auto-captioning con el mismo Qwen3-VL-4B que Krea 2 usa como text encoder

Carga `Krea-2-NF4/text_encoder/` (config.json + model.safetensors) por la vía de
generación de `transformers` en vez de la vía de embeddings de `diffusers`. El
checkpoint tiene `tie_word_embeddings: true`, así que aunque no traiga
`lm_head.weight`, `model.tie_weights()` lo materializa a partir de
`embed_tokens.weight` y la generación de texto es real (no un truco).

Técnica de carga adaptada de Fizgig (Fizgig/src/fizgig/krea2/embedder.py), que
resuelve el mismo problema para el entrenador Klein/Krea2. Aquí se simplifica
porque `config.json` y el tokenizer YA están en disco (Fizgig los vendoriza
porque no los tiene) — solo el `AutoProcessor` (preprocesado de imagen + chat
template) se trae de Hugging Face, una vez, y queda cacheado en HF_HOME.
"""
import os

import torch
from safetensors.torch import load_file
from transformers import AutoProcessor, Qwen3VLConfig, Qwen3VLForConditionalGeneration

PROCESSOR_REPO_ID = "Qwen/Qwen3-VL-4B-Instruct"

# Instrucciones de captioning (Fizgig embedder.py:173-190, verbatim). El estilo
# conciso es el default de la UI; el detallado es opt-in (checkbox "Detallado").
CAPTION_INSTRUCTION = (
    "Write one factual training caption for this image as a single sentence. Describe the subject, "
    "their pose and clothing, the camera viewpoint (e.g. 'viewed from behind', 'side profile', "
    "'close-up'), whether the face is visible, and the setting. State only what is visible — no "
    "speculation, no names, no style commentary."
)

DETAILED_CAPTION_INSTRUCTION = (
    "Write a detailed factual training caption for this image, 2-4 sentences. Cover: the subject "
    "and exactly how much of them is visible (state the camera viewpoint and explicitly whether "
    "the face is visible or hidden), their pose and body position, every visible clothing item "
    "with colors, hair style and color, any objects they hold or touch, anything partially "
    "blocking or cropping the subject, the lighting, and the background/setting with its main "
    "objects. State only what is visible — no speculation, no names, no style commentary."
)


def _convert_bare_state_dict(sd):
    """Remapea claves 'bare' (language_model.*/visual.*, como vienen en nuestro
    safetensors local) al layout que espera Qwen3VLForConditionalGeneration
    (model.language_model.*/model.visual.*). Misma lógica que
    _convert_comfyui_qwen3vl_state_dict de Fizgig, portada 1:1."""
    converted = {}
    for key, value in sd.items():
        if key.startswith("model.language_model.") or key.startswith("model.visual."):
            new_key = key
        elif key.startswith("visual."):
            new_key = "model.visual." + key[len("visual."):]
        elif key.startswith("language_model."):
            new_key = "model." + key
        elif key.startswith("model."):
            new_key = "model.language_model." + key[len("model."):]
        else:
            new_key = key
        converted[new_key] = value
    return converted


def load_captioner(text_encoder_dir, device="cuda", dtype=torch.bfloat16,
                   processor_repo=PROCESSOR_REPO_ID):
    """Carga el Qwen3-VL-4B local en modo generación + su AutoProcessor.

    `text_encoder_dir` es la carpeta `Krea-2-NF4/text_encoder/` (contiene
    config.json + model.safetensors). Devuelve (model, processor).
    """
    from accelerate import init_empty_weights

    config_path = os.path.join(text_encoder_dir, "config.json")
    weights_path = os.path.join(text_encoder_dir, "model.safetensors")
    if not os.path.isfile(config_path) or not os.path.isfile(weights_path):
        raise FileNotFoundError(
            f"No se encontró config.json / model.safetensors en {text_encoder_dir}"
        )

    config = Qwen3VLConfig.from_pretrained(text_encoder_dir)
    with init_empty_weights():
        model = Qwen3VLForConditionalGeneration._from_config(config)

    sd = load_file(weights_path, device="cpu")
    sd = _convert_bare_state_dict(sd)
    if dtype is not None:
        sd = {k: v.to(dtype) for k, v in sd.items()}

    info = model.load_state_dict(sd, strict=False, assign=True)
    # tie_word_embeddings=true: el checkpoint omite lm_head.weight a propósito;
    # tie_weights() lo materializa desde embed_tokens tras cargar el resto.
    model.tie_weights()

    unexpected = list(info.unexpected_keys)
    missing = [k for k in info.missing_keys if k != "lm_head.weight"]
    if unexpected or missing:
        raise RuntimeError(
            f"El checkpoint del text encoder no coincide con el modelo: "
            f"missing={missing[:10]}, unexpected={unexpected[:10]}"
        )

    model.to(device)
    if dtype is not None:
        model.to(dtype)
    model = model.eval().requires_grad_(False)

    processor = AutoProcessor.from_pretrained(processor_repo)

    return model, processor


def _cap_image(im, megapixels=1.0):
    """RGB + downscale para que el área de píxeles no supere `megapixels` (nunca agranda)."""
    im = im.convert("RGB")
    cap = int(megapixels * 1024 * 1024)
    w, h = im.size
    if w * h > cap and w > 0 and h > 0:
        scale = (cap / (w * h)) ** 0.5
        from PIL import Image
        im = im.resize((max(1, round(w * scale)), max(1, round(h * scale))), Image.LANCZOS)
    return im


def generate_caption(model, processor, image_path, *, max_new_tokens=120,
                     megapixels=1.0, detailed=False, seed=None):
    """Genera un caption factual para `image_path`. Decoding sampleado
    (temperature=0.5, top_p=0.9) con semilla aleatoria por llamada, así que
    reintentos sobre la misma imagen dan redacciones distintas. Guarda y
    restaura el RNG global de torch alrededor de la llamada (higiene, aunque
    aquí no hay training corriendo en paralelo que proteger)."""
    import random as _random

    from PIL import Image

    im = _cap_image(Image.open(image_path), megapixels)
    instruction = DETAILED_CAPTION_INSTRUCTION if detailed else CAPTION_INSTRUCTION
    if detailed:
        max_new_tokens = max(max_new_tokens, 240)

    messages = [{"role": "user", "content": [{"type": "image"},
                                             {"type": "text", "text": instruction}]}]
    prompt = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    inputs = processor(text=[prompt], images=[im], return_tensors="pt").to(model.device)

    cpu_state = torch.random.get_rng_state()
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    try:
        torch.manual_seed(seed if seed is not None else _random.randint(1, 2**31 - 1))
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=max_new_tokens,
                                 do_sample=True, temperature=0.5, top_p=0.9)
    finally:
        torch.random.set_rng_state(cpu_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)

    text = processor.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]
    return " ".join(text.split()).strip()
