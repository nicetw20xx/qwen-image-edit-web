import logging
import math
import os
import time
import threading
from io import BytesIO
from pathlib import Path
from urllib.parse import urlparse, unquote

import torch
from PIL import Image
from flask import Flask, request, jsonify, send_from_directory

OUTPUTS_DIR = Path(__file__).parent / "outputs"
STATIC_DIR = Path(__file__).parent / "static"
OUTPUTS_DIR.mkdir(exist_ok=True)

LORAS = [
    "https://huggingface.co/lightx2v/Qwen-Image-Lightning/blob/main/Qwen-Image-Edit-2509/Qwen-Image-Edit-2509-Lightning-4steps-V1.0-bf16.safetensors",
    "https://huggingface.co/dx8152/Qwen-Image-Edit-2509-Fusion/blob/main/%E6%BA%B6%E5%9B%BE.safetensors",
]

QUEUE_LIMIT = 10
queue_lock = threading.Lock()
queue_count = 0
inference_semaphore = threading.Semaphore(1)

pipeline = None
loaded_lora_adapters = {}
original_scheduler = None
lightning_scheduler = None

app = Flask(__name__)


class SuppressRequestLogFilter(logging.Filter):
    def filter(self, record):
        return False


logging.getLogger("werkzeug").addFilter(SuppressRequestLogFilter())


def load_model():
    global pipeline, original_scheduler, lightning_scheduler
    from diffusers import QwenImageEditPlusPipeline, FlowMatchEulerDiscreteScheduler
    pipeline = QwenImageEditPlusPipeline.from_pretrained(
        "Qwen/Qwen-Image-Edit-2509",
        torch_dtype=torch.bfloat16,
    ).to("cuda")
    original_scheduler = pipeline.scheduler
    lightning_scheduler = FlowMatchEulerDiscreteScheduler.from_config({
        "base_image_seq_len": 256,
        "base_shift": math.log(3),
        "invert_sigmas": False,
        "max_image_seq_len": 8192,
        "max_shift": math.log(3),
        "num_train_timesteps": 1000,
        "shift": 1.0,
        "shift_terminal": None,
        "stochastic_sampling": False,
        "time_shift_type": "exponential",
        "use_beta_sigmas": False,
        "use_dynamic_shifting": True,
        "use_exponential_sigmas": False,
        "use_karras_sigmas": False,
    })


def parse_huggingface_url(url):
    parsed = urlparse(url)
    path_parts = parsed.path.strip("/").split("/")
    # e.g. lightx2v/Qwen-Image-Lightning/blob/main/Qwen-Image-Edit-2509/file.safetensors
    if len(path_parts) < 5 or path_parts[2] != "blob":
        return None
    repo_id = f"{path_parts[0]}/{path_parts[1]}"
    # skip "blob" and branch name (parts[2], parts[3])
    remaining = path_parts[4:]
    weight_name = unquote(remaining[-1])
    subfolder = "/".join(remaining[:-1]) if len(remaining) > 1 else None
    return repo_id, subfolder, weight_name


def ensure_lora_loaded(lora_url):
    global loaded_lora_adapters
    if lora_url in loaded_lora_adapters:
        return loaded_lora_adapters[lora_url]

    parsed = parse_huggingface_url(lora_url)
    if not parsed:
        return None

    repo_id, subfolder, weight_name = parsed
    adapter_name = weight_name.replace(".safetensors", "").replace(".", "_")

    load_kwargs = {"weight_name": weight_name, "adapter_name": adapter_name}
    if subfolder:
        load_kwargs["subfolder"] = subfolder
    pipeline.load_lora_weights(repo_id, **load_kwargs)

    loaded_lora_adapters[lora_url] = adapter_name
    return adapter_name


def round_to_multiple_of_16(value):
    return (value // 16) * 16


def run_inference(request_id, prompt, images, lora_urls, guidance_scale, true_cfg_scale, num_inference_steps):
    has_lora = bool(lora_urls)

    if has_lora:
        adapter_names = []
        for url in lora_urls:
            name = ensure_lora_loaded(url)
            if name is None:
                raise ValueError(f"Invalid HuggingFace LoRA URL format: {url}")
            adapter_names.append(name)
        pipeline.set_adapters(adapter_names)
        pipeline.enable_lora()
        pipeline.scheduler = lightning_scheduler
    else:
        if loaded_lora_adapters:
            pipeline.disable_lora()
        pipeline.scheduler = original_scheduler

    processed_images = []
    for img in images:
        w, h = img.size
        new_w = round_to_multiple_of_16(w)
        new_h = round_to_multiple_of_16(h)
        if new_w != w or new_h != h:
            img = img.resize((new_w, new_h), Image.LANCZOS)
        processed_images.append(img)

    generate_kwargs = {
        "prompt": prompt,
        "negative_prompt": " ",
        "guidance_scale": guidance_scale,
        "true_cfg_scale": true_cfg_scale,
        "num_inference_steps": num_inference_steps,
    }
    if processed_images:
        generate_kwargs["image"] = processed_images

    result = pipeline(**generate_kwargs)
    output_image = result.images[0]

    filename = f"{request_id}.png"
    output_path = OUTPUTS_DIR / filename
    output_image.save(output_path)
    print(f"Generated {filename} | prompt: {prompt}, loras: {lora_urls}, guidance_scale: {guidance_scale}, true_cfg_scale: {true_cfg_scale}, num_inference_steps: {num_inference_steps}")


def build_lora_filename_map():
    mapping = {}
    for url in LORAS:
        parsed = parse_huggingface_url(url)
        raw_filename = parsed[2] if parsed else url.split("/")[-1]
        mapping[raw_filename] = url
    return mapping


LORA_FILENAME_TO_URL = build_lora_filename_map()


@app.route("/api/loras")
def get_loras():
    return jsonify(list(LORA_FILENAME_TO_URL.keys()))
    # return jsonify([])


def run_inference_background(request_id, prompt, images, lora_urls, guidance_scale, true_cfg_scale, num_inference_steps):
    global queue_count
    try:
        inference_semaphore.acquire()
        try:
            run_inference(request_id, prompt, images, lora_urls, guidance_scale, true_cfg_scale, num_inference_steps)
        finally:
            inference_semaphore.release()
    except Exception as e:
        import traceback
        traceback.print_exc()
        error_path = OUTPUTS_DIR / f"{request_id}.error"
        error_path.write_text(str(e), encoding="utf-8")
    finally:
        with queue_lock:
            queue_count -= 1


@app.route("/api/images", methods=["POST"])
def generate_images():
    global queue_count

    prompt = request.form.get("prompt", "").strip()
    if not prompt:
        return jsonify({"error": "prompt is required"}), 400

    lora_filenames = request.form.getlist("lora")
    # lora_filenames = list(LORA_FILENAME_TO_URL.keys())
    lora_urls = []
    for lora_filename in lora_filenames:
        lora_filename = lora_filename.strip()
        if not lora_filename:
            continue
        lora_url = LORA_FILENAME_TO_URL.get(lora_filename)
        if not lora_url:
            return jsonify({"error": f"Unknown lora: {lora_filename}"}), 400
        lora_urls.append(lora_url)

    has_lora = bool(lora_urls)
    guidance_scale = float(request.form.get("guidance_scale", 1.0))
    true_cfg_scale = float(request.form.get("true_cfg_scale", 1.0 if has_lora else 4.0))
    num_inference_steps = int(request.form.get("num_inference_steps", 4 if has_lora else 40))
    num_inference_steps = max(1, min(100, num_inference_steps))

    with queue_lock:
        if queue_count >= QUEUE_LIMIT:
            return jsonify({"error": "Queue full, try again later"}), 429
        queue_count += 1

    images = []
    uploaded_files = request.files.getlist("images")
    for f in uploaded_files:
        img = Image.open(f.stream).convert("RGB")
        images.append(img)

    request_id = str(int(time.time() * 1000))

    thread = threading.Thread(
        target=run_inference_background,
        args=(request_id, prompt, images, lora_urls, guidance_scale, true_cfg_scale, num_inference_steps),
    )
    thread.start()

    return jsonify({"request_id": request_id})


@app.route("/api/images/<request_id>")
def poll_image(request_id):
    if not request_id.isdigit():
        return jsonify({"error": "invalid request_id"}), 400

    error_path = OUTPUTS_DIR / f"{request_id}.error"
    if error_path.exists():
        return jsonify({"status": "error", "error": error_path.read_text(encoding="utf-8")})

    image_path = OUTPUTS_DIR / f"{request_id}.png"
    if image_path.exists():
        return jsonify({"status": "done", "url": f"/outputs/{request_id}.png"})

    return jsonify({"status": "pending"})


@app.route("/outputs/<path:filename>")
def serve_output(filename):
    return send_from_directory(OUTPUTS_DIR, filename)


@app.route("/")
def serve_index():
    return send_from_directory(STATIC_DIR, "index.html")


@app.route("/<path:filename>")
def serve_static(filename):
    return send_from_directory(STATIC_DIR, filename)


if __name__ == "__main__":
    import sys
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 5000
    load_model()
    app.run(host="0.0.0.0", port=port, threaded=True)
