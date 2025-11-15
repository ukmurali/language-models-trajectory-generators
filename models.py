import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import torch
from PIL import Image
from torchvision import transforms
from torchvision.utils import draw_bounding_boxes, draw_segmentation_masks

import config

# Ensure XMem is importable
sys.path.append("./XMem/")

from XMem.inference.inference_core import InferenceCore
from XMem.inference.interact.interactive_utils import (
    image_to_torch,                # (np.ndarray HxWx3 uint8) -> torch [1,3,H,W] float(0..1) on device
    index_numpy_to_one_hot_torch,  # (labels HxW uint8, K:int) -> [1,K,H,W] one-hot (CPU tensor)
    torch_prob_to_numpy_mask,      # ([1,K,H,W] prob) -> HxW uint8 (argmax)
)

# -----------------------------
# LangSAM: robust prediction
# -----------------------------


def get_langsam_output(image, model, segmentation_texts, segmentation_count):
    """
    image: PIL.Image or numpy.ndarray
    segmentation_texts: list[str]
    returns: (masks_tensor[N,H,W] float or empty), boxes(list), phrases(list)
    """

    # Normalize image to PIL RGB
    if isinstance(image, np.ndarray):
        if image.ndim == 3 and image.shape[2] == 4:
            image = image[:, :, :3]
        image = Image.fromarray(np.uint8(image))
    elif isinstance(image, Image.Image):
        image = image.convert("RGB")
    else:
        raise ValueError("❌ Unsupported image type for LangSAM")

    # Normalize prompts
    if isinstance(segmentation_texts, str):
        segmentation_texts = [segmentation_texts]
    elif isinstance(segmentation_texts, tuple):
        segmentation_texts = list(segmentation_texts)

    # Predict (LangSAM expects list of images)
    result = model.predict(
        [image],
        segmentation_texts,
        box_threshold=0.25,
        text_threshold=0.25,
    )

    # Unpack flexibly (different versions return different shapes)
    if isinstance(result, tuple):
        if len(result) == 4:
            masks, boxes, phrases, logits = result
        elif len(result) == 3:
            masks, boxes, phrases = result
            logits = None
        else:
            masks = result[0]
            boxes, phrases, logits = [], [], None
    else:
        masks = result
        boxes, phrases, logits = [], [], None

    # If batched results, pick the first item
    if isinstance(masks, list) and len(masks) > 0 and torch.is_tensor(masks[0]):
        masks = masks[0]
    if isinstance(boxes, list) and len(boxes) > 0 and isinstance(boxes[0], list):
        boxes = boxes[0]
    if isinstance(phrases, list) and len(phrases) > 0 and isinstance(phrases[0], list):
        phrases = phrases[0]

    # Create visualization (robust to N masks)
    num_masks = len(masks) if isinstance(masks, (list, tuple)) else (masks.shape[0] if torch.is_tensor(masks) else 0)
    fig, ax = plt.subplots(1, max(1, 1 + num_masks), figsize=(5 + 5 * max(0, num_masks), 5))
    if not isinstance(ax, np.ndarray):
        ax = np.array([ax])

    ax[0].imshow(image)
    ax[0].axis("off")
    ax[0].set_title("Original")

    to_tensor = transforms.PILToTensor()
    to_pil = transforms.ToPILImage()

    # Draw overlays when we have masks/boxes/phrases aligned
    try:
        for i, (mask, box, phrase) in enumerate(zip(masks, boxes, phrases)):
            img_t = to_tensor(image)
            box = box.unsqueeze(0)
            img_t = draw_bounding_boxes(img_t, box, colors=["red"], width=3)
            img_t = draw_segmentation_masks(img_t, mask, alpha=0.5, colors=["cyan"])
            overlay = to_pil(img_t)
            ax[1 + i].imshow(overlay)
            ax[1 + i].axis("off")
            ax[1 + i].set_title(str(phrase))
    except Exception:
        # If any mismatch, just skip visualization of overlays
        pass

    save_path = config.langsam_image_path.format(object=segmentation_count)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)
    print(f"✅ LangSAM result saved: {save_path}")

    # Normalize masks to torch.Tensor [N,H,W] float
    if isinstance(masks, list):
        # It may be list of dicts (common): extract the actual mask array
        if len(masks) > 0 and isinstance(masks[0], dict):
            print(f"🧪 Extracting {len(masks)} mask dicts...")
            keys = list(masks[0].keys())
            print("🧠 Available keys:", keys)
            if "segmentation" in keys:
                masks = [m["segmentation"] for m in masks]
            elif "mask" in keys:
                masks = [m["mask"] for m in masks]
            elif "masks" in keys:
                masks = [m["masks"] for m in masks]
            else:
                print("⚠️ No known mask key found; returning empty tensor.")
                return torch.zeros((0,)), boxes, phrases

        mask_tensors = []
        for m in masks:
            if isinstance(m, np.ndarray):
                m = torch.from_numpy(m)
            if torch.is_tensor(m):
                mask_tensors.append(m.bool())
        if len(mask_tensors) == 0:
            print("⚠️ No valid mask tensors; returning empty tensor.")
            return torch.zeros((0,)), boxes, phrases
        masks = torch.stack(mask_tensors).float()

    elif torch.is_tensor(masks):
        masks = masks.float()
    else:
        print("⚠️ Unexpected mask type; returning empty tensor.")
        masks = torch.zeros((0,))

    return masks, boxes, phrases


# -----------------------------
# ChatGPT (unchanged)
# -----------------------------
def get_chatgpt_output(client, model, new_prompt, messages, role, file=sys.stdout):
    print(role + ":", file=file)
    print(new_prompt, file=file)
    messages.append({"role": role, "content": new_prompt})

    completion = client.chat.completions.create(
        model=model,
        temperature=0,
        messages=messages,
        stream=True
    )

    print("assistant:", file=file)

    new_output = ""
    for chunk in completion:
        chunk_content = chunk.choices[0].delta.content
        finish_reason = chunk.choices[0].finish_reason
        if chunk_content is not None:
            print(chunk_content, end="", file=file)
            new_output += chunk_content
        else:
            print("finish_reason:", finish_reason, file=file)

    messages.append({"role": "assistant", "content": new_output})
    return messages


# -----------------------------
# XMem inference (fixed)
# -----------------------------
def _ensure_inference_core(xmem_model) -> InferenceCore:
    """
    Wrap raw XMem model into InferenceCore with a complete default configuration.
    Ensures all required keys are present.
    """
    # ✅ Choose correct device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"⚙️ XMem running on device: {device}")

    if isinstance(xmem_model, InferenceCore):
        return xmem_model

    # ✅ Full default configuration (covers all keys)
    default_cfg = {
        "enable_long_term": True,
        "enable_long_term_count_usage": False,
        "max_mid_term_frames": 10,
        "min_mid_term_frames": 5,
        "max_long_term_elements": 10000,
        "num_prototypes": 128,
        "top_k": 30,
        "mem_every": 5,
        "deep_update_every": -1,
        "disable_long_term": False,
        "no_amp": True,
        "size": 480,
        "hidden_dim": 256,
        "update_thres": 0.7,
        "device": device,
        "value_dim": 512,
        "key_dim": 64,
        "enable_debug": False,
        "multi_object": True,
    }

    # ✅ Merge with optional config.xmem_cfg
    xmem_cfg = getattr(config, "xmem_cfg", None)
    if xmem_cfg is None:
        print("⚙️ Using default XMem configuration")
        xmem_cfg = default_cfg
    else:
        print("⚙️ Using config.xmem_cfg with defaults merged")
        for k, v in default_cfg.items():
            if k not in xmem_cfg:
                xmem_cfg[k] = v

    # ✅ Create InferenceCore
    return InferenceCore(xmem_model, config=xmem_cfg)


def _load_initial_labels_as_prob(labels_path: str):
    """
    Load initial labeled mask (uint8 image) and convert to one-hot probability map for XMem.
    Labels are grayscale image values 0..K (0 = background).
    Returns torch.Tensor of shape [1, K+1, H, W].
    """
    if not os.path.exists(labels_path):
        print(f"⚠️ No initial label file found at {labels_path}")
        return None

    labels_img = Image.open(labels_path).convert("L")  # grayscale
    labels_np = np.array(labels_img, dtype=np.uint8)   # HxW

    K = int(np.max(labels_np))
    if K <= 0:
        print("⚠️ No labeled objects found in initial mask.")
        return None

    # ✅ Convert to one-hot tensor: shape [1, K+1, H, W]
    prob = index_numpy_to_one_hot_torch(labels_np, K + 1)

    # 🧠 Validate shape
    if prob.ndim == 3:
        prob = prob.unsqueeze(0)
    elif prob.ndim > 4:
        prob = prob.squeeze(0)

    print(f"✅ Loaded initial labels from {labels_path}, K={K}, prob shape={tuple(prob.shape)}")
    return prob


def get_xmem_output(xmem_model, device, trajectory_length):
    """
    Run XMem over saved trajectory frames.
    Returns list of numpy label maps (HxW).
    """
    # ✅ Ensure correct core and device
    core = _ensure_inference_core(xmem_model)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    out_masks = []
    first_initialized = False

    for t in range(trajectory_length):
        frame_path = config.rgb_image_trajectory_path.format(step=t)
        if not os.path.exists(frame_path):
            print(f"⚠️ Frame not found: {frame_path}, skipping...")
            continue

        # ✅ Load and convert to tensor [1,3,H,W]
        frame_np = np.array(Image.open(frame_path).convert("RGB"))
        frame_torch = torch.from_numpy(frame_np).permute(2, 0, 1).float() / 255.0
        frame_torch = frame_torch.unsqueeze(0).to(device)

        if frame_torch.ndim != 4 or frame_torch.shape[1] != 3:
            raise ValueError(f"❌ Frame tensor should be [1,3,H,W], got {frame_torch.shape}")

        # 🧠 Initialize with mask on first frame
        if not first_initialized:
            init_prob = _load_initial_labels_as_prob(config.xmem_input_path)
            if init_prob is None:
                _, _, H, W = frame_torch.shape
                init_prob = torch.zeros((1, 1, H, W), device=device)
                init_prob[:, 0] = 1.0  # background only

            # Set labels list
            num_classes = init_prob.shape[1]
            core.all_labels = list(range(num_classes))

            pred_prob = core.step(frame_torch, init_prob)
            first_initialized = True
        else:
            pred_prob = core.step(frame_torch, None)

        # ✅ Convert to numpy
        if isinstance(pred_prob, torch.Tensor):
            mask_np = torch_prob_to_numpy_mask(pred_prob)
        else:
            prob = getattr(core, "prob", None)
            if isinstance(prob, torch.Tensor):
                mask_np = torch_prob_to_numpy_mask(prob)
            else:
                h, w = frame_np.shape[:2]
                mask_np = np.zeros((h, w), dtype=np.uint8)

        out_masks.append(mask_np)

    print(f"✅ Generated {len(out_masks)} XMem masks")
    return out_masks






