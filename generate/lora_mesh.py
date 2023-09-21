import sys
import time
import warnings
from pathlib import Path
from typing import Literal, Optional

import lightning as L
import omegaconf
import torch
import trimesh
from lightning.fabric.strategies import FSDPStrategy
from tqdm import tqdm

from scripts.ngon_helpers import plot_vertices_and_faces
from scripts.ngon_soup import NgonSoup

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

from generate.base import generate
from lit_gpt import Tokenizer
from lit_gpt.lora import GPT, Block, Config, merge_lora_weights
from lit_gpt.utils import check_valid_checkpoint_dir, get_default_supported_precision, lazy_load, quantization

# import pydevd_pycharm
# pydevd_pycharm.settrace('131.159.40.27', port=8000, stdoutToServer=True, stderrToServer=True)


def main(
    lora_path: Path = Path("out/lora/alpaca/lit_model_lora_finetuned.pth"),
    quantize: Optional[Literal["bnb.nf4", "bnb.nf4-dq", "bnb.fp4", "bnb.fp4-dq", "bnb.int8", "gptq.int4"]] = None,
    max_new_tokens: int = 4096,
    top_k: int = 200,
    temperature: float = 0.8,
    strategy: str = "auto",
    devices: int = 1,
    precision: Optional[str] = None,
) -> None:
    """Generates a response based on a given instruction and an optional input.
    This script will only work with checkpoints from the instruction-tuned GPT-LoRA model.
    See `finetune/lora.py`.

    Args:
        prompt: The prompt/instruction (Alpaca style).
        input: Optional input (Alpaca style).
        lora_path: Path to the checkpoint with trained adapter weights, which are the output of
            `finetune/lora.py`.
        checkpoint_dir: The path to the checkpoint folder with pretrained GPT weights.
        quantize: Whether to quantize the model and using which method:
            - bnb.nf4, bnb.nf4-dq, bnb.fp4, bnb.fp4-dq: 4-bit quantization from bitsandbytes
            - bnb.int8: 8-bit quantization from bitsandbytes
            - gptq.int4: 4-bit quantization from GPTQ
            for more details, see https://github.com/Lightning-AI/lit-gpt/blob/main/tutorials/quantize.md
        max_new_tokens: The number of generation steps to take.
        top_k: The number of top most probable tokens to consider in the sampling process.
        temperature: A value controlling the randomness of the sampling process. Higher values result in more random
            samples.
        strategy: Indicates the Fabric strategy setting to use.
        devices: How many devices to use.
        precision: Indicates the Fabric precision setting to use.
    """
    precision = precision or get_default_supported_precision(training=False)

    if strategy == "fsdp":
        strategy = FSDPStrategy(auto_wrap_policy={Block}, cpu_offload=False)
    fabric = L.Fabric(devices=devices, precision=precision, strategy=strategy)
    fabric.launch()

    config_hydra = omegaconf.OmegaConf.load(Path(lora_path).parents[0] / "config.yaml")
    checkpoint_dir = Path(config_hydra.checkpoint_dir)
    config_hydra.out_dir = Path(config_hydra.out_dir)

    check_valid_checkpoint_dir(checkpoint_dir)

    config = Config.from_json(
        checkpoint_dir / "lit_config.json",
        r=config_hydra.lora_r,
        alpha=config_hydra.lora_alpha,
        dropout=config_hydra.lora_dropout,
        to_query=config_hydra.lora_query,
        to_key=config_hydra.lora_key,
        to_value=config_hydra.lora_value,
        to_projection=config_hydra.lora_projection,
        to_mlp=config_hydra.lora_mlp,
        to_head=config_hydra.lora_head,
    )

    if quantize is not None and devices > 1:
        raise NotImplementedError
    if quantize == "gptq.int4":
        model_file = "lit_model_gptq.4bit.pth"
        if not (checkpoint_dir / model_file).is_file():
            raise ValueError("Please run `python quantize/gptq.py` first")
    else:
        model_file = "lit_model.pth"
    checkpoint_path = checkpoint_dir / model_file

    fabric.print(f"Loading model {str(checkpoint_path)!r} with {config.__dict__}", file=sys.stderr)
    t0 = time.perf_counter()
    with fabric.init_module(empty_init=True), quantization(quantize):
        model = GPT(config)
    fabric.print(f"Time to instantiate model: {time.perf_counter() - t0:.02f} seconds.", file=sys.stderr)

    t0 = time.perf_counter()
    with lazy_load(checkpoint_path) as checkpoint, lazy_load(lora_path) as lora_checkpoint:
        checkpoint.update(lora_checkpoint.get("model", lora_checkpoint))
        model.load_state_dict(checkpoint, strict=quantize is None)
    fabric.print(f"Time to load the model weights: {time.perf_counter() - t0:.02f} seconds.", file=sys.stderr)

    model.eval()
    merge_lora_weights(model)
    model = fabric.setup(model)

    tokenizer = Tokenizer(checkpoint_dir)
    val_data = NgonSoup(tokenizer, config_hydra, 'val', False, False)

    max_returned_tokens = max_new_tokens
    (config_hydra.out_dir / "inference").mkdir(exist_ok=True)

    with fabric.init_tensor():
        # enable the kv cache
        # todo: delete this line, just a debug
        # model.max_seq_length = 64
        model.set_kv_cache(batch_size=1)

    t0 = time.perf_counter()
    tokens_generated = 0
    for i in tqdm(range(32)):
        sample = val_data.get_start(device=fabric.device)[0][0]
        y = generate(
            model,
            sample,
            max_returned_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=0.9,
            eos_id=tokenizer.eos_id,
        )
        tokens_generated += y.size(0)
        gen_vertices, gen_faces = val_data.decode(y)
        plot_vertices_and_faces(gen_vertices, gen_faces, config_hydra.out_dir / "inference" / f"{i:02d}.jpg")
        trimesh.Trimesh(vertices=gen_vertices, faces=gen_faces, process=False).export(config_hydra.out_dir / "inference" / f"{i:02d}.obj")

    t = time.perf_counter() - t0
    model.clear_kv_cache()

    fabric.print(f"\n\nTime for inference: {t:.02f} sec total, {tokens_generated / t:.02f} tokens/sec", file=sys.stderr)
    if fabric.device.type == "cuda":
        fabric.print(f"Memory used: {torch.cuda.max_memory_allocated() / 1e9:.02f} GB", file=sys.stderr)


if __name__ == "__main__":
    from jsonargparse import CLI

    torch.set_float32_matmul_precision("high")
    warnings.filterwarnings(
        # Triggered internally at ../aten/src/ATen/EmptyTensor.cpp:31
        "ignore",
        message="ComplexHalf support is experimental and many operators don't support it yet",
    )
    CLI(main)
