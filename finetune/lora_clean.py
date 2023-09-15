from pathlib import Path
import sys
import os
import time
import hydra
import lightning as L
import torch
import trimesh
import wandb
from lightning.fabric.strategies import DDPStrategy
from lightning.pytorch.loggers import WandbLogger
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm

from finetune.lora import save_lora_checkpoint
from generate.base import generate
from lit_gpt import Tokenizer
from lit_gpt.lora import GPT, mark_only_lora_as_trainable, Config
from lit_gpt.utils import get_default_supported_precision, check_valid_checkpoint_dir, quantization, num_parameters, chunked_cross_entropy, lazy_load
from lit_gpt.speed_monitor import SpeedMonitorFabric, estimate_flops, measure_flops
import datetime
import randomname

from scripts.ngon_helpers import plot_vertices_and_faces
from scripts.ngon_soup import NgonSoup

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))


def generate_experiment_name(name, config):
    if config.resume is not None:
        experiment = Path(config.resume).parents[1].name
        os.environ['experiment'] = experiment
    elif not os.environ.get('experiment'):
        experiment = f"{datetime.datetime.now().strftime('%m%d%H%M')}_{name}_{config.experiment}_{randomname.get_name()}"
        os.environ['experiment'] = experiment
    else:
        experiment = os.environ['experiment']
    return experiment


@hydra.main(config_path='../config', config_name='lora', version_base='1.2')
def setup(config):
    name = "LlaMesh"
    if not config.wandb_main and config.suffix == '':
        config.suffix = '-dev'
    config.experiment = generate_experiment_name(name, config)
    config.data_dir = Path(config.data_dir)
    config.checkpoint_dir = Path(config.checkpoint_dir)
    config.out_dir = Path("runs", config.experiment)

    precision = config.precision or get_default_supported_precision(training=True)
    gpu_count = torch.cuda.device_count()
    if gpu_count > 1:
        if config.quantize:
            raise NotImplementedError(
                "Quantization is currently not supported for multi-GPU training. "
                "Please set devices=1 when using the --quantization flag."
            )
        strategy = DDPStrategy(process_group_backend="nccl")
    else:
        strategy = "auto"

    logger = WandbLogger(project=f'{name}{config.suffix}', name=config.experiment, id=config.experiment, settings=wandb.Settings(start_method='thread'))
    fabric = L.Fabric(devices=gpu_count, strategy=strategy, precision=precision, loggers=logger)
    fabric.launch(main, config)


def main(fabric, config):
    check_valid_checkpoint_dir(config.checkpoint_dir)
    speed_monitor = SpeedMonitorFabric(fabric, window_size=50, time_unit="seconds")
    speed_monitor.enabled = False
    fabric.seed_everything(1337)  # same seed for every process to init model (FSDP)

    if fabric.global_rank == 0:
        os.makedirs(config.out_dir, exist_ok=True)

    lora_config = Config.from_name(
        name=config.checkpoint_dir.name,
        r=config.lora_r,
        alpha=config.lora_alpha,
        dropout=config.lora_dropout,
        to_query=config.lora_query,
        to_key=config.lora_key,
        to_value=config.lora_value,
        to_projection=config.lora_projection,
        to_mlp=config.lora_mlp,
        to_head=config.lora_head,
    )

    checkpoint_path = config.checkpoint_dir / "lit_model.pth"
    fabric.print(f"Loading model {str(checkpoint_path)!r} with {lora_config.__dict__}")

    with fabric.init_module(empty_init=False), quantization(config.quantize):
        model = GPT(lora_config)
    with lazy_load(checkpoint_path) as checkpoint:
        # strict=False because missing keys due to LoRA weights not contained in state dict
        model.load_state_dict(checkpoint, strict=False)
    mark_only_lora_as_trainable(model)

    fabric.print(f"Number of trainable parameters: {num_parameters(model, requires_grad=True):,}")
    fabric.print(f"Number of non trainable parameters: {num_parameters(model, requires_grad=False):,}")
    trainable_params = [p for p in model.parameters() if p.requires_grad]

    if config.quantize and config.quantize.startswith("bnb."):
        import bitsandbytes as bnb
        optimizer = bnb.optim.PagedAdamW(trainable_params, lr=config.learning_rate, weight_decay=config.weight_decay)
    else:
        optimizer = torch.optim.AdamW(trainable_params, lr=config.learning_rate, weight_decay=config.weight_decay)

    model, optimizer = fabric.setup(model, optimizer)
    fabric.seed_everything(1337 + fabric.global_rank)

    train_time = time.perf_counter()
    train(fabric, model, optimizer, config, speed_monitor)
    fabric.print(f"Training time: {(time.perf_counter() - train_time):.2f}s")
    if fabric.device.type == "cuda":
        fabric.print(f"Memory used: {torch.cuda.max_memory_allocated() / 1e9:.02f} GB")

    # Save the final LoRA checkpoint at the end of training
    save_path = config.out_dir / "lit_model_lora_finetuned.pth"
    save_lora_checkpoint(fabric, model, save_path)


def train(fabric, model, optimizer, config, speed_monitor):
    fabric.logger.log_hyperparams(config.__dict__)
    Path(config.out_dir, "config.yaml").write_text(OmegaConf.to_yaml(config))

    tokenizer = Tokenizer(config.checkpoint_dir)
    train_data = NgonSoup(tokenizer, config, 'train', config.scale_augment)
    val_data = NgonSoup(tokenizer, config, 'val', config.scale_augment_val)

    max_iters = (len(train_data) // config.micro_batch_size) * config.num_epochs
    warmup_steps = int((len(train_data) // config.micro_batch_size) * config.warmup_epochs)
    gradient_accumulation_iters = config.batch_size // config.micro_batch_size

    eval_interval = int((len(train_data) // config.micro_batch_size) * config.eval_interval)
    eval_iters = int((len(val_data) // config.micro_batch_size) * config.eval_iters)
    save_interval = int((len(train_data) // config.micro_batch_size) * config.eval_interval)

    print('max_iters', max_iters)
    print('warmup_steps', warmup_steps)
    print('gradient_accumulation_iters', gradient_accumulation_iters)
    print('eval_interval', eval_interval)
    print('eval_iters', eval_iters)
    print('save_interval', save_interval)

    max_seq_length = config.chunk_size

    print('Setting up dataloaders...')
    train_dataloader = DataLoader(train_data, batch_size=config.micro_batch_size, shuffle=True, drop_last=not config.overfit, num_workers=config.num_workers, pin_memory=True)
    val_dataloader = DataLoader(val_data, batch_size=config.micro_batch_size, shuffle=True, drop_last=False, num_workers=config.num_workers)

    train_dataloader = cycle(fabric.setup_dataloaders(train_dataloader))
    val_dataloader = fabric.setup_dataloaders(val_dataloader)

    validate(fabric, model, val_data, val_dataloader, eval_iters // 10, config.max_new_tokens, tokenizer, config.out_dir / f"{0:06d}")  # sanity check

    with torch.device("meta"):
        meta_model = GPT(model.config)
        mark_only_lora_as_trainable(meta_model)
        # "estimated" is not as precise as "measured". Estimated is optimistic but widely used in the wild.
        # When comparing MFU or FLOP numbers with other projects that use estimated FLOPs,
        # consider passing `SpeedMonitor(flops_per_batch=estimated_flops)` instead
        estimated_flops = estimate_flops(meta_model) * config.micro_batch_size
        fabric.print(f"Estimated TFLOPs: {estimated_flops * fabric.world_size / 1e12:.2f}")
        # this assumes that all samples have a fixed length equal to the longest sequence length
        # which is most likely false during finetuning
        x = torch.randint(0, 1, (config.micro_batch_size, max_seq_length))
        measured_flops = measure_flops(meta_model, x)
        fabric.print(f"Measured TFLOPs: {measured_flops * fabric.world_size / 1e12:.2f}")
        del meta_model, x

    step_count = 0
    total_lengths = 0
    total_t0 = time.perf_counter()

    with tqdm(list(range(max_iters))) as titer:
        for iter_num in titer:
            if step_count <= warmup_steps:
                # linear warmup
                lr = config.learning_rate * step_count / warmup_steps
                for param_group in optimizer.param_groups:
                    param_group["lr"] = lr

            iter_t0 = time.perf_counter()
            sample = next(train_dataloader)
            input_ids = sample['input']
            targets = sample['target']
            is_accumulating = (iter_num + 1) % gradient_accumulation_iters != 0
            with fabric.no_backward_sync(model, enabled=is_accumulating):
                logits = model(input_ids, lm_head_chunk_size=128)
                # shift the targets such that output n predicts token n+1
                logits[-1] = logits[-1][..., :-1, :]
                loss = chunked_cross_entropy(logits, targets[..., 1:])
                unchunked_logits = torch.cat([x.detach() for x in logits], dim=1)
                acc = accuracy(unchunked_logits, targets[..., 1:], ignore_label=-1, device=fabric.device)
                fabric.backward(loss / gradient_accumulation_iters)

            if not is_accumulating:
                optimizer.step()
                optimizer.zero_grad()
                step_count += 1

            t1 = time.perf_counter()
            total_lengths += input_ids.size(1)
            speed_monitor.on_train_batch_end(
                (iter_num + 1) * config.micro_batch_size,
                t1 - total_t0,
                # this assumes that device FLOPs are the same and that all devices have the same batch size
                fabric.world_size,
                flops_per_batch=measured_flops,
                lengths=total_lengths,
            )
            if iter_num % config.log_interval == 0:
                # fabric.print(
                #     f"iter {iter_num} step {step_count}: loss {loss.item():.4f}, iter time:"
                #     f" {(t1 - iter_t0) * 1000:.2f}ms{' (optimizer.step)' if not is_accumulating else ''}"
                # )
                fabric.logger.log_metrics({"train/ce_loss": loss.item()}, step=iter_num)
                fabric.logger.log_metrics({"train/acc": acc.item()}, step=iter_num)
                titer.set_postfix(loss=loss.item(), acc=acc.item())

            if (iter_num + 1) % eval_interval == 0:
                t0 = time.perf_counter()
                val_loss, val_acc = validate(fabric, model, val_data, val_dataloader, eval_iters, config.max_new_tokens, tokenizer, config.out_dir / f"{iter_num:06d}")
                t1 = time.perf_counter() - t0
                speed_monitor.eval_end(t1)
                # fabric.print(f"step {iter_num}: val loss {val_loss.item():.4f}, val time: {t1 * 1000:.2f}ms")
                fabric.logger.log_metrics({"val/ce_loss": val_loss.item()}, step=iter_num)
                fabric.logger.log_metrics({"val/acc": val_acc.item()}, step=iter_num)
                fabric.barrier()

            if (iter_num + 1) % save_interval == 0:
                checkpoint_path = config.out_dir / f"iter-{iter_num:06d}-ckpt.pth"
                fabric.print('saving...', checkpoint_path)
                save_lora_checkpoint(fabric, model, checkpoint_path)


@torch.no_grad()
def validate(fabric, model, val_data, val_dataloader, eval_iters, max_new_tokens, tokenizer, outdir):
    fabric.print("Validating ...")
    model.eval()
    losses = torch.zeros(eval_iters)
    accs = torch.zeros(eval_iters)
    for k, batch in enumerate(val_dataloader):
        input_ids = batch['input']
        targets = batch['target']
        logits = model(input_ids)
        losses[k] = chunked_cross_entropy(logits[..., :-1, :], targets[..., 1:], chunk_size=0)
        accs[k] = accuracy(logits[..., :-1, :].detach(), targets[..., 1:], ignore_label=-1, device=fabric.device)
        if k + 1 == eval_iters:
            break

    val_loss = losses.mean()
    acc = accs.mean()

    # produce an example:
    max_returned_tokens = min(max_new_tokens, model.config.block_size)
    outdir.mkdir(exist_ok=True)

    for i in range(8):
        sample = val_data.get_start(device=fabric.device)[0][0]
        with fabric.init_tensor():
            # do not set `max_seq_length=max_returned_token` because memory is not a concern here
            model.set_kv_cache(batch_size=1)
        output = generate(
            model, idx=sample, max_returned_tokens=max_returned_tokens, temperature=0.8, eos_id=tokenizer.eos_id
        )
        model.clear_kv_cache()
        gen_vertices, gen_faces = val_data.decode(output)
        plot_vertices_and_faces(gen_vertices, gen_faces, outdir / f"{i:02d}.jpg")
        trimesh.Trimesh(vertices=gen_vertices, faces=gen_faces, process=False).export(outdir / f"{i:02d}.obj")

    # output = tokenizer.decode(output)
    # fabric.print(output)

    model.train()
    return val_loss, acc


def accuracy(y_pred, y_true, ignore_label=None, device=None):
    y_pred = y_pred.argmax(dim=-1)

    if ignore_label:
        normalizer = torch.sum(y_true != ignore_label)  # type: ignore
        ignore_mask = torch.where(  # type: ignore
            y_true == ignore_label,
            torch.zeros_like(y_true, device=device),
            torch.ones_like(y_true, device=device)
        ).type(torch.float32)
    else:
        normalizer = y_true.shape[0]
        ignore_mask = torch.ones_like(y_true, device=device).type(torch.float32)
    acc = (y_pred.reshape(-1) == y_true.reshape(-1)).type(torch.float32)  # type: ignore
    acc = torch.sum(acc*ignore_mask.flatten())
    return acc / normalizer


def cycle(iterable):
    while True:
        for x in iterable:
            yield x


if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")
    setup()
