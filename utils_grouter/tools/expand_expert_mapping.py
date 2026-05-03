import argparse
import json
import os
from typing import Dict, Iterable, Tuple

import torch
import torch.distributed as dist
import torch.nn.functional as F
from megatron.training.tokenizer import build_tokenizer

from utils_grouter.core.config import OptimizationConfig
from utils_grouter.grouter import grouter as grt
from utils_grouter.utils.megatron_dataloader import create_megatron_dataloader


def parse_args(args):
    parser = argparse.ArgumentParser()
    # grouter config
    parser.add_argument("--grouter-config-path", type=str, required=True)
    parser.add_argument("--grouter-checkpoint-path", type=str, required=True)
    parser.add_argument("--output-checkpoint-path", type=str, required=True)
    parser.add_argument(
        "--target-num-experts",
        type=int,
        required=True,
        help="Target number of experts after expansion, must be larger than source.",
    )

    # dataset config
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--max-length", type=int, required=True)
    parser.add_argument("--data-prefix", type=str, nargs="*", required=True)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--tokenizer-type", type=str, default=None)
    parser.add_argument("--tokenizer-model", type=str, default=None)
    parser.add_argument("--total-steps", type=int, default=100)

    # expansion config
    parser.add_argument(
        "--bad-token-quantile",
        type=float,
        default=0.2,
        help="Low-score token quantile threshold.",
    )
    parser.add_argument("--kmeans-iters", type=int, default=100)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args(args)


def _get_router_linear(model: torch.nn.Module) -> torch.nn.Linear:
    if not hasattr(model, "grouter_structure"):
        raise AttributeError("Model has no attribute `grouter_structure`.")
    if not hasattr(model.grouter_structure, "score"):
        raise AttributeError("Model.grouter_structure has no attribute `score`.")
    score_module = model.grouter_structure.score
    if not hasattr(score_module, "score"):
        raise AttributeError("Model.grouter_structure.score has no attribute `score`.")
    router_linear = score_module.score
    if not isinstance(router_linear, torch.nn.Linear):
        raise TypeError("Router score layer is not `torch.nn.Linear`.")
    return router_linear


def _build_model_inputs(batch, device: torch.device) -> Dict[str, torch.Tensor]:
    if isinstance(batch, dict):
        input_ids = batch.get("input_ids", batch.get("tokens", None))
        attention_mask = batch.get("attention_mask", None)
        position_ids = batch.get("position_ids", None)
        if input_ids is None:
            raise ValueError("Cannot find `input_ids` or `tokens` in dataloader batch dict.")
        model_inputs = {"input_ids": input_ids}
        if attention_mask is not None:
            model_inputs["attention_mask"] = attention_mask
        if position_ids is not None:
            model_inputs["position_ids"] = position_ids

    for key, value in model_inputs.items():
        if torch.is_tensor(value):
            if key == "attention_mask" and value.dtype == torch.bool:
                value = value.to(dtype=torch.long)
            model_inputs[key] = value.to(device=device, non_blocking=True)
    return model_inputs


def _run_torch_kmeans(
    x: torch.Tensor,
    num_clusters: int,
    max_iters: int = 100,
    seed: int = 42,
) -> torch.Tensor:
    """
    A lightweight fallback KMeans implementation in pure torch.
    Args:
        x: [N, D] float32 tensor on cpu/cuda.
    Returns:
        centroids: [K, D].
    """
    if x.dim() != 2:
        raise ValueError(f"x must be 2D, got {x.shape}.")
    n_points = x.shape[0]
    if n_points == 0:
        raise ValueError("Cannot run KMeans on empty tensor.")

    g = torch.Generator(device=x.device)
    g.manual_seed(seed)

    if n_points >= num_clusters:
        init_idx = torch.randperm(n_points, generator=g, device=x.device)[:num_clusters]
    else:
        repeat_idx = torch.randint(0, n_points, (num_clusters,), generator=g, device=x.device)
        init_idx = repeat_idx
    centroids = x[init_idx].clone()

    for _ in range(max_iters):
        distances = torch.cdist(x, centroids, p=2)
        labels = torch.argmin(distances, dim=1)
        new_centroids = torch.empty_like(centroids)
        for k in range(num_clusters):
            mask = labels == k
            if torch.any(mask):
                new_centroids[k] = x[mask].mean(dim=0)
            else:
                rand_idx = torch.randint(0, n_points, (1,), generator=g, device=x.device)
                new_centroids[k] = x[rand_idx]
        if torch.allclose(new_centroids, centroids, atol=1e-5, rtol=1e-4):
            centroids = new_centroids
            break
        centroids = new_centroids
    return centroids


def _cluster_centroids(
    x_bad: torch.Tensor,
    num_new_experts: int,
    random_seed: int,
    kmeans_iters: int,
) -> torch.Tensor:
    x_bad = x_bad.to(torch.float32)
    try:
        from sklearn.cluster import KMeans

        kmeans = KMeans(
            n_clusters=num_new_experts,
            random_state=random_seed,
            max_iter=kmeans_iters,
            n_init="auto",
        )
        centers_np = kmeans.fit(x_bad.cpu().numpy()).cluster_centers_
        return torch.from_numpy(centers_np).to(dtype=torch.float32, device=x_bad.device)
    except Exception:
        return _run_torch_kmeans(
            x=x_bad,
            num_clusters=num_new_experts,
            max_iters=kmeans_iters,
            seed=random_seed,
        )


def _extract_residual_tokens(
    model: torch.nn.Module,
    data_loader: Iterable,
    device: torch.device,
    total_steps: int,
    verbose: bool = False,
) -> torch.Tensor:
    """
    Capture the input hidden states of the final routing linear layer.
    Output shape: [N, d].
    """
    router_linear = _get_router_linear(model)
    hidden_chunks = []

    def _capture_router_input(_, hook_inputs):
        hidden = hook_inputs[0].detach()
        hidden = hidden.reshape(-1, hidden.shape[-1]).to(torch.float32)
        hidden_chunks.append(hidden.cpu())

    handle = router_linear.register_forward_pre_hook(_capture_router_input)
    model.eval()
    with torch.no_grad():
        for step, batch in enumerate(data_loader):
            if step >= total_steps:
                break
            model_inputs = _build_model_inputs(batch, device)
            _ = model(**model_inputs)
            if verbose and (step + 1) % 10 == 0:
                print(f"[extract] processed steps: {step + 1}")
    handle.remove()

    if len(hidden_chunks) == 0:
        raise RuntimeError("No hidden states were captured from router input.")

    all_hidden = torch.cat(hidden_chunks, dim=0)
    return all_hidden


def _all_gather_tokens_to_rank0(
    x_local: torch.Tensor,
    device: torch.device,
    rank: int,
    world_size: int,
) -> torch.Tensor:
    if not (dist.is_available() and dist.is_initialized()):
        return x_local

    if x_local.dim() != 2:
        raise ValueError(f"x_local must be 2D, got {x_local.shape}.")

    local_rows, hidden_size = x_local.shape
    local_rows_t = torch.tensor([local_rows], dtype=torch.long, device=device)
    all_rows_t = [torch.zeros_like(local_rows_t) for _ in range(world_size)]
    dist.all_gather(all_rows_t, local_rows_t)
    all_rows = [int(v.item()) for v in all_rows_t]
    expected_rows = all_rows[0]
    if any(rows != expected_rows for rows in all_rows):
        raise RuntimeError(
            f"Inconsistent token counts across ranks: {all_rows}. "
            "Please ensure each rank processes the same number of tokens."
        )

    x_local_dev = x_local.to(device=device, dtype=torch.float32, non_blocking=True)

    gathered = [
        torch.empty((expected_rows, hidden_size), dtype=x_local_dev.dtype, device=device)
        for _ in range(world_size)
    ]
    dist.all_gather(gathered, x_local_dev)

    if rank != 0:
        return None

    chunks = []
    for tensor, rows_t in zip(gathered, all_rows_t):
        rows = int(rows_t.item())
        if rows > 0:
            chunks.append(tensor[:rows].cpu())
    if len(chunks) == 0:
        raise RuntimeError("No tokens gathered across ranks.")
    return torch.cat(chunks, dim=0)


def _project_to_old_null_space(
    centroids: torch.Tensor,
    old_weight: torch.Tensor,
) -> torch.Tensor:
    """
    Orthogonal projection to old experts' null space.
    Equivalent to: C @ (I - A(A^T A)^-1 A^T), where A = W_old^T.
    """
    a = old_weight.t().to(torch.float32)  # [d, E_old]
    c = centroids.to(torch.float32)  # [E_new, d]
    gram = a.t() @ a  # [E_old, E_old]
    gram_inv = torch.linalg.pinv(gram)
    projected = c - ((c @ a) @ gram_inv) @ a.t()
    return projected


def expand_moe_experts(
    model: torch.nn.Module,
    data_loader: Iterable,
    device: torch.device,
    rank: int,
    world_size: int,
    target_num_experts: int,
    total_steps: int = 100,
    bad_token_quantile: float = 0.2,
    kmeans_iters: int = 100,
    random_seed: int = 42,
    verbose: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Expand the router output dimension by appending new experts.

    Returns:
        new_weight_full: [target_num_experts, d]
        w_new: [num_new_experts, d]
    """
    if not 0.0 < bad_token_quantile < 1.0:
        raise ValueError("bad_token_quantile must be in (0, 1).")

    torch.manual_seed(random_seed)
    router_linear = _get_router_linear(model)
    old_weight = router_linear.weight.detach()
    old_num_experts, hidden_size = old_weight.shape
    if target_num_experts <= old_num_experts:
        raise ValueError(
            f"target_num_experts ({target_num_experts}) must be greater than current "
            f"num_experts ({old_num_experts})."
        )
    num_new_experts = target_num_experts - old_num_experts

    x_all = _extract_residual_tokens(
        model=model,
        data_loader=data_loader,
        device=device,
        total_steps=total_steps,
        verbose=verbose,
    )  # [N, d]
    if x_all.shape[1] != hidden_size:
        raise RuntimeError(
            f"Hidden size mismatch: extracted {x_all.shape[1]} vs router {hidden_size}."
        )
    x_all = _all_gather_tokens_to_rank0(
        x_local=x_all,
        device=device,
        rank=rank,
        world_size=world_size,
    )
    if rank != 0:
        return old_weight, old_weight.new_empty((0, hidden_size))

    old_weight_cpu = old_weight.detach().to(torch.float32).cpu()
    scores = x_all @ old_weight_cpu.t()  # [N, E_old]
    max_scores = scores.max(dim=-1).values
    threshold = torch.quantile(max_scores, bad_token_quantile)
    bad_mask = max_scores <= threshold
    x_bad = x_all[bad_mask]
    if x_bad.shape[0] < num_new_experts:
        take = min(x_all.shape[0], max(num_new_experts * 4, num_new_experts))
        low_idx = torch.argsort(max_scores)[:take]
        x_bad = x_all[low_idx]
    if verbose:
        print(
            f"[expand] tokens={x_all.shape[0]}, bad_tokens={x_bad.shape[0]}, "
            f"new_experts={num_new_experts}"
        )

    centroids = _cluster_centroids(
        x_bad=x_bad,
        num_new_experts=num_new_experts,
        random_seed=random_seed,
        kmeans_iters=kmeans_iters,
    ).to(device)

    w_new = _project_to_old_null_space(
        centroids=centroids,
        old_weight=old_weight.to(device),
    )

    # Keep norm scale aligned with old experts.
    old_norm_mean = old_weight.to(torch.float32).norm(dim=-1).mean()
    w_new = F.normalize(w_new, p=2, dim=-1) * old_norm_mean

    # Intra-group orthogonalization for new experts.
    q, _ = torch.linalg.qr(w_new.t(), mode="reduced")
    w_new = q.t()
    w_new = F.normalize(w_new, p=2, dim=-1) * old_norm_mean

    new_weight_full = torch.cat(
        [old_weight, w_new.to(dtype=old_weight.dtype, device=old_weight.device)],
        dim=0,
    )
    router_linear.weight = torch.nn.Parameter(new_weight_full)

    # Keep runtime metadata consistent with new router shape.
    if hasattr(model, "num_experts"):
        model.num_experts = target_num_experts
    if hasattr(model, "target_num_experts"):
        model.target_num_experts = target_num_experts
    if hasattr(model, "bias") and isinstance(model.bias, torch.nn.Parameter):
        old_bias = model.bias.data
        if old_bias.numel() < target_num_experts:
            expanded_bias = torch.zeros(
                target_num_experts,
                dtype=old_bias.dtype,
                device=old_bias.device,
            )
            expanded_bias[: old_bias.numel()] = old_bias
            model.bias = torch.nn.Parameter(expanded_bias)

    if verbose:
        with torch.no_grad():
            old_f = old_weight.to(torch.float32).to(device)
            new_f = w_new.to(torch.float32).to(device)
            cross = torch.abs(new_f @ old_f.t()).max().item()
            intra = torch.abs(
                (new_f @ new_f.t()) - torch.diag_embed((new_f * new_f).sum(dim=-1))
            ).max().item()
            print(f"[expand] max |W_new W_old^T| = {cross:.6e}")
            print(f"[expand] max offdiag |W_new W_new^T| = {intra:.6e}")

    return new_weight_full, w_new


def _load_checkpoint(checkpoint_path: str, device: torch.device) -> Dict[str, torch.Tensor]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if not isinstance(checkpoint, dict):
        raise TypeError("Checkpoint must be a dict-like state dict.")
    if checkpoint_path.endswith(".pt") and isinstance(checkpoint, dict):
        if "grouter_states" in checkpoint:
            grouter_states = checkpoint["grouter_states"]
            if isinstance(grouter_states, dict) and len(grouter_states) > 0:
                first_key = sorted(grouter_states.keys())[0]
                return grouter_states[first_key]
        return checkpoint
    return checkpoint


def _build_dataloader(args, rank: int, world_size: int):
    config = OptimizationConfig(
        num_nodes=1,
        num_experts=args.num_experts,
        training_config=None,
        experts_per_gpu=None,
        network_topology=None,
        split_config="1,0,0",
        random_seed=args.random_seed,
        train_iters=args.total_steps,
        global_batch_size=args.batch_size * world_size,
        tokenizer_type=args.tokenizer_type,
        tokenizer_model=args.tokenizer_model,
    )
    config.rank = rank
    config.make_vocab_size_divisible_by = 128
    config.tensor_model_parallel_size = 1
    config.trust_remote_code = True
    tokenizer = build_tokenizer(config)
    args.checkpoint_dir = os.path.dirname(args.output_checkpoint_path)
    return create_megatron_dataloader(args, tokenizer, config, rank, world_size, 0)


def _resolve_relative_to(base_file: str, maybe_relative_path: str) -> str:
    if os.path.isabs(maybe_relative_path):
        return maybe_relative_path
    # First try path as-is (workspace/cwd relative). Some configs already store
    # a project-root-relative path like "utils_grouter/.../structure_config.json".
    if os.path.exists(maybe_relative_path):
        return os.path.normpath(maybe_relative_path)

    # Fallback: resolve relative to the config file directory.
    return os.path.normpath(os.path.join(os.path.dirname(base_file), maybe_relative_path))


def _save_expanded_configs(
    grouter_config_path: str,
    grouter_config: Dict,
    target_num_experts: int,
    output_dir: str,
) -> Tuple[str, str]:
    updated_config = dict(grouter_config)
    old_target = updated_config.get("target_num_experts")
    if isinstance(old_target, str):
        updated_config["target_num_experts"] = str(target_num_experts)
    else:
        updated_config["target_num_experts"] = target_num_experts

    structure_cfg_rel = updated_config.get("model_config_path")
    if not structure_cfg_rel:
        raise KeyError("`model_config_path` is missing in grouter config.")
    structure_cfg_path = _resolve_relative_to(grouter_config_path, structure_cfg_rel)
    with open(structure_cfg_path, "r") as f:
        structure_config = json.load(f)
    structure_config["num_labels"] = target_num_experts

    output_config_path = os.path.join(output_dir, os.path.basename(grouter_config_path))
    output_structure_path = os.path.join(output_dir, os.path.basename(structure_cfg_path))
    with open(output_config_path, "w") as f:
        json.dump(updated_config, f, indent=2)
        f.write("\n")
    with open(output_structure_path, "w") as f:
        json.dump(structure_config, f, indent=2)
        f.write("\n")
    return output_config_path, output_structure_path


def main(args):
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")

    with open(args.grouter_config_path, "r") as f:
        grouter_config = json.load(f)

    model = grt(**grouter_config, output_logits=True).to(device=device)
    args.topk = grouter_config.get("topk", None)
    current_num_experts = grouter_config.get("target_num_experts", grouter_config.get("num_experts"))
    args.num_experts = int(current_num_experts)
    if args.bf16:
        model = model.to(torch.bfloat16)

    checkpoint = _load_checkpoint(args.grouter_checkpoint_path, device=device)
    # for history compatible
    if "bias" not in checkpoint and hasattr(model, "bias"):
        checkpoint["bias"] = torch.zeros(model.num_experts, device=device)
    model.load_state_dict(checkpoint, strict=False)
    model.eval()

    dataloader = _build_dataloader(args=args, rank=rank, world_size=world_size)

    if rank == 0:
        old_num = _get_router_linear(model).weight.shape[0]
        print(f"[expand] start experts: {old_num}, target experts: {args.target_num_experts}")

    expand_moe_experts(
        model=model,
        data_loader=dataloader,
        device=device,
        rank=rank,
        world_size=world_size,
        target_num_experts=args.target_num_experts,
        total_steps=args.total_steps,
        bad_token_quantile=args.bad_token_quantile,
        kmeans_iters=args.kmeans_iters,
        random_seed=args.random_seed,
        verbose=args.verbose and rank == 0,
    )

    if rank == 0:
        output_dir = os.path.dirname(args.output_checkpoint_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        torch.save(model.state_dict(), args.output_checkpoint_path)
        print(f"[expand] expanded checkpoint saved to {args.output_checkpoint_path}")
        saved_cfg, saved_structure_cfg = _save_expanded_configs(
            grouter_config_path=args.grouter_config_path,
            grouter_config=grouter_config,
            target_num_experts=args.target_num_experts,
            output_dir=output_dir or ".",
        )
        print(f"[expand] updated config saved to {saved_cfg}")
        print(f"[expand] updated structure config saved to {saved_structure_cfg}")


if __name__ == "__main__":
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        dist.init_process_group(backend="nccl")

    try:
        parsed_args = parse_args(None)
        main(parsed_args)
    finally:
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()
