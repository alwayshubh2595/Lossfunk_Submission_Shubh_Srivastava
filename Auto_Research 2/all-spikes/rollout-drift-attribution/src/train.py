from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict

import numpy as np
import torch

from .envs import SequenceDataset, make_datasets
from .metrics import evaluate_model
from .models import WorldModel
from .regularizers import regularizer_loss


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--env",
        choices=["stable-linear", "stable-linear-det", "ou", "lorenz", "pointmass-state", "pointmass"],
        default="stable-linear",
    )
    parser.add_argument("--regularizer", choices=["none", "moment", "sigreg"], default="sigreg")
    parser.add_argument("--protocol", choices=["1step", "hstep-tf", "bptt"], default="bptt")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=20_000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--eval-horizon", type=int, default=50)
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--latent-dim", type=int, default=None)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--lambda-reg", type=float, default=0.1)
    parser.add_argument("--sigreg-projections", type=int, default=256)
    parser.add_argument("--sigreg-gamma", type=float, default=0.5)
    parser.add_argument("--sigreg-max-samples", type=int, default=256)
    parser.add_argument("--n-train", type=int, default=None)
    parser.add_argument("--n-val", type=int, default=None)
    parser.add_argument("--n-test", type=int, default=None)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-dir", default="runs")
    parser.add_argument("--log-every", type=int, default=500)
    parser.add_argument("--save-model", action="store_true")
    parser.add_argument("--load-encoder-from", default=None, help="path to model.pt to freeze the encoder from (encoder-swap mechanism probe)")
    parser.add_argument(
        "--reg-target",
        choices=["encoded", "encoded-and-rollout"],
        default="encoded",
        help="encoded = main spec (R applied only to enc(o_t..o_t+H)); "
        "encoded-and-rollout = ablation (R also applied to rolled-out z_hat under bptt, "
        "at the same lambda as the encoder term; the dose-matched control is a separate "
        "run with --reg-target encoded and doubled --lambda-reg)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = choose_device(args.device)

    visual = args.env == "pointmass"
    if args.n_train is None:
        args.n_train = 800 if visual else 3000
    if args.n_val is None:
        args.n_val = 128 if visual else 512
    if args.n_test is None:
        args.n_test = 128 if visual else 512
    if args.latent_dim is None:
        args.latent_dim = 32 if visual else 16

    datasets = make_datasets(
        args.env,
        seed=args.seed,
        n_train=args.n_train,
        n_val=args.n_val,
        n_test=args.n_test,
        seq_len=args.seq_len,
        image_size=args.image_size,
    )
    train = datasets["train"]
    val = datasets["val"]
    model = WorldModel(train.obs_shape, train.action_dim, args.latent_dim, train.visual).to(device)

    trainable_params = model.parameters()
    if args.load_encoder_from is not None:
        state = torch.load(args.load_encoder_from, map_location=device)
        encoder_state = {k: v for k, v in state.items() if k.startswith("encoder.")}
        model.load_state_dict(encoder_state, strict=False)
        for name, p in model.named_parameters():
            if name.startswith("encoder."):
                p.requires_grad_(False)
        trainable_params = [p for p in model.parameters() if p.requires_grad]

    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)

    run_dir = make_run_dir(args)
    history_path = run_dir / "history.jsonl"
    args_path = run_dir / "args.json"
    args_path.write_text(json.dumps(vars(args), indent=2, sort_keys=True) + "\n")

    last_log: Dict[str, float] = {}
    for step in range(1, args.steps + 1):
        model.train()
        obs, actions = sample_training_batch(train, args.batch_size, args.horizon, device)
        pred_loss, z_encoded, z_rollout = compute_prediction_loss(model, obs, actions, args.protocol)
        reg_loss = regularizer_loss(
            args.regularizer,
            z_encoded,
            projections=args.sigreg_projections,
            gamma=args.sigreg_gamma,
        )
        if args.reg_target == "encoded-and-rollout" and z_rollout is not None:
            reg_loss = reg_loss + regularizer_loss(
                args.regularizer,
                z_rollout,
                projections=args.sigreg_projections,
                gamma=args.sigreg_gamma,
            )
        loss = pred_loss + args.lambda_reg * reg_loss

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        optimizer.step()

        if step == 1 or step % args.log_every == 0 or step == args.steps:
            last_log = {
                "step": step,
                "loss": float(loss.item()),
                "pred_loss": float(pred_loss.item()),
                "reg_loss": float(reg_loss.item()),
            }
            with history_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(last_log, sort_keys=True) + "\n")
            print(json.dumps(last_log, sort_keys=True), flush=True)

    metrics = evaluate_model(
        model,
        train,
        val,
        device=device,
        eval_horizon=min(args.eval_horizon, args.seq_len),
        batch_size=min(args.batch_size, 256),
    )
    metrics.update({f"final_{k}": v for k, v in last_log.items() if k != "step"})
    metrics.update(
        {
            "env": args.env,
            "regularizer": args.regularizer,
            "protocol": args.protocol,
            "seed": args.seed,
            "steps": args.steps,
            "latent_dim": args.latent_dim,
            "lambda_reg": args.lambda_reg,
            "sigreg_projections": args.sigreg_projections,
            "reg_target": args.reg_target,
            "load_encoder_from": args.load_encoder_from,
        }
    )
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
    if args.save_model:
        torch.save(model.state_dict(), run_dir / "model.pt")
    print(json.dumps(metrics, indent=2, sort_keys=True), flush=True)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def make_run_dir(args: argparse.Namespace) -> Path:
    name = f"{args.env}__{args.regularizer}__{args.protocol}__seed{args.seed}"
    root = Path(args.output_dir)
    run_dir = root / name
    suffix = 1
    while run_dir.exists():
        suffix += 1
        run_dir = root / f"{name}__r{suffix}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def sample_training_batch(
    dataset: SequenceDataset,
    batch_size: int,
    horizon: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    max_t = dataset.seq_len - horizon
    if max_t < 0:
        raise ValueError(f"horizon {horizon} exceeds sequence length {dataset.seq_len}")
    idx = torch.randint(0, dataset.obs.shape[0], (batch_size,))
    t0 = torch.randint(0, max_t + 1, (batch_size,))
    steps_obs = torch.arange(horizon + 1)
    steps_act = torch.arange(horizon)
    obs = dataset.obs[idx[:, None], t0[:, None] + steps_obs[None, :]].to(device)
    actions = dataset.actions[idx[:, None], t0[:, None] + steps_act[None, :]].to(device)
    return obs, actions


def compute_prediction_loss(
    model: WorldModel,
    obs: torch.Tensor,
    actions: torch.Tensor,
    protocol: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Returns (pred_loss, z_encoded[B,T,D], z_rollout[B,H,D] or None).

    z_encoded is the (B, T, D) tensor of encoded observations the regularizer
    is applied to under the main spec (T=2 under 1-step, T=H+1 under
    hstep-tf/bptt). z_rollout is the model's own rolled-out predictions
    (only populated under bptt), used only by the reg-target=encoded-and-
    rollout ablation.
    """
    bsz, horizon_plus_one = obs.shape[:2]
    horizon = horizon_plus_one - 1
    z_obs = model.encode_flat(obs.reshape(-1, *obs.shape[2:])).reshape(bsz, horizon_plus_one, -1)

    if protocol == "1step":
        z_hat = model.dynamics(z_obs[:, 0], actions[:, 0])
        pred_loss = (z_hat - z_obs[:, 1]).square().mean()
        return pred_loss, z_obs[:, :2], None

    if protocol == "hstep-tf":
        losses = []
        for k in range(1, horizon + 1):
            z_hat = model.dynamics(z_obs[:, k - 1], actions[:, k - 1])
            losses.append((z_hat - z_obs[:, k]).square().mean())
        pred_loss = torch.stack(losses).mean()
        return pred_loss, z_obs, None

    if protocol == "bptt":
        pred = z_obs[:, 0]
        losses = []
        rollout = []
        for k in range(1, horizon + 1):
            pred = model.dynamics(pred, actions[:, k - 1])
            losses.append((pred - z_obs[:, k]).square().mean())
            rollout.append(pred)
        pred_loss = torch.stack(losses).mean()
        z_rollout = torch.stack(rollout, dim=1)
        return pred_loss, z_obs, z_rollout

    raise ValueError(f"unknown protocol: {protocol}")


if __name__ == "__main__":
    main()
