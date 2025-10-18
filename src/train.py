
import argparse, os
import jax, jax.numpy as jnp
import numpy as np
import optax
import equinox as eqx
from tqdm import trange

from src.utils import load_config, makedirs, save_json
from src.metrics import l2, avg_divergence, energy_conservation, vorticity_l2, spectral_log_mse
from src.model_wrappers import AmplitudeWrapper

def _coord_grid(grid_size: int) -> jnp.ndarray:
    xs = jnp.linspace(0.0, 1.0, grid_size, dtype=jnp.float32)
    ys = jnp.linspace(0.0, 1.0, grid_size, dtype=jnp.float32)
    xv, yv = jnp.meshgrid(xs, ys, indexing="xy")
    coords = jnp.stack([xv, yv], axis=-1)  # (H, W, 2)
    return jnp.reshape(coords, (-1, 2))
from src.data.pdebench_ns2d import build_data_source
from models.fno import FNO2d
from models.divfree_fno import DivFreeFNO
from models.cvae_fno import CVAEFNO
from models.pino import PINO
from models.bayesian_deeponet import BayesDeepONet

def get_model(name, cfg, key):
    depth = cfg["model"].get("depth", 4)
    if name == "fno":
        return FNO2d(in_ch=2, out_ch=2, width=cfg["model"]["width"], modes=cfg["model"]["modes"],
                     depth=depth, key=key)
    if name == "divfree_fno":
        return DivFreeFNO(width=cfg["model"]["width"], modes=cfg["model"]["modes"], depth=depth, key=key)
    if name == "cvae_fno":
        return CVAEFNO(in_ch=2, latent_dim=cfg["model"]["latent_dim"], width=cfg["model"]["width"],
                       modes=cfg["model"]["modes"], depth=depth, beta=cfg["model"]["cvae_beta"], key=key)
    if name == "pino":
        pino_cfg = cfg.get("pino", {})
        return PINO(in_ch=2, out_ch=2, width=cfg["model"]["width"],
                    modes=cfg["model"]["modes"], depth=depth,
                    physics_weight=pino_cfg.get("physics_weight", cfg["model"].get("physics_weight", 0.0)),
                    nu=pino_cfg.get("nu", cfg["model"].get("nu", 1e-3)), key=key)
    if name == "bayes_deeponet":
        grid = cfg["data"]["grid_size"]
        branch_dim = grid * grid * 2
        latent = cfg["model"].get("latent_dim", 64)
        hidden = cfg["model"].get("deep_hidden", [64, 64, 64])
        return BayesDeepONet(
            branch_input_dim=branch_dim,
            trunk_input_dim=2,
            hidden_dims=hidden,
            latent_dim=latent,
            output_dim=2,
            key=key,
        )
    raise ValueError(f"Unknown model: {name}")


def _format_row(row: np.ndarray, max_cols: int = 8) -> str:
    cols = min(len(row), max_cols)
    core = " ".join(f"{float(val): .4f}" for val in row[:cols])
    suffix = " ..." if len(row) > cols else ""
    return f"[{core}{suffix}]"


def preview_training_samples(data_source, count: int, key: jax.Array) -> jax.Array:
    if count <= 0:
        return key

    print(f"\n--- Previewing first {count} training samples ---")

    if hasattr(data_source, "x") and hasattr(data_source, "y"):
        xs = np.asarray(data_source.x[:count])
        ys = np.asarray(data_source.y[:count])
        actual = xs.shape[0]
        for idx in range(actual):
            x_sample = xs[idx]
            y_sample = ys[idx]
            first_u = _format_row(x_sample[0, :, 0])
            first_v = _format_row(x_sample[0, :, 1]) if x_sample.shape[-1] > 1 else ""
            target_u = _format_row(y_sample[0, :, 0])
            target_v = _format_row(y_sample[0, :, 1]) if y_sample.shape[-1] > 1 else ""
            print(f"Sample {idx:02d} | x[u0]: {first_u} | x[v0]: {first_v}")
            print(f"           | y[u0]: {target_u} | y[v0]: {target_v}")
    else:
        for idx in range(count):
            key, sample_key = jax.random.split(key)
            x_batch, y_batch = data_source.sample_batch(sample_key, batch_size=1, normalize=False)
            x_sample = np.asarray(x_batch[0])
            y_sample = np.asarray(y_batch[0])
            first_u = _format_row(x_sample[0, :, 0])
            first_v = _format_row(x_sample[0, :, 1]) if x_sample.shape[-1] > 1 else ""
            target_u = _format_row(y_sample[0, :, 0])
            target_v = _format_row(y_sample[0, :, 1]) if y_sample.shape[-1] > 1 else ""
            print(f"Sample {idx:02d} | x[u0]: {first_u} | x[v0]: {first_v}")
            print(f"           | y[u0]: {target_u} | y[v0]: {target_v}")

    print("--- End of preview ---\n")
    return key


def get_loss_weights(loss_cfg: dict, model_name: str) -> dict:
    weights = {
        "l2": float(loss_cfg.get("l2", 1.0)),
        "div": float(loss_cfg.get("div", 0.0)),
        "energy": float(loss_cfg.get("energy", 0.0)),
        "vorticity_l2": float(loss_cfg.get("vorticity_l2", 0.0)),
        "h1": float(loss_cfg.get("h1", 0.0)),
        "spectral": float(loss_cfg.get("spectral", loss_cfg.get("spectral_mse", 0.0))),
    }
    if model_name == "divfree_fno":
        weights["div"] = 0.0
    elif weights["div"] == 0.0 and model_name in ("fno", "pino", "cvae_fno", "bayes_deeponet"):
        weights["div"] = 1.0
    return weights


def h1_seminorm(pred: jnp.ndarray, target: jnp.ndarray) -> jnp.ndarray:
    diff = pred - target
    grad_u_y, grad_u_x = jnp.gradient(diff[..., 0], axis=(1, 2))
    grad_v_y, grad_v_x = jnp.gradient(diff[..., 1], axis=(1, 2))
    seminorm = jnp.mean(
        grad_u_y**2 + grad_u_x**2 + grad_v_y**2 + grad_v_x**2
    )
    return seminorm


def compute_weighted_terms(y_pred, y_true, weights):
    metrics = {}
    metrics["l2"] = jnp.mean((y_pred - y_true) ** 2)
    metrics["div"] = avg_divergence(y_pred[..., 0], y_pred[..., 1])
    metrics["energy"] = energy_conservation(
        y_pred[..., 0], y_pred[..., 1], y_true[..., 0], y_true[..., 1]
    )
    metrics["vorticity_l2"] = vorticity_l2(
        y_pred[..., 0], y_pred[..., 1], y_true[..., 0], y_true[..., 1]
    )
    metrics["h1"] = h1_seminorm(y_pred, y_true)
    metrics["spectral"] = spectral_log_mse(y_pred, y_true)

    total = jnp.array(0.0)
    for name, value in metrics.items():
        weight = float(weights.get(name, 0.0))
        if weight > 0.0:
            total = total + weight * value
    return total, metrics


def clip_gradients(grads, clip_norm):
    if clip_norm is None or clip_norm <= 0.0:
        return grads
    g_norm = optax.global_norm(grads)
    factor = jnp.minimum(1.0, clip_norm / (g_norm + 1e-6))
    return jax.tree_util.tree_map(lambda g: factor * g, grads)

def train_step_divfree(model, opt, opt_state, batch, key, weights, clip_norm):
    x, y = batch  # (B,H,W,2)
    def loss_fn(m):
        y_pred = m(x)
        total_loss, metrics = compute_weighted_terms(y_pred, y, weights)
        amp_reg = m.amplitude_regularizer()
        total_loss = total_loss + amp_reg
        metrics["amp_reg"] = amp_reg
        metrics["amp_gain"] = m.gain()
        metrics["total_loss"] = total_loss
        return total_loss, metrics
    (loss, logs), grads = eqx.filter_value_and_grad(loss_fn, has_aux=True)(model)
    grads = eqx.filter(grads, eqx.is_array)
    grads = clip_gradients(grads, clip_norm)
    params = eqx.filter(model, eqx.is_array)
    updates, opt_state = opt.update(grads, opt_state, params)
    model = eqx.apply_updates(model, updates)
    return model, opt_state, logs

def train_step_cvae(model, opt, opt_state, batch, key, epoch, schedule_cfg, weights, clip_norm):
    x, y = batch
    warmup_epochs = int(schedule_cfg.get("kl_warmup_epochs", 50))
    target_beta = float(schedule_cfg.get("kl_target_beta", 1.0))
    free_bits = float(schedule_cfg.get("kl_free_bits", 0.0))
    noise_std = float(schedule_cfg.get("cvae_input_noise", 0.0))
    if warmup_epochs > 0:
        beta_coef = min(1.0, epoch / warmup_epochs)
    else:
        beta_coef = 1.0
    current_beta = target_beta * beta_coef
    x_in = x
    if noise_std > 0.0:
        key, noise_key = jax.random.split(key)
        noise = noise_std * jax.random.normal(noise_key, x.shape)
        x_in = x + noise
    def loss_fn(m):
        y_pred, mu, logvar = m(x_in, key)
        base_loss, metrics = compute_weighted_terms(y_pred, y, weights)
        kl_raw = -0.5 * (1 + logvar - mu**2 - jnp.exp(logvar))
        if free_bits > 0.0:
            kl = jnp.mean(jnp.maximum(kl_raw, free_bits))
        else:
            kl = jnp.mean(kl_raw)
        total_loss = base_loss + current_beta * kl
        amp_reg = m.amplitude_regularizer()
        total_loss = total_loss + amp_reg
        metrics.update({"kl": kl, "beta": current_beta})
        metrics["amp_reg"] = amp_reg
        metrics["amp_gain"] = m.gain()
        metrics["total_loss"] = total_loss
        return total_loss, metrics
    (loss, logs), grads = eqx.filter_value_and_grad(loss_fn, has_aux=True)(model)
    grads = eqx.filter(grads, eqx.is_array)
    grads = clip_gradients(grads, clip_norm)
    params = eqx.filter(model, eqx.is_array)
    updates, opt_state = opt.update(grads, opt_state, params)
    model = eqx.apply_updates(model, updates)
    return model, opt_state, logs

def train_step_bdeeponet(model, opt, opt_state, batch, coord_flat, weights, clip_norm):
    x, y = batch
    B, H, W, C = x.shape
    branch_input = jnp.reshape(x, (B, H * W * C))
    trunk = jnp.broadcast_to(coord_flat[None, :, :], (B, coord_flat.shape[0], coord_flat.shape[1]))

    def loss_fn(m):
        mean, variance = m(branch_input, trunk)
        mean = jnp.reshape(mean, (B, H, W, -1))
        variance = jnp.reshape(variance, (B, H, W, -1))
        var_safe = variance + 1e-6
        nll = jnp.mean((y - mean) ** 2 / var_safe + jnp.log(var_safe))
        base_loss, metrics = compute_weighted_terms(mean, y, weights)
        total = base_loss + nll
        metrics.update({"nll": nll})
        amp_reg = m.amplitude_regularizer()
        total = total + amp_reg
        metrics["amp_reg"] = amp_reg
        metrics["amp_gain"] = m.gain()
        metrics["total_loss"] = total
        return total, metrics

    (loss, logs), grads = eqx.filter_value_and_grad(loss_fn, has_aux=True)(model)
    grads = eqx.filter(grads, eqx.is_array)
    grads = clip_gradients(grads, clip_norm)
    params = eqx.filter(model, eqx.is_array)
    updates, opt_state = opt.update(grads, opt_state, params)
    model = eqx.apply_updates(model, updates)
    return model, opt_state, logs

def train_step_pino(model, opt, opt_state, batch, weights, physics_weight, clip_norm):
    x, y = batch
    def loss_fn(m):
        y_pred = m(x)
        base_loss, metrics = compute_weighted_terms(y_pred, y, weights)
        physics = m.physics_loss(x, y_pred)
        total = base_loss + physics_weight * physics
        amp_reg = m.amplitude_regularizer()
        total = total + amp_reg
        metrics.update({"physics_loss": physics, "amp_reg": amp_reg, "amp_gain": m.gain()})
        metrics["total_loss"] = total
        return total, metrics
    (loss, logs), grads = eqx.filter_value_and_grad(loss_fn, has_aux=True)(model)
    grads = eqx.filter(grads, eqx.is_array)
    grads = clip_gradients(grads, clip_norm)
    params = eqx.filter(model, eqx.is_array)
    updates, opt_state = opt.update(grads, opt_state, params)
    model = eqx.apply_updates(model, updates)
    return model, opt_state, logs

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--model", type=str, default="divfree_fno", choices=["fno","divfree_fno","cvae_fno","pino","bayes_deeponet"])
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--quickrun", action="store_true")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--lambda_phys", type=float, default=None, help="Override PINO physics weight")
    args = parser.parse_args()

    cfg = load_config(args.config)
    tcfg = cfg["train"]

    if args.seed is not None:
        cfg["seed"] = int(args.seed)
    if args.lambda_phys is not None:
        cfg.setdefault("model", {})["physics_weight"] = float(args.lambda_phys)

    seed = cfg.get("seed", 42)
    key = jax.random.PRNGKey(seed)

    results_dir = cfg["outputs"]["results_dir"]
    ckpt_dir = cfg["outputs"]["checkpoints_dir"].format(model=args.model)
    makedirs(results_dir)
    makedirs(ckpt_dir)

    model = get_model(args.model, cfg, key)
    amp_cfg = cfg["model"]
    amp_enabled = bool(amp_cfg.get("amplitude_calibration", True))
    amp_init = float(amp_cfg.get("amplitude_init", 1.0))
    amp_reg = float(amp_cfg.get("amplitude_reg", 1e-6))
    model = AmplitudeWrapper(
        model,
        init_gain=amp_init,
        reg_weight=amp_reg,
        enabled=amp_enabled,
    )
    epochs = args.epochs or (1 if args.quickrun else tcfg["epochs"])
    batch_size = tcfg["batch_size"]
    H = cfg["data"]["grid_size"]
    W = cfg["data"]["grid_size"]

    data_source, manifest = build_data_source(cfg["data"])
    if manifest:
        print(
            f"Loaded PDEBench data: {manifest['total_pairs']} pairs "
            f"from {len(manifest['processed_files'])} processed files."
        )
    else:
        print("Using synthetic Navier–Stokes data (real data not found).")

    train_size_attr = getattr(data_source, "size", None)
    if train_size_attr is None:
        synthetic_samples = cfg["data"].get("synthetic_samples")
        if synthetic_samples is None:
            train_size = batch_size
        else:
            train_size = int(synthetic_samples)
    else:
        train_size = int(train_size_attr)

    lr = float(tcfg["lr"])
    weight_decay = float(tcfg.get("weight_decay", 0.0))
    schedule_cfg = tcfg.get("lr_schedule", {}) or {}
    if isinstance(schedule_cfg, dict) and schedule_cfg.get("type", "").lower() == "cosine":
        num_train = max(1, train_size)
        steps_per_epoch = max(1, (num_train + batch_size - 1) // batch_size)
        total_steps = max(1, steps_per_epoch * epochs)

        cfg_sched = schedule_cfg
        warmup_steps = int(cfg_sched.get("warmup_steps", 0))
        decay_cfg = cfg_sched.get("decay_steps", None)
        if isinstance(decay_cfg, str) and decay_cfg.lower() == "total_steps:auto":
            schedule_total_steps = total_steps
        elif decay_cfg is None:
            schedule_total_steps = total_steps
        else:
            schedule_total_steps = max(1, int(decay_cfg))

        if warmup_steps >= schedule_total_steps:
            warmup_steps = max(0, schedule_total_steps - 1)
        else:
            warmup_steps = max(0, warmup_steps)

        cosine_steps = max(1, schedule_total_steps - warmup_steps)

        lr_schedule = optax.warmup_cosine_decay_schedule(
            init_value=0.0,
            peak_value=lr,
            warmup_steps=warmup_steps,
            decay_steps=cosine_steps,
            end_value=0.0,
        )
        opt = optax.adamw(learning_rate=lr_schedule, weight_decay=weight_decay)
    else:
        opt = optax.adamw(lr, weight_decay=weight_decay)
    opt_state = opt.init(eqx.filter(model, eqx.is_array))

    clip_norm = tcfg.get("grad_clip_norm")
    if clip_norm is not None:
        clip_norm = float(clip_norm)
    ema_decay = float(tcfg.get("ema_decay", 0.0))
    ema_params = None
    if ema_decay:
        ema_params = eqx.filter(model, eqx.is_array)

    loss_cfg = cfg.get("loss", {})
    weights = get_loss_weights(loss_cfg, args.model)
    pino_cfg = cfg.get("pino", {})
    physics_weight = float(pino_cfg.get("physics_weight", cfg["model"].get("physics_weight", 0.0)))

    preview_rows = int(tcfg.get("preview_rows", 0))
    if preview_rows > 0:
        key = preview_training_samples(data_source, preview_rows, key)

    coord_flat = _coord_grid(H) if args.model == "bayes_deeponet" else None

    history = []
    for ep in trange(epochs, desc="Training"):
        key, data_key = jax.random.split(key)
        x, y = data_source.sample_batch(data_key, batch_size)
        key, step_key = jax.random.split(key)
        if args.model in ("fno", "divfree_fno"):
            model, opt_state, logs = train_step_divfree(model, opt, opt_state, (x, y), step_key, weights, clip_norm)
        elif args.model == "cvae_fno":
            model, opt_state, logs = train_step_cvae(model, opt, opt_state, (x, y), step_key, ep+1, tcfg, weights, clip_norm)
        elif args.model == "bayes_deeponet":
            model, opt_state, logs = train_step_bdeeponet(model, opt, opt_state, (x, y), coord_flat, weights, clip_norm)
        else:
            model, opt_state, logs = train_step_pino(model, opt, opt_state, (x, y), weights, physics_weight, clip_norm)
        if ema_params is not None:
            current_params = eqx.filter(model, eqx.is_array)
            ema_params = jax.tree_util.tree_map(
                lambda ema, p: ema_decay * ema + (1.0 - ema_decay) * p,
                ema_params,
                current_params,
            )
        logs = {k: float(v) for k, v in logs.items()}
        logs["epoch"] = int(ep+1)
        history.append(logs)

        if (ep+1) % tcfg["save_every"] == 0:
            ckpt_path = os.path.join(ckpt_dir, "last_ckpt.npz")
            eqx.tree_serialise_leaves(ckpt_path, model)

    if ema_params is not None:
        _, static = eqx.partition(model, eqx.is_array)
        model = eqx.combine(ema_params, static)

    save_json({"history": history}, os.path.join(results_dir, f"{args.model}_train_history.json"))
    print("Training complete. Checkpoints and history saved.")

if __name__ == "__main__":
    main()
