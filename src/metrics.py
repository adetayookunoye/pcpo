
import jax.numpy as jnp
from constraint_lib.divergence_free import divergence, vorticity as vort_fn
from constraint_lib.energy import kinetic_energy, energy_error
import jax

def l2(a, b):
    return jnp.sqrt(jnp.mean((a - b) ** 2))

def avg_divergence(u, v):
    return jnp.mean(jnp.abs(divergence(u, v)))

def energy_conservation(u_pred, v_pred, u_ref, v_ref):
    return energy_error(u_pred, v_pred, u_ref, v_ref)

def vorticity(u, v):
    return vort_fn(u, v)

def vorticity_l2(u_pred, v_pred, u_ref, v_ref):
    return l2(vorticity(u_pred, v_pred), vorticity(u_ref, v_ref))

def enstrophy(field_u, field_v):
    w = vorticity(field_u, field_v)
    return 0.5 * jnp.mean(w ** 2)

def enstrophy_rel_err(up, vp, ur, vr):
    ep = enstrophy(up, vp)
    er = enstrophy(ur, vr)
    return jnp.mean(jnp.abs(ep - er) / (er + 1e-8))

def spectrum(field):
    ft = jnp.fft.fft2(field)
    ps = (ft * jnp.conj(ft)).real
    return ps

def spectra_distance(u_pred, v_pred, u_ref, v_ref):
    """Compute normalized L2 distance between spectral energy distributions.
    
    Uses log-binned spectra for better resolution across scales.
    """
    sp_pred = spectrum(u_pred) + spectrum(v_pred)
    sp_ref = spectrum(u_ref) + spectrum(v_ref)
    
    # Normalize by reference energy to make metric scale-invariant
    ref_energy = jnp.sqrt(jnp.mean(sp_ref ** 2))
    if ref_energy < 1e-10:
        return 0.0
    
    normalized_dist = l2(sp_pred, sp_ref) / ref_energy
    return normalized_dist


def _log_binned_energy_spectrum(u: jnp.ndarray, v: jnp.ndarray, num_bins: int = 32) -> jnp.ndarray:
    height, width = u.shape[-2], u.shape[-1]
    kx = jnp.fft.fftfreq(width) * width
    ky = jnp.fft.fftfreq(height) * height
    kx_grid, ky_grid = jnp.meshgrid(kx, ky, indexing="xy")
    k_mag = jnp.sqrt(kx_grid**2 + ky_grid**2)
    energy = spectrum(u) + spectrum(v)

    log_k = jnp.log(k_mag + 1.0)
    max_mode = jnp.sqrt((height // 2) ** 2 + (width // 2) ** 2) + 1.0
    log_k_max = jnp.log(max_mode)
    edges = jnp.linspace(0.0, log_k_max, num_bins + 1, dtype=jnp.float32)
    lowers = edges[:-1]
    uppers = edges[1:]
    idxs = jnp.arange(num_bins, dtype=jnp.int32)

    def bin_energy(idx, lower, upper):
        upper_adj = jnp.where(idx == num_bins - 1, upper + 1e-6, upper)
        mask = (log_k >= lower) & (log_k < upper_adj)
        energy_sum = jnp.sum(jnp.where(mask, energy, 0.0))
        count = jnp.sum(mask)
        return jnp.where(count > 0, energy_sum / count, 0.0)

    return jax.vmap(bin_energy)(idxs, lowers, uppers)


def spectral_log_mse(y_pred: jnp.ndarray, y_true: jnp.ndarray, num_bins: int = 32, eps: float = 1e-8) -> jnp.ndarray:
    if y_pred.ndim == 3:
        y_pred = y_pred[None, ...]
        y_true = y_true[None, ...]

    def sample_bins(sample):
        return _log_binned_energy_spectrum(sample[..., 0], sample[..., 1], num_bins=num_bins)

    pred_bins = jax.vmap(sample_bins)(y_pred)
    true_bins = jax.vmap(sample_bins)(y_true)

    pred_mean = jnp.mean(pred_bins, axis=0)
    true_mean = jnp.mean(true_bins, axis=0)
    log_diff = jnp.log(pred_mean + eps) - jnp.log(true_mean + eps)
    return jnp.mean(log_diff**2)

def pairwise_l2(xs):
    # xs: (N,B,H,W,2) or (N, ...)
    n = xs.shape[0]
    dsum = 0.0
    cnt = 0
    for i in range(n):
        for j in range(i+1, n):
            dsum += jnp.sqrt(jnp.mean((xs[i] - xs[j])**2))
            cnt += 1
    return dsum / (cnt + 1e-8)

def sample_aggregate(pred_fn, x, key, n_samples=8):
    keys = jax.random.split(key, n_samples)
    samples = []
    for subkey in keys:
        samples.append(pred_fn(x, subkey))
    return jnp.stack(samples, axis=0)

def sharpness(Ys):
    return jnp.mean(jnp.var(Ys, axis=0))

def empirical_coverage(Ys, y_true, alpha=0.1):
    lower = jnp.quantile(Ys, alpha / 2.0, axis=0)
    upper = jnp.quantile(Ys, 1.0 - alpha / 2.0, axis=0)
    inside = jnp.logical_and(y_true >= lower, y_true <= upper)
    return jnp.mean(inside.astype(jnp.float32))

def crps_samples(Ys, y_true):
    s = Ys.shape[0]
    term1 = jnp.mean(jnp.abs(Ys - y_true), axis=0)
    diffs = jnp.abs(Ys[:, None] - Ys[None, :])
    term2 = jnp.mean(diffs, axis=(0, 1))
    crps = jnp.mean(term1) - 0.5 * jnp.mean(term2)
    return crps

def pde_residual_surrogate(u, v, nu=1e-3):
    def lap(f):
        return (-4.0 * f +
                jnp.roll(f, 1, axis=-1) + jnp.roll(f, -1, axis=-1) +
                jnp.roll(f, 1, axis=-2) + jnp.roll(f, -1, axis=-2))
    ru = jnp.abs(nu * lap(u))
    rv = jnp.abs(nu * lap(v))
    return jnp.mean(ru + rv)
