import jax
import jax.numpy as jnp
import equinox as eqx

def complx(shape, key):
    """Initialize complex weights with proper scaling"""
    a = jax.random.normal(key, shape)
    b = jax.random.normal(key, shape)
    return (a + 1j * b) * 0.01  # Added scaling

class SpectralConv2d(eqx.Module):
    modes1: int
    modes2: int
    in_channels: int
    out_channels: int
    weights1: jnp.ndarray  # Complex weights for first frequency block
    weights2: jnp.ndarray  # Complex weights for second frequency block

    def __init__(self, in_channels, out_channels, modes1, modes2, key):
        self.modes1 = modes1
        self.modes2 = modes2
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        key1, key2 = jax.random.split(key)
        # Initialize complex weights
        self.weights1 = complx((in_channels, out_channels, modes1, modes2), key1)
        self.weights2 = complx((in_channels, out_channels, modes1, modes2), key2)

    def __call__(self, x):
        """
        x: (B, H, W, C_in)
        returns: (B, H, W, C_out)
        """
        batch_size, h, w, in_ch = x.shape
        
        # Compute Fourier coefficients
        x_ft = jnp.fft.rfft2(x, axes=(1, 2))  # (B, H, Wc, C_in)
        batch_size, h_ft, w_ft, in_ch = x_ft.shape
        
        # Initialize output Fourier coefficients
        out_ft = jnp.zeros((batch_size, h_ft, w_ft, self.out_channels), dtype=jnp.complex64)
        
        # Process low-frequency modes
        m1 = min(self.modes1, h_ft)
        m2 = min(self.modes2, w_ft)
        
        # Multiply relevant Fourier modes
        # First block: low frequencies in both dimensions
        out_ft_block1 = jnp.einsum("bxyi,ioxy->bxyo", 
                                  x_ft[:, :m1, :m2, :], 
                                  self.weights1[:, :, :m1, :m2])
        out_ft = out_ft.at[:, :m1, :m2, :].add(out_ft_block1)
        
        # Second block: high frequencies in first dimension, low in second
        if h_ft > self.modes1:
            out_ft_block2 = jnp.einsum("bxyi,ioxy->bxyo", 
                                      x_ft[:, -self.modes1:, :m2, :], 
                                      self.weights2[:, :, :self.modes1, :m2])
            out_ft = out_ft.at[:, -self.modes1:, :m2, :].add(out_ft_block2)
        
        # Transform back to physical space
        x_out = jnp.fft.irfft2(out_ft, s=(h, w), axes=(1, 2))
        return x_out

class FNO2d(eqx.Module):
    conv_layers: list
    w_in: jnp.ndarray
    w_out: jnp.ndarray

    def __init__(self, in_ch, out_ch, width, modes, depth, key):
        keys = jax.random.split(key, depth + 2)
        
        # Simple weight matrices (equivalent to 1x1 convolutions)
        self.w_in = jax.random.normal(keys[0], (in_ch, width)) * 0.02
        self.w_out = jax.random.normal(keys[1], (width, out_ch)) * 0.02
        
        # Spectral convolution layers
        self.conv_layers = []
        for i in range(depth):
            conv_key = keys[i + 2]
            conv_layer = SpectralConv2d(width, width, modes, modes, conv_key)
            self.conv_layers.append(conv_layer)

    def __call__(self, x):
        """
        x: (B, H, W, C_in)
        returns: (B, H, W, C_out)
        """
        # Input projection using einsum
        x = jnp.einsum("bhwc,co->bhwo", x, self.w_in)
        
        # FNO layers
        for conv_layer in self.conv_layers:
            x_res = conv_layer(x)
            x = x + x_res  # Residual connection
            x = jax.nn.gelu(x)
        
        # Output projection
        x = jnp.einsum("bhwc,co->bhwo", x, self.w_out)
        return x
