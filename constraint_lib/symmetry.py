
# Symmetry helpers (placeholder)
def normalize_field(x):
    x_min = x.min()
    x_max = x.max()
    return (x - x_min) / (x_max - x_min + 1e-8)
