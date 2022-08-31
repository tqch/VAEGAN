from .gan import GAN
from .vae import VAE
from .vaegan import VAEGAN

MODEL_LIST = {
    "gan": GAN,
    "vae": VAE,
    "vaegan": VAEGAN
}

__all__ = ["MODEL_LIST"]