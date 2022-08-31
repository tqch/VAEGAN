import torch.autograd

if __name__ == "__main__":
    import sys
    import models.unet
    import models.gan
    import models.vae
    import models.vaegan

    # torch.autograd.set_detect_anomaly(True)

    script = sys.argv[1]
    test = {
        "unet": models.unet.test,
        "gan": models.gan.test,
        "vae": models.vae.test,
        "vaegan": models.vaegan.test
    }[script]()
