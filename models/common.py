import importlib

def get_model(cfg):
    """ Load model
    Args:
        cfg: config
    Returns:
        model: model
    """
    CommonNet = importlib.import_module(cfg.model).CommonNet
    model = CommonNet(cfg)
    return model