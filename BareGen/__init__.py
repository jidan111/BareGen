from .autoencoder import *
from .clip import *
from .diffusion import *
from .gan import *
from .tokenizer import *
from .unet import *
from .esr import *


def init_model_by_params(config: dict):
    """
    :param config: {class_name:{a:1, b:2, c:3}}
    :return:
    """
    model_name = list(config.keys())[0]
    model_params = config[model_name]
    try:
        obj = globals()[model_name]
        return obj(**model_params)
    except KeyError:
        print(model_name, "未进行预设")
        return -1


def init_model_from_config_state_dict(config_path, state_dict_path, strict=False):
    with open(config_path, "r") as f:
        config = json.load(f)
    model_name = list(config.keys())[0]
    model_params = config[model_name]
    try:
        if model_name in ["Diffusion", "RectifiedFlow"]:
            noise_model_name = list(config["model"].keys())[0]
            noise_model_obj = globals()[noise_model_name](**config["model"][noise_model_name])
            model_obj = globals()[model_name](**model_params, model=noise_model_obj)
        else:
            model_obj = globals()[model_name](**model_params)
        model_obj.load_state_dict(torch.load(state_dict_path, map_location="cpu"), strict=strict)
        model_obj = model_obj
        return model_obj
    except KeyError:
        print(model_name, "未进行预设")
        return -1
