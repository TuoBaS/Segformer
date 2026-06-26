import copy
import os

import yaml


def deep_merge(base, override):
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        conf = yaml.safe_load(f)

    if "_base_" in conf:
        base_path = conf.pop("_base_")
        base_dir = os.path.dirname(os.path.abspath(config_path))
        base_conf = load_config(os.path.join(base_dir, base_path))
        conf = deep_merge(base_conf, conf)

    return conf
