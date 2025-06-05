from face.configs.base import cfg


def param_groups(model):
    params = {
        "layer4_conv_weights": {"params": [], "lr": cfg.lr_layer4, "wd": cfg.wd},
        "layer4_bn_weights": {"params": [], "lr": cfg.lr_layer4, "wd": 0},
        "layer4_bn_biases": {"params": [], "lr": cfg.lr_layer4 * 0.1, "wd": 0},
        "layer4_prelu": {"params": [], "lr": cfg.lr_layer4, "wd": 0},
        "layer4_down_conv": {"params": [], "lr": cfg.lr_layer4, "wd": cfg.wd * 0.5},
        "layer4_down_bn_weights": {"params": [], "lr": cfg.lr_layer4, "wd": 0},
        "layer4_down_bn_biases": {"params": [], "lr": cfg.lr_layer4 * 0.1, "wd": 0},
        "dropout_linear1_weights": {"params": [], "lr": cfg.lr, "wd": cfg.wd * 0.8},
        "dropout_linear1_biases": {"params": [], "lr": cfg.lr * 0.1, "wd": 0},
        "dropout_prelu1": {"params": [], "lr": cfg.lr, "wd": 0},
        "dropout_bn1_weights": {"params": [], "lr": cfg.lr, "wd": 0},
        "dropout_bn1_biases": {"params": [], "lr": cfg.lr * 0.1, "wd": 0},
        "dropout_linear2_weights": {"params": [], "lr": cfg.lr, "wd": cfg.wd * 0.8},
        "dropout_linear2_biases": {"params": [], "lr": cfg.lr * 0.1, "wd": 0},
        "dropout_prelu2": {"params": [], "lr": cfg.lr, "wd": 0},
        "dropout_ln_weights": {"params": [], "lr": cfg.lr, "wd": cfg.wd * 0.3},
        "dropout_ln_biases": {"params": [], "lr": cfg.lr * 0.1, "wd": 0},
        "bn_feats_weights": {"params": [], "lr": cfg.lr, "wd": 0},
        "bn_feats_biases": {"params": [], "lr": cfg.lr * 0.1, "wd": 0},
        "arcface_weights": {"params": [], "lr": cfg.lr_arcloss, "wd": cfg.wd * 0.1},
        "arcface_centers": {"params": [], "lr": cfg.lr_arcloss * 0.1, "wd": cfg.wd * 0.1},
    }

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        if "layer4." in name and "downsample" not in name:
            if "conv" in name and "weight" in name:
                params["layer4_conv_weights"]["params"].append(param)
            elif "bn" in name:
                if "weight" in name:
                    params["layer4_bn_weights"]["params"].append(param)
                elif "bias" in name:
                    params["layer4_bn_biases"]["params"].append(param)
            elif "prelu" in name:
                params["layer4_prelu"]["params"].append(param)

        elif "layer4." in name and "downsample" in name:
            if "0.weight" in name:
                params["layer4_down_conv"]["params"].append(param)
            elif "1.weight" in name:
                params["layer4_down_bn_weights"]["params"].append(param)
            elif "1.bias" in name:
                params["layer4_down_bn_biases"]["params"].append(param)

        elif "dropout" in name:
            if name == "dropout.1.weight":
                params["dropout_linear1_weights"]["params"].append(param)
            elif name == "dropout.1.bias":
                params["dropout_linear1_biases"]["params"].append(param)
            elif name == "dropout.2.weight":
                params["dropout_prelu1"]["params"].append(param)
            elif name == "dropout.3.weight":
                params["dropout_bn1_weights"]["params"].append(param)
            elif name == "dropout.3.bias":
                params["dropout_bn1_biases"]["params"].append(param)
            elif name == "dropout.4.weight":
                params["dropout_linear2_weights"]["params"].append(param)
            elif name == "dropout.4.bias":
                params["dropout_linear2_biases"]["params"].append(param)
            elif name == "dropout.5.weight":
                params["dropout_prelu2"]["params"].append(param)
            elif name == "dropout.6.weight":
                params["dropout_ln_weights"]["params"].append(param)
            elif name == "dropout.6.bias":
                params["dropout_ln_biases"]["params"].append(param)

        elif "bn_feats" in name:
            if "weight" in name:
                params["bn_feats_weights"]["params"].append(param)
            elif "bias" in name:
                params["bn_feats_biases"]["params"].append(param)

        elif "arc_loss.weight" in name:
            params["arcface_weights"]["params"].append(param)
        elif "arc_loss.centers" in name:
            params["arcface_centers"]["params"].append(param)
            
    trainable_params = sum(len(g["params"]) for g in params.values())
    expected_params = sum(1 for p in model.parameters() if p.requires_grad)

    if trainable_params != expected_params:
        missing = expected_params - trainable_params
        raise ValueError(f"Missing {missing} trainable parameters in optimizer groups")

    return [group for group in params.values() if group["params"]]
