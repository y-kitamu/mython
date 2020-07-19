import chainer


def load_weights(weights, model, name=""):
    """完全一致しない重みファイルから重みを読み込むための関数
    Args:
        weights (NpzFile) : numpy npz file
        model (chainer.Chain) : target model to load weights
        name (string) :
    """
    for child in model.children():
        if isinstance(child, chainer.Chain):
            load_weights(weights, child, f"{name}/{child.name}")
        elif isinstance(child, chainer.ChainList):
            for c in child:
                load_weights(weights, c, f"{name}/{child.name}/{c.name}")
        elif isinstance(child, chainer.Link):
            for n, p in child.namedparams():
                param_name = f"{name}/{child.name}{n}"[1:]
                if param_name not in weights:
                    print(f"{param_name} does not exist")
                    continue
                if p.data is None and param_name in weights:
                    print(f"initialize parameter : {param_name}")
                    p.initialize(weights[param_name].shape)
                if p.data.shape != weights[param_name].shape:
                    print(f"shape mismatch : {param_name}. Skip")
                    continue
                p.data[...] = weights[param_name]
                # print(f"Load weight {param_name}")
