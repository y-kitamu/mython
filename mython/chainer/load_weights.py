import chainer


def load_weights(model, weights, name=""):
    """完全一致しない重みファイルから重みを読み込むための関数
    Args:
        model (chainer.Chain) : target model to load weights
        weights (NpzFile) : numpy npz file
        name (string) :
    """
    for child in model.children():
        if isinstance(child, chainer.Chain):
            load_weights(child, weights, f"{name}/{child.name}")
        elif isinstance(child, weights, chainer.ChainList):
            for c in child:
                load_weights(c, f"{name}/{child.name}/{c.name}")
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
                print(f"Load weight {param_name}")
