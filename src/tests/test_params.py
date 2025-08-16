# test_params.py
from src.domain.params import param


@param("time", needs=["fircall"], deps=[])
def get_time(ctx, deps):
    # 単に固定値を返す
    return [0, 1, 2, 3, 4]

@param("nel", needs=["fircall"], deps=["time"])
def get_nel(ctx, deps):
    time = deps["time"]
    # time に比例したダミー電子密度
    return [t * 0.1 for t in time]

@param("nelgrad", needs=["fircall"], deps=["time", "nel"])
def get_nelgrad(ctx, deps):
    time = deps["time"]
    nel = deps["nel"]
    # 差分近似で勾配を計算
    grad = []
    for i in range(1, len(time)):
        dt = time[i] - time[i-1]
        dn = nel[i] - nel[i-1]
        grad.append(dn/dt if dt != 0 else 0)
    return grad