import matplotlib.pyplot as plt
import numpy as np

# データの読み込み
datapath = './makedata/dataset_25_7.csv'
data = np.loadtxt(datapath, delimiter=',', skiprows=1)
datarow = np.loadtxt(datapath, delimiter=',', dtype=str)
header = datarow[0]

# 対象のショット番号リスト
shotlist = [
    114282, 115083, 115085, 152718, 152726, 174691, 174701, 174739,
    176727, 176729, 176731, 176734, 176744, 176745, 163379, 163381,
    163385, 163396, 163400, 163402, 163404, 189303, 189305, 189309,
    166562, 166564, 166590
]

def change_detection(target_number):
    target = df[:, target_number]
    d = 10  # スライド窓のサイズ
    vector_count = 5   # ベクトル数

    # 行列Xの作成
    X = np.zeros((d, 1))
    for i in range(target.shape[0] - d):
        x = target[i:i+d]
        X = np.append(X, np.reshape(x, (d, 1)), axis=1)

    # 行列X^t(X_t)，Z^t(Z_t)の作成
    X_t = np.zeros((d, vector_count, 1))
    Z_t = np.zeros((d, vector_count, 1))
    for i in range(1, target.shape[0] - (d + vector_count + 8)):
        X_t = np.append(X_t, np.reshape(X[:, i:i+vector_count], (d, vector_count, 1)), axis=2)
        Z_t = np.append(Z_t, np.reshape(X[:, i+d:i+d+vector_count], (d, vector_count, 1)), axis=2)

    # 特異スペクトル抽出
    X_t_U = np.zeros((d, vector_count, 1))
    Z_t_U = np.zeros((d, vector_count, 1))
    for i in range(1, X_t.shape[2]):
        if np.isnan(Z_t[:, :, i]).any() or np.isinf(Z_t[:, :, i]).any():
            print(f"Z_t[:,:,{i}] に NaN または Inf が含まれています: {Z_t[:,:,i]}")
            continue  # NaN または Inf を含む場合はスキップ

        try:
            U_Z_t = np.linalg.svd(Z_t[:, :, i], full_matrices=False)[0]
        except np.linalg.LinAlgError:
            print("SVDが収束しませんでした")
            continue  # SVDが収束しない場合はスキップ

        U_X_t = np.linalg.svd(X_t[:, :, i], full_matrices=False)[0]
        U_Z_t = np.linalg.svd(Z_t[:, :, i], full_matrices=False)[0]

        X_t_U = np.append(X_t_U, np.reshape(U_X_t, (d, vector_count, 1)), axis=2)
        Z_t_U = np.append(Z_t_U, np.reshape(U_Z_t, (d, vector_count, 1)), axis=2)

    # 時刻tごとの特異スペクトルの内積計算
    e_list = []
    for i in range(X_t_U.shape[2]):
        e = np.linalg.svd(np.dot(X_t_U[:, :, i].T, Z_t_U[:, :, i]))[1]
        e_list.append(e[0])

    # 変化検知の出力
    ab = 0
    for i in range(1, len(e_list)):
        ab = np.append(ab, (1 - e_list[i]**1))  # 2000かけない方が良さそう
    label = header[target_number]
    return times, ab, label, target

# 各ショット番号に対して変化検知を実行
for shotNO in shotlist:
    df = data[data[:, 0] == shotNO]
    times = df[:, 1]

    # 時間範囲のフィルタリング
    start_time = 3.5
    end_time = 9
    df = df[times > start_time]
    times = times[times > start_time]
    df = df[times < end_time]
    times = times[times < end_time]

    # プロットの設定
    indices = [2, 3, 7, 8, 9, 10]
    plt.rcParams["font.size"] = 10  # フォントサイズを一括で変更
    plt.rcParams['lines.linewidth'] = 2  # 線の太さを一括で変更
    fig = plt.figure(figsize=(6, 10))  # figureを準備
    axes = [fig.add_subplot(len(indices), 1, i + 1) for i in range(len(indices))]

    j = 0
    for i in indices:
        times, ab, label, target = change_detection(i)
        if i == 33:
            label = r'$\Delta\Phi_\mathrm{eff}$'
        if i == 3:
            axes[j].set_ylim(0, 0.000008)
        axes[j].plot(times[0:ab.shape[0] - 1], ab[1:], label=label)
        axes[j].set_ylabel(label)
        j += 1

    for i in range(len(axes)):
        ax = axes[i]
        ax.set_xlim(3.5, 9)
        if i != len(axes) - 1:
            ax.set_xticklabels('')

    axes0 = [axes[i].twinx() for i in range(len(indices))]
    for i in range(len(indices)):
        axes0[i].plot(times, df[:, indices[i]], color='orange')

    axes[-1].set_xlabel("Time[s]", fontsize=14)
    axes[0].set_title('#' + str(shotNO) + ' anomaly')
    plt.subplots_adjust(left=0.25, right=0.85, bottom=0.15, top=0.90, hspace=0.4)
    plt.savefig('./change_detection/change_detection_%s_%s.png' % (shotNO, indices))
    print(str(shotNO))