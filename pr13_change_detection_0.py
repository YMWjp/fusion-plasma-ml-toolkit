import matplotlib.pyplot as plt
import numpy as np

datapath = './makedata/dataset_25_7.csv'

data = np.loadtxt(
    datapath, delimiter=',',skiprows=1
    )

datarow = np.loadtxt(
    datapath, delimiter=',',dtype=str
    )
header = datarow[0]

shotlist = [115083,115085,114282,152718]

def change_detection(target_numebr):
    target = df[:,target_numebr]
    # target = target*100
    # print(target)
    # target = target/np.max(target)
    # print(target)
    d = 10  #スライド窓のサイズ
    l = 5   #ベクトル数

    #%%
    #行列Xの作成
    X = np.zeros((d, 1))

    for i in range(target.shape[0] - d):
        x = target[i:i+d]
        X = np.append(X, np.reshape(x, (d, 1)), axis = 1)

    # print(target)
    # print(X)
    #%%
    #行列X^t(X_t)，Z^t(Z_t)の作成
    X_t = np.zeros((d, l, 1))
    Z_t = np.zeros((d, l, 1))

    for i in range(1, target.shape[0]-(d+l+8)):
        X_t = np.append(X_t, np.reshape(X[:, i:i+l], (d, l, 1)), axis=2)
        Z_t = np.append(Z_t, np.reshape(X[:, i+d:i+d+l], (d, l, 1)), axis=2)    

    #%%
    #特異スペクトル抽出
    X_t_U = np.zeros((d, l, 1))
    Z_t_U = np.zeros((d, l, 1))

    for i in range(1, X_t.shape[2]):
        if np.isnan(Z_t[:,:,i]).any() or np.isinf(Z_t[:,:,i]).any():
            print(f"Z_t[:,:,{i}] に NaN または Inf が含まれています: {Z_t[:,:,i]}")
            continue  # NaN または Inf を含む場合はスキップ

        # SVDを試みる
        try:
            U_Z_t = np.linalg.svd(Z_t[:,:,i], full_matrices=False)[0]
        except np.linalg.LinAlgError:
            print("SVDが収束しませんでした")
            continue  # SVDが収束しない場合はスキップ
            
        U_X_t = np.linalg.svd(X_t[:,:,i], full_matrices=False)[0]
        U_Z_t = np.linalg.svd(Z_t[:,:,i], full_matrices=False)[0]

        X_t_U = np.append(X_t_U, np.reshape(U_X_t, (d, l, 1)), axis=2)
        Z_t_U = np.append(Z_t_U, np.reshape(U_Z_t, (d, l, 1)), axis=2)
    # %%
    #時刻tごとの特異スペクトルの内積計算
    e_list = []
    for i in range(X_t_U.shape[2]):  # X_t_Uの第三次元のサイズに合わせる
        e = np.linalg.svd(np.dot(X_t_U[:,:,i].T, Z_t_U[:,:,i]))[1]
        e_list.append(e[0])
    #%%
    #変化検知の出力
    ab = 0
    for i in range(1, len(e_list)):
        ab = np.append(ab, (1-e_list[i]**1)) #2000かけない方が良さそう
    label = header[target_numebr]
    return times, ab, label, target

for shotNO in shotlist:
    df = data[data[:,0]==shotNO]
    times = df[:, 1]

    start_time = 3.5
    end_time = 9
    df = df[times>start_time]
    times = times[times>start_time]
    df = df[times<end_time]
    times = times[times<end_time]

    l = [2,10,11]
    plt.rcParams["font.size"] = 10 #フォントサイズを一括で変更
    plt.rcParams['lines.linewidth'] = 2 #線の太さを一括で変更
    fig = plt.figure(figsize=(6,10))#figureを準備
    axes = [fig.add_subplot(len(l),1,i+1) for i in range(len(l))]

    j = 0
    for i in l:
        times, ab, label, target= change_detection(i)
        if i == 33:
            label = r'$\Delta\Phi_\mathrm{eff}$'
        if i == 3:
            axes[j].set_ylim(0,0.000008)
        axes[j].plot(times[0:ab.shape[0]-1],ab[1:]  , label = label)
        axes[j].set_ylabel(label)
        j = j + 1
    for i in range(len(axes)):
        ax = axes[i]
        ax.set_xlim(3.5,9)
        if i != len(axes)-1:
            ax.set_xticklabels('')
    axes0 = [axes[i].twinx() for i in range(len(l))]
    for i in range(len(l)):
        axes0[i].plot(times,df[:,l[i]],color='orange')
    axes[-1].set_xlabel("Time[s]",fontsize=14)
    axes[0].set_title('#'+str(shotNO)+' anomaly')
    plt.subplots_adjust(
        left=0.25,right=0.85,bottom=0.15,top=0.90,hspace=0.4
        )
    plt.savefig('./change_detection/change_detection_%s_%s.png' % (shotNO, l))
    print(str(shotNO))


