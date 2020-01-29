import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from matplotlib.ticker import MaxNLocator


def remove_zeros(dt, dp):

    ma_dt = np.copy(dt)
    ma_dp = np.copy(dp)

    ma_dt[ma_dt > 0] = 1
    ma_dp[ma_dp > 0] = 1

    total_zeros_test = ma_dt.tolist().count(0)
    total_zeros_pred = ma_dp.tolist().count(0)

    nz = np.divide(100 * total_zeros_pred, total_zeros_test)

    new_t = []
    new_p = []
    for i in range(len(dt)):
        if dt[i] != 0:
            new_t.append(dt[i])
        if dp[i] != 0:
            new_p.append(dp[i])

    return [new_t, new_p, nz, total_zeros_test, total_zeros_pred]


def plot_compare_histograms(ax, nrow, samples_per_leaf, ytest, pred_sk, pred_rf):
    mean_ytest = np.mean(ytest)
    # print("Mean: ", mean_ytest)
    same_part = " vs True, SPL: {0}"
    titles_list = [r"$\bf{POI}$", r"$\bf{NB}$", r"$\bf{ZIP}$", r"$\bf{ZINB}$", r"$\bf{RF}$"]

    props = dict(boxstyle='round', facecolor='white', alpha=0.7)

    for j in range(0, 5):
        if j == 4:
            dist_t = ytest
            dist_p = np.round(pred_rf, decimals=0)
        else:
            dist_t = ytest
            dist_p = pred_sk[:, j]

        # print(titles_list[j])
        dist_t_nz, dist_p_nz, pct, tz, pz = remove_zeros(dist_t, dist_p)

        bins = np.linspace(0, 19, 20)

        textstr = '\n'.join((
            r'$TZ=%d$' % (tz,),
            r'$PZ=%d$' % (pz,),
            r'$PCT=%d$' % (pct,) + "%"))

        ax[nrow, j].set_facecolor("#EEEEEE")
        ax[nrow, j].grid()
        ax[nrow, j].set_title(titles_list[j] + same_part.format(samples_per_leaf), size=20, loc="right")
        ax[nrow, j].axvline(x=mean_ytest, color="black", label="Mean true dist.", linewidth=2, linestyle="dotted")
        ax[nrow, j].set_ylim(0, 3000)
        ax[nrow, j].set_xlim(0, 15)
        ax[nrow, j].set_xlabel("Predicted", size=14, labelpad=10)
        ax[nrow, j].set_ylabel("Frequency", size=14, labelpad=10)
        ax[nrow, j].xaxis.set_tick_params(labelsize=12)
        ax[nrow, j].yaxis.set_tick_params(labelsize=12)
        ax[nrow, j].text(0.67, 0.93, textstr, transform=ax[nrow, j].transAxes, fontsize=16, verticalalignment='top', bbox=props)
        # ax[nrow, j].set_xticks(bins, range(1, 21))
        ax[nrow, j].xaxis.set_major_locator(MaxNLocator(integer=True))

        # Blue: 023B91, red: D9042B
        h = ax[nrow, j].hist([dist_t_nz, dist_p_nz], bins=bins-0.5, color=["#D9042B", "#023B91"], label=["True dist.", "Pred. dist."])
        # plt.text(5, 0, "Predicted values", size=16)
        # plt.text(3, 0, "Frequency", size=16)
        # ax.get_legend().remove()

        if j==4 and nrow==1:
            ax[nrow, j].legend(loc="upper right", ncol=3, bbox_to_anchor=(-1.5, -5.3), prop={'size': 24})

    return h



################
# Main program #
################


meta_cols = range(0, 3)
data_test_cols = range(3, 4)
data_pred_cols_count = range(4, 8)
data_pred_cols_rf = range(8, 9)

path_tmp = r"D:/UTwente/PycharmProjects/04_Risk_Model/data/skewed_leaves/to_keep_for_taylor_2nd/testing_SPL{0}_NESTI{1}.csv"

samples_per_leaf = [100, 200, 400, 600, 800]
nesti = [10]

fig, ax = plt.subplots(nrows=5, ncols=5, sharex=False, sharey=False, figsize=(30, 25))
plt.subplots_adjust(wspace=0.5, hspace=0.6)

nrow = 0
for spl in samples_per_leaf:
    for nt in nesti:
        path_curr = path_tmp.format(spl, nt)
        meta_m = np.loadtxt(path_curr, usecols=meta_cols, delimiter=";", dtype=np.float)
        ytest = np.loadtxt(path_curr, usecols=data_test_cols, delimiter=";", dtype=np.float)
        pred = np.loadtxt(path_curr, usecols=data_pred_cols_count, delimiter=";", dtype=np.float)
        pred_rf = np.loadtxt(path_curr, usecols=data_pred_cols_rf, delimiter=";", dtype=np.float)
        # mean_ytest = np.mean(ytest)

        plot_compare_histograms(ax, nrow, spl, ytest, pred, pred_rf)
        nrow += 1


path_fig_out = r"C:/Users/irene/Pictures/paper4_fin/0204_Compare_Histograms_100-800SPL.png"
manager = plt.get_current_fig_manager()
manager.window.showMaximized()
plt.pause(10)
plt.gcf().savefig(path_fig_out, format='png', dpi=300)

# plt.show()

