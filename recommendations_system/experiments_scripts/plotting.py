from sklearn.metrics import auc, roc_curve
import matplotlib.pyplot as plt


def plot_roc_auc(y_true, y_pred, savepath=None):
    if type(y_true) in (list, tuple) and type(y_pred) in (list, tuple):
        assert len(y_true) == len(y_pred)
        if len(y_true) > 5:
            raise ValueError('Up to 5 lines supported.')
        colors = ('b', 'g', 'c', 'm', 'y')
        for i, (split_y_true, split_y_pred) in enumerate(zip(y_true, y_pred)):
            fpr, tpr, threshold = roc_curve(split_y_true, split_y_pred)
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f'Split {i}, AUC = %0.4f' % roc_auc, color=colors[i])
    else:
        fpr, tpr, threshold = roc_curve(y_true, y_pred)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label='AUC = %0.4f' % roc_auc)

    plt.title('Receiver Operating Characteristic')
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')

    if savepath is not None:
        plt.savefig(savepath)
    else:
        plt.show()
