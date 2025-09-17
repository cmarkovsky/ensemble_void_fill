import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-f', '--filename', type=str, default='data/ts1.csv')
# parser.add_argument('-r', '--region', type=str, default = 'ehim')
args = parser.parse_args()
filename = args.filename
# reg = args.region

# import matplotlib.ticker as ticker
# # Define the formatting function
# def format_fn(value):
#     if int(value) == value:
#         return f'{int(value)}'
#     else:
#         return f'{float(value)}'
    
sns.set_style('whitegrid')
def plot_diff(whim, ehim, methods = ['constant_mean','hypsometric_mean', 'xgb'], title='Observed vs Predicted Elevation Change'):
    """
    Plot a scatter plot comparing true versus predicted elevation change.

    Parameters:
    - y_true: Array-like of true elevation change values
    - y_pred: Array-like of predicted elevation change values
    - title: Title of the plot (default: 'True vs Predicted Elevation Change')
    """
   
    y_true = 'target'
    method_names = {
        'constant_mean': 'Constant',
        'hypsometric_mean': 'Hypsometric',
        'xgb': 'XGBoost',
    }
    fig, axes = plt.subplots(2, len(methods), figsize=(15, 12), sharex = True, sharey=True)
    # c_train_rmse, c_train_mae, c_test_rmse, c_test_mae = calc_error(df, 'constant')
    # h_train_rmse, h_train_mae, h_test_rmse, h_test_mae = calc_error(df, 'hypsometric')
    
    for i, method in enumerate(methods):
        # print(i, method)
        if method == 'constant_mean':
            show_legend = True
        else:
            show_legend = False
        y_pred = method        
        # # Create the scatter plot
        kde1 = sns.kdeplot(data=whim, x=y_true, hue = 'void_mask', y=y_pred, fill=True, ax=axes[0,i], bw_adjust = 0.5, label = 'data', legend = show_legend)
        axes[0,i].plot([-4, 4], [-4, 4], color='#636363', linestyle='--', linewidth = 1)

        kde2 = sns.kdeplot(data=ehim, x=y_true, hue = 'void_mask', y=y_pred, fill=True, ax=axes[1,i], bw_adjust = 0.5, label = 'data', legend = False)
        axes[1,i].plot([-4, 4], [-4, 4], color='#636363', linestyle='--', linewidth = 1)

        if show_legend:
            leg = kde1.axes.get_legend()
            new_title = ''
            leg.set_title(new_title)
            new_labels = ['Non-void', 'Void']
            for t, l in zip(leg.texts, new_labels):
                t.set_text(l)
        # # Add labels and title
        axes[0,i].set_xlabel('')
        axes[1,i].set_xlabel('Observed Elevation Change (m a$^{-1}$)')

        axes[0,i].set_ylabel('')
        axes[1,i].set_ylabel('')
        # ax.set_ylabel('Predicted Elevation Change (m a$^{-1}$)')
        # ax.set_title(f'{method_names[method]}')
        axes[0,i].set_xlim([-3.5,3.5])
        axes[0,i].set_ylim([-3.5,3.5])
        axes[1,i].set_xlim([-3.5,3.5])
        axes[1,i].set_ylim([-3.5,3.5])
        # axes[0,i].grid(True)
    axes[0,0].set_ylabel('Predicted Elevation Change (m a$^{-1}$)')
    axes[1,0].set_ylabel('Predicted Elevation Change (m a$^{-1}$)')
    axes[0,0].set_title('Constant')
    axes[0,1].set_title('Hypsometric')
    axes[0,2].set_title('XGBoost')
    # axes[1,1].set_title('Eastern Himalaya')

    # plt.suptitle(title)
    plt.tight_layout()
    plt.show()
    plt.savefig('kde_plot.png', bbox_inches='tight')

results = pd.read_feather(filename)

whim = results[results['reg'] == 'whim']
# whim = whim.sample(1000)

ehim = results[results['reg'] == 'ehim']
# ehim = whim.sample(1000)
print(len(whim), len(ehim))

plot_diff(whim, ehim)