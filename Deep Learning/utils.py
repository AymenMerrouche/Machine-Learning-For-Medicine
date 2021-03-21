import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable 


def plot_param_sapace_2D(cv_results ,param_names, fig, axes, with_values = None):
    """
    Plots a heat map out of cross validation results (mean and std for two parameters)
    :param cv_results: cross validation result as returned by cv.cv_results_ in sklearn.
    :param param_names: list of parameters to plot (of length 2, the first one will be on the y axis. Expects format param_+param_name)
    :param fig: figure object plt.
    :param axes: tuple of two axis plt. The first will be used for the mean, the second for std.
    :param with_values: filters to apply to other parameters (dict {"param_"+param_name : values}), if None then mean over possible values.
    """
    # get cv_results as df
    cv_res_df = pd.DataFrame(cv_results)
    # filter cv results with values given in with_values input (dict)
    if with_values is not None :
        cv_res_df = cv_res_df.loc[(cv_res_df[list(with_values)] == pd.Series(with_values)).all(axis=1)]
    # group by params of interest
    cv_res_df = cv_res_df.groupby(param_names).mean()
    # convert indexes into columns (twice since we have 2 param_names)
    cv_res_df.reset_index(level=0, inplace=True)
    cv_res_df.reset_index(level=0, inplace=True)
    # select metrics of interest
    cv_res_df_std = cv_res_df[["std_test_score"]+param_names]
    cv_res_df_mean = cv_res_df[["mean_test_score"]+param_names]
    # create images that will be shown in heat map
    cv_res_df_mean = cv_res_df_mean.pivot(index=param_names[0], columns=param_names[1], values="mean_test_score")
    cv_res_df_std = cv_res_df_std.pivot(index=param_names[0], columns=param_names[1], values="std_test_score")
    # show the images
    images = {"mean" : cv_res_df_mean, "std" : cv_res_df_std}
    for i, (ax, metric) in enumerate(zip(axes, images)):
        # show the image
        im = ax.imshow(images[metric].to_numpy())
        # title, layout and annotations
        ax.set_title(metric+"\n"+str(with_values), fontsize=10, fontweight='bold')
        # We want to show all ticks
        ax.set_xticks(np.arange(len(images[metric].columns)))
        ax.set_yticks(np.arange(len(images[metric].index)))
        # and label them with the respective list entries
        ax.set_xticklabels(images[metric].columns)
        ax.set_yticklabels(images[metric].index)
        # labels
        ax.set_xlabel(param_names[1])
        ax.set_ylabel(param_names[0])
        # Rotate the tick labels and set their alignment.
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
                 rotation_mode="anchor")
        # Loop over data dimensions and create text annotations.
        for i in range(len(images[metric].index)):
            for j in range(len(images[metric].columns)):
                text = ax.text(j, i, str(round(images[metric].to_numpy()[i, j], 4)),
                               ha="center", va="center", color="w")
        # set color bar layout
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(im, cax=cax)