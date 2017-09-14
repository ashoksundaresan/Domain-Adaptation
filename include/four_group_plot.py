import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import numpy as np
import mpld3
from mpld3 import plugins

def four_group_plot(csv_file,x_col,y_col,group1_col,group2_col,group3_col,show_only_col,show_only_col_flag=True):
    data=pd.read_csv(csv_file)

    data_group1 = data.groupby(group1_col)
    group2_labels=data[group2_col]

    fig, ax = plt.subplots()
    ax.margins(0.05)

    group1_marker_type=['o','s']
    group1_marker_size=[80,100]
    group2_colors=['purple','orange']
    group3_colors=['purple','orange']
    group3_marker_size = [20,30]

    # PLOT GROUP 1 and Group 2
    # TO be replaced by backgorund mesh
    # g1_cnt = 0
    # for name1,group1 in data_group1:
    #     marker=group1_marker_type[g1_cnt]
    #     marker_size = group1_marker_size[g1_cnt]
    #     data_group2 = group1.groupby(group2_col)
    #     g2_cnt=0
    #     for name2,group2 in data_group2:
    #         ax.scatter(group2[x_col],group2[y_col],marker=marker,s= marker_size,color=group2_colors[g2_cnt],alpha=.4)
    #         g2_cnt+=1
    #     g1_cnt+=1

    # Plot with just group 1
    # Plot with just group 1 and group 3
    g1_cnt = 0
    for name1, group1 in data_group1:
        marker = group1_marker_type[g1_cnt]
        marker_size = group3_marker_size[g1_cnt]
        data_group3 = group1.groupby(group3_col)
        g3_cnt = 0
        for name3, group3 in data_group3:
            lbl_str = group3_col + ': ' + str(name3)

            ax.scatter(group3[x_col], group3[y_col], marker=marker, s=marker_size, color=group3_colors[g3_cnt],
                       alpha=.6,label=lbl_str)
            g3_cnt += 1
        g1_cnt += 1

    g1_cnt=0
    for name1,group1 in data_group1:
        marker=group1_marker_type[g1_cnt]
        marker_size=group1_marker_size[g1_cnt]
        ax.scatter(group1[x_col], group1[y_col], marker=marker, s= marker_size,facecolors='cyan',edgecolors='k',alpha=.1,label=group1_col+'_'+str(name1))
        g1_cnt+=1



    ### show_only_col
    show_only_data=data[data[show_only_col]==show_only_col_flag]
    ax.scatter(show_only_data[x_col],show_only_data[y_col],marker='*',color='red',label='Please Label Manually ('+show_only_col+')')
    #
    # ax.legend()
    # plt.show()

    handles, labels = ax.get_legend_handles_labels()  # return lines and labels
    interactive_legend = plugins.InteractiveLegendPlugin(zip(handles,
                                                             ax.collections),
                                                         labels)
                                                         # alpha_unsel=0.5,
                                                         # alpha_over=1.5,
                                                         # start_visible=True)

    plugins.connect(fig, interactive_legend)
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_title('LOW DIM VISUALIZATION', size=20)
    mpld3.show()
    # return mpld3.fig_to_html(fig)
if __name__=='__main__':
    csv_file= '/Users/karthikkappaganthu/Documents/online_learning/demo_testing_data/3_label_output_files/visualization_data_.csv'
    four_group_plot(csv_file,x_col='embeds_tsne_0',y_col='embeds_tsne_1',group1_col='in_source_data_flag',group2_col='true_label',group3_col='dl_pred',show_only_col='dl_low_prob_samples')






