import include.four_group_plot as gplt


csv_file= '/Users/karthikkappaganthu/Documents/online_learning/demo_testing_data/3_label_output_files/visualization_data_.csv'
gplt.four_group_plot(csv_file,x_col='embeds_tsne_0',y_col='embeds_tsne_1',group1_col='in_source_data_flag',group2_col='true_label',group3_col='dl_pred',show_only_col='dl_low_prob_samples')