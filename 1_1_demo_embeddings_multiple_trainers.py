from include.infer_from_trainer import Load_trainer


# name_prefix='/home/ubuntu/online_learning/model_files/sqNet_domain_adapt_v3/'
# params = {'mix_pct': [(0,99),(10, 99),(50,99),(50,50)], 'crop_size': 227,'perf_layers': ['loss'],'prediction_key':'prob'}

name_prefix='/home/ubuntu/online_learning/model_files/sqNet_domain_adapt_tl_start_ov_train/'
params = {'mix_pct': [(0, 99)], 'crop_size': 227,'perf_layers': ['loss'],'prediction_key': 'prob'}

source_test_data_file='/home/ubuntu/online_learning/model_files/sqNet_domain_adapt_v3/_base/dbs/source_test_txt_data'
target_test_data_file='/home/ubuntu/online_learning/model_files/sqNet_domain_adapt_v3/_base/dbs/target_test_txt_data'
source_all_data_file='/home/ubuntu/online_learning/data/loc_data_txt_aws/source_data_files.txt'
target_all_data_file='/home/ubuntu/online_learning/data/loc_data_txt_aws/target_data_files.txt'

train_dirs=[]#[name_prefix+'_base']
for pct in params['mix_pct']:
    train_dirs.append(name_prefix+'_mix_trgt_'+str(pct[0])+'_source_'+str(pct[1]))


weights_files=['']*len(pct)
weights_files[0]='/home/ubuntu/online_learning/model_files/sqNet_domain_adapt_tl_start_ov_train/_mix_trgt_0_source_99/weight_files/weights_base_29999.caffemodel'
save_data_suffix='weight_29999'
for idx,train_dir in enumerate(train_dirs):
    print('Analysis on {}'.format(train_dir))
    trainer=Load_trainer(train_dir)
    print(trainer.final_weights_files)
    ###### source_test
    source_test_pred, source_test_cnf_matrix_array = trainer.get_performance(source_test_data_file, params['crop_size'], params['prediction_key'], labels_col=1, layer='conv10',weights_file=weights_files[idx],save_data_suffix=save_data_suffix)
    ###### target_test
    target_test_pred, target_test_cnf_matrix_array = trainer.get_performance(target_test_data_file, params['crop_size'],
                                                                   params['prediction_key'], labels_col=1,
                                                                   layer='conv10',save_data_suffix=save_data_suffix)
    #### source_all
    source_test_all, source_all_cnf_matrix_array = trainer.get_performance(source_all_data_file, params['crop_size'],
                                                                             params['prediction_key'], labels_col=1,
                                                                             layer='conv10',save_data_suffix=save_data_suffix)
    #### target_all
    target_all_pred, target_all_cnf_matrix_array = trainer.get_performance(target_all_data_file, params['crop_size'],
                                                                             params['prediction_key'], labels_col=1,
                                                                             layer='conv10',save_data_suffix=save_data_suffix)





