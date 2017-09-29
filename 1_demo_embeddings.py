from include.infer_from_trainer import Load_trainer



### Initial Inputs from user
trainer_path='/home/ubuntu/online_learning_demo/_base_model'
target_data_file='/home/ubuntu/online_learning_demo/demo_data/target_data_files.txt'
params = {'crop_size': 227,'perf_layers': ['loss'],'prediction_key': 'prob'}
trainer=Load_trainer(trainer_path)


### Prompt user
print('Available weights for inference')
print('------------------------------')
print(trainer.available_weights_files)
print(trainer.final_weights_files)
print('------------------------------')
print('Using {0} for feature extraction'.format(trainer.deploy_meanfile))
print("To change please set the 'trainer.deploy_weights_file' to one of the avialable weights")
print('------------------------------')


print(target_data_file)

## Feature Extraction
target_all_pred, target_all_cnf_matrix_array = trainer.get_performance(target_data_file, params['crop_size'],
                                                                             params['prediction_key'], labels_col=[],
                                                                             layer='conv10',save_data_suffix='')

print('Features saved to {0}'.format(trainer.dir_strc.analysis_results_files))
