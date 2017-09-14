import domain_adaptation
from include.infer_from_trainer import Load_trainer
import pandas as pd

###
source_data_df = pd.read_csv('/Users/karthikkappaganthu/Documents/online_learning/demo_testing_data/2_domain_adapt_output_files/source_predictions_vis_.csv',sep=',')
target_data_df = pd.read_csv('/Users/karthikkappaganthu/Documents/online_learning/demo_testing_data/2_domain_adapt_output_files/target_predictions_vis_.csv',sep=',')

source_data_df = pd.read_csv('/home/ubuntu/online_learning_demo/demo_data/domain_adapt_output_files/source_predictions_vis_.csv',sep=',')
target_data_df = pd.read_csv('/home/ubuntu/online_learning_demo/demo_data/domain_adapt_output_files/target_predictions_vis_.csv',sep=',')


source_files=source_data_df['image_name'].tolist()
source_labels=source_data_df['true_label'].tolist()

target_files=target_data_df['image_name'].tolist()
target_labels=target_data_df['voting_pred'].tolist()
print(source_labels)
# print(target_labels)


# ###
init_trainer_path='/home/ubuntu/online_learning_demo/_base_model'
init_trainer_path=Load_trainer(init_trainer_path)
base_model_files={'arch_proto':'../../model_files/base_models/sns_model2/sns_train_val_sq_smaller_batch_size.prototxt',\
                      'solver_proto':'../../model_files/base_models/sns_model2/sns_solver_sq_smaller_batch_size.prototxt',\
                      'deploy_proto':'../../model_files/base_models/sns_model2/sns_deploy_sq.prototxt',\
                      'initial_weights':''}

params = {'mix_pct': [(99, 0)], 'crop_size': 227,
              'perf_layers': ['loss'],
              'prediction_key': 'prob'}

name_prefix='demo_domain_adapt/'


serving_path = '/home/ubuntu/online_learning/'
##
da=domain_adaptation.Domain_adaptation(source_files,source_labels,target_files,target_labels, \
                                   base_model_files,params,name_prefix,serving_path)

da.setup_training_models()
da.split_test_train()
da.generate_training_val_data()
da.generate_proto_files()
da.train_transfer_learning()
