import domain_adaptation
from include.infer_from_trainer import Load_trainer
import pandas as pd
from flask import Flask,request
import os

app = Flask(__name__)
@app.route('/embed_space_learn',methods=['POST','GET'])
def retain_data():
    if request.method == 'POST':
        data_dir = request.form['data_dir']
        compute_tsne_flag = request.form['compute_tsne_flag']
    else:
        data_dir = ''
        compute_tsne_flag = ''

    if data_dir == '':
        data_dir = '../domain_adaptation_demo_data'
        print('Using data in {0}'.format(data_dir))

    data_file=data_dir+'/relabled_out/visualization_data_relabled.csv'

    data_df=pd.read_csv(data_file)

    source_df=data_df[data_df['in_source_data_flag']==True]
    target_df = data_df[data_df['in_source_data_flag'] == False]

    source_files=source_df['image_name'].tolist()
    source_labels = source_df['true_label'].tolist()

    target_files = target_df['image_name'].tolist()
    target_labels=target_df['updated_label'].tolist()

    base_model_files = {
    'arch_proto': '../domain_adaptation_demo_data/model_files/base_models/sns_model2/sns_train_val_sq_smaller_batch_size.prototxt', \
    'solver_proto': '../domain_adaptation_demo_data/model_files/base_models/sns_model2/sns_solver_sq_smaller_batch_size.prototxt', \
    'deploy_proto': '../domain_adaptation_demo_data/model_files/base_models/sns_model2/sns_deploy_sq.prototxt', \
    'initial_weights': ''}


    params = {'mix_pct': [(99, 99)], 'crop_size': 227,
              'perf_layers': ['loss'],
              'prediction_key': 'prob'}

    name_prefix = 'demo_domain_adapt/'
    serving_path = '/home/ubuntu/online_learning/'

    da = domain_adaptation.Domain_adaptation(source_files, source_labels, target_files, target_labels, \
                                             base_model_files, params, name_prefix, serving_path)

    da.setup_training_models()
    da.split_test_train()
    da.generate_training_val_data()
    da.generate_proto_files()
    da.train_transfer_learning()
    return "OK"

if __name__=="__main__":
    retain_data()