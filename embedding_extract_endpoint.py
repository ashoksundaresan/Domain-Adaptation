from flask import Flask,request,jsonify
import os
from include.infer_from_trainer import Load_trainer
import pandas as pd

app = Flask(__name__)
@app.route('/embed_space_learn',methods=['POST','GET'])
def compute_embed_space():
    if request.method == 'POST':
        data_dir = request.form['data_dir']
        trainer_path=request.form['trainer_path']
        run_source=request.form['run_source']
        training_data_file = request.form['training_data_file']
    else:
        data_dir=''
        trainer_path =''
        run_source=''
        training_data_file=''

    # data_dir = ''
    # trainer_path = ''
    # run_source = ''
    # training_data_file = '/home/ubuntu/domain_adaptation_demo_data/target_data_dir/target_data.csv'
    # trainer_path=data_dir+'/model_files/demo_domain_adapt/_mix_trgt_99_source_99'

    if data_dir == '':
        data_dir = '../domain_adaptation_demo_data'
        print('Using data in {0}'.format(data_dir))


    if trainer_path=='':
        trainer_path = data_dir+'/initial_model'

    if run_source=='':
        run_source='True'

    if training_data_file=='':
        training_data_file = data_dir+'/target_data_dir/target_data.csv'

    params = {'crop_size': 227, 'perf_layers': ['loss'], 'prediction_key': 'prob'}
    trainer = Load_trainer(trainer_path)

    if run_source=='True':
        source_data_file='../domain_adaptation_demo_data/source_data_dir/source_data.csv'
        source_pred, source_cnf_matrix_array = trainer.get_performance(source_data_file, params['crop_size'],
                                                                           params['prediction_key'], labels_col=[],
                                                                           layer='conv10', save_data_suffix='')
        os.system('mkdir data_dir/embeddings_out')
        os.system('cp '+trainer.dir_strc.analysis_results_files+'/source_test_txt_data_predictions'+' '+data_dir+'/embeddings_out/updated/source_data.csv')
        os.system('cp ' + trainer.dir_strc.analysis_results_files + '/source_test_txt_data_embeds.npy'' '+data_dir+'/embeddings_out/updated/source_data_embeds.npy')

    target_pred, target_cnf_matrix_array = trainer.get_performance(training_data_file, params['crop_size'],
                                                                   params['prediction_key'], labels_col=[],
                                                                   layer='conv10', save_data_suffix='')

    os.system('cp ' + trainer.dir_strc.analysis_results_files + '/target_data.csv_predictions' +' '+data_dir+'/embeddings_out/updated/target_data.csv')
    os.system('cp ' + trainer.dir_strc.analysis_results_files + '/target_test_txt_data_embeds.npy' +' '+data_dir+'/embeddings_out/updated/target_data_embeds.npy')
    return_obj={'source_predictions':data_dir+'/embeddings_out/updated/source_data.csv',\
                'target_predictions':data_dir+'/embeddings_out/updated/target_data.csv',\
                'source_embeddings':data_dir+'/embeddings_out/updated/source_data_embeds.npy',\
                'target_embeddings':data_dir+'/embeddings_out/updated/target_data_embeds.npy'}

    return jsonify(return_obj)

if __name__=="__main__":
    compute_embed_space()

