from sklearn.model_selection import train_test_split
from include.mix_source_target_data import  mix_source_target
import include.ml_train as ml_trainer
from include.base_proto_to_exp_proto import modify_solver_net_path, modify_arch_proto_data_paths
from include.inference import Inference
from include.cnf_matrix_pct import compute_cnf_matrix

import numpy as np
import pandas as pd
import os
import csv
import cv2
class Domain_adaptation():
    def __init__(self,source_files,source_labels,target_files,base_model_files,params,name_prefix,serving_path,target_labels=[]):
        self.source_files=source_files

        self.target_files=target_files
        self.source_labels = source_labels
        if not target_labels:
            target_labels=[-100]*len(target_files)
        self.target_labels = target_labels

        self.name_prefix=name_prefix
        self.base_model_files=base_model_files

        self.params=params
        if 'source_test_train_split_frac' not in params.keys():
            self.source_test_train_split_frac=.2

        if 'target_test_train_split_frac' not in params.keys():
            self.target_test_train_split_frac=.2

        self.serving_path=serving_path

        if 'mix_pct' not in self.params.keys():
            self.mix_pct=[(0,99),(10,99)]
        else:
            self.mix_pct=self.params['mix_pct']

    def split_test_train(self):
        # Generate train, val, test data
        self.source_train_files, self.source_test_files, self.source_train_labels,self.source_test_labels= train_test_split(self.source_files, self.source_labels,test_size=self.source_test_train_split_frac)
        self.target_train_files, self.target_test_files, self.target_train_labels, self.target_test_labels = train_test_split(
            self.target_files, self.target_labels, test_size=self.target_test_train_split_frac)

        self.mix_train_dfs, self.mix_val_dfs=mix_source_target(self.mix_pct, self.target_train_files, self.target_train_labels, self.source_train_files, self.source_train_labels)

    def setup_training_models(self):
        if not self.name_prefix:
            self.name_prefix='domain_adapt'
        ml_trainers=[]
        for pct in self.mix_pct:
            train_run_name = self.name_prefix+'_mix_trgt_'+str(pct[0])+'_source_'+str(pct[1])
            ml_trainers.append(ml_trainer.ML_train(train_run_name,serving_dir=self.serving_path))
        self.ml_trainers=ml_trainers


    def generate_training_val_data(self):
        for idx,trainer in enumerate(self.ml_trainers):
            print('Generating training and validation ')
            self.mix_train_dfs[idx][['filename','labels']].to_csv(trainer.dir_strc.dbs+'/train_txt_data',index=None,header=None,sep=' ')
            self.mix_val_dfs[idx][['filename','labels']].to_csv(trainer.dir_strc.dbs + '/val_txt_data',index=None, header=None,sep=' ')
            # self.mix_train_dfs[idx].to_csv(trainer.dir_strc.dbs+'/train_txt_data',index=None)
            # self.mix_val_dfs[idx].to_csv(trainer.dir_strc.dbs + '/val_txt_data', index=None)
            trainer.create_img_lmdb_os(trainer.dir_strc.dbs+'/train_txt_data',out_base_name='train_lmdb',resize_height=self.params['crop_size'],resize_width=self.params['crop_size'])
            trainer.create_img_lmdb_os(trainer.dir_strc.dbs + '/val_txt_data', out_base_name='val_lmdb',resize_height=self.params['crop_size'],resize_width=self.params['crop_size'])
            trainer.compute_mean(trainer.dir_strc.dbs+'/train_lmdb')

            with open(trainer.dir_strc.dbs+'/source_test_txt_data','wb') as f:
                writer=csv.writer(f,delimiter=' ', quotechar='#', quoting=csv.QUOTE_MINIMAL)
                for idx,file in enumerate(self.source_test_files):
                    writer.writerow([file,self.source_test_labels[idx]])

            with open(trainer.dir_strc.dbs+'/target_test_txt_data','wb') as f:
                writer=csv.writer(f,delimiter=' ', quotechar='#', quoting=csv.QUOTE_MINIMAL)
                for idx,file in enumerate(self.target_test_files):
                    writer.writerow([file,self.target_test_labels[idx]])

    def generate_proto_files(self):
        base_arch_path=self.base_model_files['arch_proto']
        base_solver_path=self.base_model_files['solver_proto']
        for idx, trainer in enumerate(self.ml_trainers):
            arch_train_proto_path = trainer.dir_strc.model_protofiles+'/train.prototxt'
            arch_deploy_proto_path = trainer.dir_strc.model_protofiles + '/deploy.prototxt'
            solver_proto_path = trainer.dir_strc.model_protofiles + '/solver.prototxt'
            mean_file_path = trainer.dir_strc.dbs+'/meanfile.binaryproto'
            train_lmbd_path = trainer.dir_strc.dbs + '/train_lmdb'
            test_lmbd_path = trainer.dir_strc.dbs + '/val_lmdb'

            source_paths=[train_lmbd_path,test_lmbd_path]

            modify_arch_proto_data_paths(base_arch_path, mean_file_path, source_paths, arch_train_proto_path)
            modify_solver_net_path(base_solver_path,arch_train_proto_path,solver_proto_path)
            if 'deploy_proto' in self.base_model_files.keys():
                os.system('cp '+ self.base_model_files['deploy_proto']+' '+ arch_deploy_proto_path)


    def train_transfer_learning(self,solver_method='pycaffe'):
        if 'initial_weights' in self.base_model_files.keys():
            initial_weights=self.base_model_files['initial_weights']
        else:
            initial_weights=''
        if 'perf_layers' in self.params.keys():
            perf_layers=self.params['perf_layers']
        else:
            perf_layers=''
        for idx, trainer in enumerate(self.ml_trainers):
            os.system('clear')
            print('Training {0} of {1}').format(idx,len(self.ml_trainers))
            solver_proto=trainer.dir_strc.model_protofiles + '/solver.prototxt'
            if idx==0:
                weights_file=initial_weights
            else:
                weights_file=final_weights
            if solver_method=='pycaffe':
                slvr_perf_vec, final_weights, weights_save_dir=trainer.train_solver(solver_proto,niter=20000,weights_file=weights_file,perf_layers=perf_layers,display_interval=100,weights_save_interval=2000)
            else:
                # RUn with caffe....not yet tested
                if 'caffe_root' in self.params.keys():
                    caffe_root=params[caffe_root]
                else:
                    caffe_root=os.environ["CAFFE_ROOT"]
                solver_log_path=trainer.dir_strc.weight_files+'/caffe_solver_log.log'
                if weights_file:
                    solver_str=caffe_root+'/build/tools/caffe train -solver'+ solver_proto +' -weights '+weights_file+'-gpu 0 2>&1 | tee '+solver_log_path
                else:
                    solver_method=caffe_root+'/build/tools/caffe train -solver'+ solver_proto +'-gpu 0 2>&1 | tee '+solver_log_path

    def get_performance(self):
        for idx, trainer in enumerate(self.ml_trainers):
            available_weights = [[x, int(filter(str.isdigit,x))] for x in os.listdir(trainer.dir_strc.weight_files) if x.split('.')[-1]=='caffemodel']
            available_weights = np.array(available_weights)
            last_weight_idx = np.argmax(available_weights[:,1])
            last_weight_file = available_weights[last_weight_idx,0]
            deploy_weights=trainer.dir_strc.weight_files+'/'+last_weight_file

            deploy_inputs={'net': trainer.dir_strc.model_protofiles+'/deploy.prototxt', 'mean': trainer.dir_strc.dbs+'/meanfile.binaryproto', 'model': deploy_weights, 'mean_type': 'binaryproto'}
            infer_params = {'img_w': self.params['crop_size'], 'img_h': self.params['crop_size']}
            infer=Inference(deploy_inputs, params=infer_params)

            val_data=pd.read_csv(trainer.dir_strc.dbs+'/val_txt_data',header=None,sep=' ')
            pred_class=[]
            pred_prob=[]
            for idx2,file in enumerate(val_data[0]):
                print('Running Infrence for {0} ({1} or {2})'.format(file,idx2,len(val_data[1])))
                img=cv2.imread(file)
                output=infer.predict(img)
                pred_class.append(output[self.params['prediction_key']][0].argmax())
                pred_prob.append(output[self.params['prediction_key']][0].max())
            val_data[2]=pred_class
            val_data[3]=pred_prob
            val_data.to_csv(trainer.dir_strc.analysis_results_files+'/val_data_predictions',header=['image_name','true_label','dl_pred','dl_prob'],index=None,sep=',')

            val_cnf_matrix,val_cnf_matrix_pct=compute_cnf_matrix(val_data[1],val_data[2])
            val_cnf_matrix_array = np.vstack([val_cnf_matrix, val_cnf_matrix_pct])
            np.savetxt(trainer.dir_strc.analysis_results_files+'/val_cnf_matrix',val_cnf_matrix_array,'%f')


            train_data = pd.read_csv(trainer.dir_strc.dbs + '/train_txt_data', header=None, sep=' ')
            pred_class = []
            pred_prob = []
            for idx2, file in enumerate(train_data[0]):
                print('Running Infrence for {0} ({1} or {2})'.format(file, idx2, len(train_data[1])))
                img = cv2.imread(file)
                output = infer.predict(img)
                pred_class.append(output[self.params['prediction_key']][0].argmax())
                pred_prob.append(output[self.params['prediction_key']][0].max())
            train_data[2] = pred_class
            train_data[3] = pred_prob
            train_data.to_csv(trainer.dir_strc.analysis_results_files + '/train_data_predictions', header=['image_name','true_label','dl_pred','dl_prob'], index=None,
                            sep=',')

            train_cnf_matrix, train_cnf_matrix_pct = compute_cnf_matrix(train_data[1], train_data[2])
            train_cnf_matrix_array = np.vstack([train_cnf_matrix, train_cnf_matrix_pct])
            np.savetxt(trainer.dir_strc.analysis_results_files + '/train_cnf_matrix', train_cnf_matrix_array, '%f')




if __name__=='__main__':
    # AWS
    source_data_df = pd.read_csv('/home/ubuntu/online_learning/data/loc_data_txt_aws/source_data_files.txt',header=None,sep=' ')
    target_data_df = pd.read_csv('/home/ubuntu/online_learning/data/loc_data_txt_aws/target_data_files.txt',header=None,sep=' ')

    # source_data_df = pd.read_csv('/home/ubuntu/online_learning_demo/demo_data/domain_adapt_output_files/source_predictions_vis_.csv', header=None,sep=',')
    # target_data_df = pd.read_csv('/home/ubuntu/online_learning_demo/demo_data/domain_adapt_output_files/target_predictions_vis_.csv', header=None,sep=',')
    source_files = source_data_df[0].tolist()
    source_labels = source_data_df[1].tolist()
    target_files = target_data_df[0].tolist()
    target_labels = target_data_df[1].tolist()

    serving_path = '/home/ubuntu/online_learning/'
    # VISHNU
    # source_data_df = pd.read_csv('../../data/loc_holdout_data_2_class/source_data_files.txt', header=None, sep=' ')
    # target_data_df = pd.read_csv('../../data/loc_holdout_data_2_class/target_data_files.txt', header=None, sep=' ')
    # serving_path='/raid/karthik/online_learning/'

    name_prefix='demo_domain_adapt/'

    # Other things to include number of iterations for each run
    # params = {'mix_pct': [(0,99),(10, 99),(50,99),(50,50)], 'crop_size': 227,
    #           'perf_layers': ['loss'],
    #           'prediction_key':'prob'}
    params = {'mix_pct': [(99, 0)], 'crop_size': 227,
              'perf_layers': ['loss'],
              'prediction_key': 'prob'}

    # base_model_files={'arch_proto':'../../model_files/base_models/sns_model2/sns_train_val_sq_smaller_batch_size.prototxt',\
    #                   'solver_proto':'../../model_files/base_models/sns_model2/sns_solver_sq_smaller_batch_size.prototxt',\
    #                   'deploy_proto':'../../model_files/base_models/sns_model2/sns_deploy_sq.prototxt',\
    #                   'initial_weights':'../../model_files/base_models/sns_model2/train_iter_10000.caffemodel'}
    # base_model_files = {
    #     'arch_proto': '../../model_files/base_models/sns_model2/sns_train_val_sq_smaller_batch_size.prototxt', \
    #     'solver_proto': '../../model_files/base_models/sns_model2/sns_solver_sq_smaller_batch_size.prototxt', \
    #     'deploy_proto': '../../model_files/base_models/sns_model2/sns_deploy_sq.prototxt', \
    #     'initial_weights': '../../model_files/sqNet_domain_adapt_v3/_mix_trgt_10_source_99/weight_files/weights_base_7999.caffemodel'}

    base_model_files={'arch_proto':'/home/ubuntu/online_learning/model_files/base_models/sns_model2/sns_train_val_sq_smaller_batch_size.prototxt',\
                      'solver_proto':'/home/ubuntu/online_learning/model_files/base_models/sns_model2/sns_solver_sq_smaller_batch_size.prototxt',\
                      'deploy_proto':'/home/ubuntu/online_learning/model_files/base_models/sns_model2/sns_deploy_sq.prototxt',\
                      'initial_weights':''}



    # source_files, source_labels, target_files, target_labels, base_model_files, params, name_prefix
    domain_adapt=Domain_adaptation(source_files,source_labels,target_files, \
                                   base_model_files,params,name_prefix,serving_path,target_labels=target_labels)
    domain_adapt.setup_training_models()
    domain_adapt.split_test_train()
    domain_adapt.generate_training_val_data()
    domain_adapt.generate_proto_files()
    domain_adapt.train_transfer_learning()
    # domain_adapt.get_performance()



