import os
from include.inference import Inference
import numpy as np
import pandas as pd
import cv2
from include.cnf_matrix_pct import compute_cnf_matrix

class Load_trainer:

    def __init__(self,dir_path):
        model_id=dir_path.split('/')[-1]
        self.model_id = model_id
        self.dir_strc = type('', (), {})()
        self.create_dir_structure(dir_path)
        self.final_weights_files=''
        self.available_weights_files = []

        self.check_for_deploy_components()
        # self.logger = logging.getLogger(model_id)
        # log_hdlr = logging.FileHandler(self.dir_strc.main + '/training_log.log')
        # log_formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
        # log_hdlr.setFormatter(log_formatter)
        # self.logger.addHandler(log_hdlr)
        # self.logger.setLevel(logging.DEBUG)


    def create_dir_structure(self,serving_dir):
        if serving_dir:
            cur_dir=serving_dir
        else:
            cur_dir='../..'
        fldrs=self.model_id.split('/')
        dir_str=cur_dir + '/model_files/'
        for fldr in fldrs:
            dir_str=dir_str+'/'+fldr+'/'
        self.dir_strc.main=cur_dir
        self.dir_strc.dbs= cur_dir+'/dbs'
        self.dir_strc.weight_files=cur_dir+'/weight_files'
        self.dir_strc.model_protofiles=cur_dir+'/model_protofiles'
        self.dir_strc.analysis_results_files=cur_dir+'/analysis_results_files'

    def check_for_deploy_components(self):
        # Check if protofiles are there
        components_detected={}
        for property,value in vars(self.dir_strc).iteritems():
            if os.path.isdir(value):
                components_detected[property]=True
            else:
                components_detected[property] = False

            if property=='model_protofiles':
                if os.path.isfile(value+'/deploy.prototxt'):
                    self.deploy_proto=value+'/deploy.prototxt'
                else:
                    self.deploy_proto=None

            if property=='weight_files':
                if os.path.isdir(value):
                    available_weights = [[x,int(filter(str.isdigit,x))] for x in os.listdir(value) if x.split('.')[-1]=='caffemodel']
                    if available_weights:
                        available_weights_sorted = sorted(available_weights, key=lambda x: x[1])
                        available_weights_sorted = np.array(available_weights_sorted)
                        self.available_weights_files =[value+'/'+x for x in available_weights_sorted[:,0]]

                        self.final_weights_files = value+'/'+available_weights_sorted[-1, 0]
                    else:
                        self.available_weights_files = None
                        self.final_weights_files = None
                else:
                    self.available_weights_files = None
                    self.final_weights_files = None
            print(self.final_weights_files)
            self.set_deploy_weights_file()
            if property=='dbs':
                if os.path.isfile(value+'/meanfile.binaryproto'):
                    self.deploy_meanfile=value+'/meanfile.binaryproto'
                else:
                    self.deploy_meanfile=None


    def set_deploy_weights_file(self,weight_file=''):
        if not weight_file:
            self.deploy_weights_file=self.final_weights_files
        elif weight_file in self.available_weights_files:
            self.deploy_weights_file=weight_file
        else:
            print('Weight file not found')
            raise NameError


    def get_performance(self,data_file,crop_size,prediction_key,save_data_flag=True,labels_col=[],layer='',weights_file='',save_data_suffix=''):

        if not weights_file:
            weights_file=self.deploy_weights_file
        else:
            if os.path.isfile(weights_file):
                weights_file=weights_file
            else:
                raise NameError
        deploy_inputs = {'net': self.deploy_proto,
                         'mean': self.deploy_meanfile, 'model': weights_file,
                         'mean_type': 'binaryproto'}
        infer_params = {'img_w': crop_size, 'img_h': crop_size}
        infer = Inference(deploy_inputs, params=infer_params)
        data = pd.read_csv(data_file, header=None, sep=' ')

        # print('---------------------------------')
        # print(data[0][0])
        # print('---------------------------------')
        pred_class = [ ]
        pred_prob = [ ]
        representations=[]
        for idx2, file in enumerate(data[0]):
            print('Running Infrence on {0} ({1} or {2}) with model {3}'.format(data_file.split('/')[-1], idx2, len(data[0]),self.dir_strc.main.split('/')[-1]))
            img = cv2.imread(file)
            if layer:
                output,reps=infer.predict(img,layer=layer)
                representations.append(reps[0])
            else:
                output = infer.predict(img)
            pred_class.append(output[prediction_key][0].argmax())
            pred_prob.append(output[prediction_key][0].max())



        data['pred_class']=pred_class
        data['pred_prob']=pred_prob


        if labels_col:
            targets=data[labels_col]
            cnf_matrix, cnf_matrix_pct = compute_cnf_matrix(targets, data['pred_class'])
            cnf_matrix_array = np.vstack([cnf_matrix, cnf_matrix_pct])
        else:
            cnf_matrix_array=None
            data['true_label']=''
        data_2_save=data[[0,'pred_class','pred_prob','true_label']]
        if save_data_flag:
            preds_save_file_name=self.dir_strc.analysis_results_files+'/'+data_file.split('/')[-1]+'_predictions'+save_data_suffix
            data_2_save.to_csv(preds_save_file_name,sep=',',index=None,header=['image_name','true_label','dl_pred','dl_prob'])
            print('Predictions saved to {0}'.format(preds_save_file_name))
            if labels_col:
                cnf_save_file_name = self.dir_strc.analysis_results_files+'/'+data_file.split('/')[-1] + '_cnf_matrix'+save_data_suffix
                np.savetxt(cnf_save_file_name, cnf_matrix_array, '%f')
                print('Confusion saved to {0}'.format(cnf_save_file_name))
            if layer:
                embeds_save_file_name = self.dir_strc.analysis_results_files + '/' + data_file.split('/')[
                    -1] + '_embeds'+save_data_suffix
                np.save(embeds_save_file_name,representations)

        return data_2_save,cnf_matrix_array

if __name__=='__main__':

    dir_path='/home/ubuntu/online_learning/model_files/sqNet_domain_adapt_transfer_learning/_base'
    trainer=Load_trainer(dir_path)



    crop_size=227
    prediction_key='prob'
    data_file='/home/ubuntu/online_learning/data/loc_data_txt_aws/source_data_files.txt'
    data, cnf_matrix_array=trainer.get_performance(data_file,crop_size,prediction_key,labels_col=1,layer='conv10')


    # print trainer.available_weights_files
    # print trainer.deploy_proto
    # print trainer.deploy_meanfile
    print cnf_matrix_array
