import os
import numpy as np
import pandas as pd

class Create_trainer:
    def __init__(self,dir_path,model_id):
        self.model_id=model_id
        self.dir_strc = type('', (), {})()
        self.create_dir_structure(dir_path)

    def create_dir_structure(self, serving_dir):
        if serving_dir:
            cur_dir = serving_dir
        else:
            cur_dir = '../..'
        fldrs = self.model_id.split('/')

        dir_str = cur_dir + '/model_files/'
        os.system('mkdir '+dir_str)

        for fldr in fldrs:
            print(dir_str + fldr)
            os.system('mkdir ' + dir_str + fldr)
            dir_str = dir_str + '/' + fldr + '/'

        self.dir_strc.main = cur_dir + '/model_files/' + self.model_id
        self.dir_strc.dbs = cur_dir + '/model_files/' + self.model_id + '/dbs'
        self.dir_strc.weight_files = cur_dir + '/model_files/' + self.model_id + '/weight_files'
        self.dir_strc.model_protofiles = cur_dir + '/model_files/' + self.model_id + '/model_protofiles'
        self.dir_strc.analysis_results_files = cur_dir + '/model_files/' + self.model_id + '/analysis_results_files'
        os.system('mkdir ' + self.dir_strc.dbs)
        os.system('mkdir ' + self.dir_strc.weight_files)
        os.system('mkdir ' + self.dir_strc.model_protofiles)
        os.system('mkdir ' + self.dir_strc.analysis_results_files)

    def set_components(self,trainer_files):
        self.trainer_files = trainer_files

        # Copy data/databases
        os.system('cp ' + trainer_files['meanfile_binaryproto'] + ' ' + self.dir_strc.dbs + '/meanfile.binaryproto')
        if 'train_lmbd' in trainer_files.keys():
            os.system('cp -r ' + trainer_files['train_lmbd'] + ' ' + self.dir_strc.dbs + '/train_lmbd')
        else:
            os.system('cp ' + trainer_files['train_txt'] + ' ' + self.dir_strc.dbs + '/train_txt')


        if 'val_lmbd' in trainer_files.keys():
            os.system('cp -r ' + trainer_files['val_lmbd'] + ' ' + self.dir_strc.dbs + '/val_lmbd')
        else:
            os.system('cp ' + trainer_files['val_txt'] + ' ' + self.dir_strc.dbs + '/val_txt')

        # Copy each prototxt files
        os.system(
            'cp ' + trainer_files['train'] + ' ' + self.dir_strc.model_protofiles + '/train_original.prototxt')
        if 'test' in trainer_files.keys():
            os.system('cp ' + trainer_files['test'] + ' ' + self.dir_strc.model_protofiles + '/test_original  .prototxt')
        os.system('cp ' + trainer_files['solver'] + ' ' + self.dir_strc.model_protofiles + '/solver_original.prototxt')
        os.system('cp ' + trainer_files['deploy'] + ' ' + self.dir_strc.model_protofiles + '/deploy.prototxt')



if __name__=='__main__':
    dir_path='/Users/karthikkappaganthu/Documents/online_learning/model_files/temp'
    ct=Create_trainer(dir_path,'test')
    deploy_inputs = {'net': self.deploy_proto,
                     'mean': self.deploy_meanfile, 'model': weights_file,
                     'mean_type': 'binaryproto'}


