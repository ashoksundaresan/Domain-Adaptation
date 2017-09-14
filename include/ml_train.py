import os
import time
import numpy as np
import sys
import lmdb
import cv2
from include import gen_netProto
from include import train_solver
import logging

# Select a GPU
# GPU_ID = 0
# caffe.set_device(GPU_ID)

from caffe import layers as L, params as P
from caffe.proto import caffe_pb2

class ML_train:
    def __init__(self,model_id=[],serving_dir='',caffe_root=''):
        if not caffe_root:
            caffe_root=os.environ["CAFFE_ROOT"]
        sys.path.insert(0, caffe_root + 'python')
        import caffe
        caffe.set_mode_gpu()
        self.caffe_root=caffe_root

        if not model_id:
            model_id='Model_'+time.strftime("%Y%m%d-%H%M%S")
        self.model_id=model_id
        self.dir_strc=type('',(),{})()
        self.create_dir_structure(serving_dir)
        self.logger=logging.getLogger(model_id)
        log_hdlr=logging.FileHandler(self.dir_strc.main+'/training_log.log')
        log_formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
        log_hdlr.setFormatter(log_formatter)
        self.logger.addHandler(log_hdlr)
        self.logger.setLevel(logging.DEBUG)


    def create_dir_structure(self,serving_dir=''):
        if serving_dir:
            cur_dir=serving_dir
        else:
            cur_dir='../..'
        fldrs=self.model_id.split('/')
        dir_str=cur_dir + '/model_files/'
        os.system('mkdir ' + dir_str)
        for fldr in fldrs:
            os.system('mkdir ' + dir_str + fldr)
            dir_str=dir_str+'/'+fldr+'/'

        # if not os.path.isdir(cur_dir+'../../model_files/'+self.model_id):
        #     os.system('mkdir ' + cur_dir+'../../model_files/'+self.model_id)
        self.dir_strc.main=cur_dir+'/model_files/'+self.model_id
        self.dir_strc.dbs= cur_dir+'/model_files/'+self.model_id+'/dbs'
        self.dir_strc.weight_files=cur_dir+'/model_files/'+self.model_id+'/weight_files'
        self.dir_strc.model_protofiles=cur_dir+'/model_files/'+self.model_id+'/model_protofiles'
        self.dir_strc.analysis_results_files=cur_dir+'/model_files/'+self.model_id+'/analysis_results_files'
        os.system('mkdir ' +self.dir_strc.dbs)
        os.system('mkdir ' +self.dir_strc.weight_files)
        os.system('mkdir ' +self.dir_strc.model_protofiles)
        os.system('mkdir ' + self.dir_strc.analysis_results_files)

    def create_img_lmdb_os(self,source,resize_height ,resize_width,out_base_name=[]):
        if not out_base_name:
            out_base_name=source.split('/')[-1].split('.')[0]
        out_lmdb=self.dir_strc.dbs+'/'+out_base_name

        exec_str="GLOG_logtostderr=1 " + self.caffe_root+"/build/tools/convert_imageset --resize_height=" +str(resize_height) +" --resize_width="+ \
            str(resize_width)+" --shuffle / "+ source +" "+ out_lmdb
        os.system(exec_str)
        #print exec_str

        return out_lmdb

    def create_img_lmdb(self,source,img_width,img_height,img_channels,sep=' ',out_base_name=[]):
        """ Source: Text file with labels
        out_base_name: Filename
        """
        if not out_base_name:
            out_base_name=source.split('/')[-1].split('.')[0]
        out_lmdb=self.dir_strc.dbs+'/'+out_base_name
        with open(source,'r') as f:
            lines=f.readlines()
            x=[line.split(sep)[0] for line in lines]
            y=[int(line.split(sep)[1]) for line in lines]
            db=lmdb.open(out_lmdb,map_size=int(1e12))
        print('Creating db with %s files')%(len(x))

        with db.begin(write=True) as txn:
            for idx,img_path in enumerate(x):
                img=cv2.imread(img_path,cv2.IMREAD_COLOR)
                img=cv2.resize(img,(img_width,img_height))
                label=y[idx]
                datum=caffe_pb2.Datum()
                datum.channels=img_channels
                datum.height=img_height
                datum.width=img_width
                datum.data=np.rollaxis(img,2).tobytes()#RGB --> BGR
                datum.label=label

                str_id='{:6d}'.format(idx)
                txn.put(str_id,datum.SerializeToString())

                if idx%1000==0:
                    print('Creating (%s): %s,  img: %s')%(str_id,out_lmdb,img_path)

        return self.dir_strc.dbs+'/'+out_lmdb

    def compute_mean(self,lmdb_source,mean_file_path=[]):
        if not mean_file_path:
            mean_file_path='/'.join(lmdb_source.split('/')[0:-1])+'/meanfile.binaryproto'
        os.system(self.caffe_root+'/build/tools/compute_image_mean '+ lmdb_source+'  '+mean_file_path)
        return mean_file_path

    def set_lmdbs(self, train_lmdb, test_lmdb,mean_file,crop_size,img_channels):
        self.train_lmdb = train_lmdb
        self.test_lmdb = test_lmdb
        self.mean_file=mean_file
        self.crop_size=crop_size
        self.img_channels=img_channels

    def gen_archProtoFiles(self, topology, params, run_tag='base',batch_size=128):
        train_proto_fileName= self.dir_strc.model_protofiles+'/train_'+run_tag+'.prototxt'
        test_proto_fileName = self.dir_strc.model_protofiles + '/test_' + run_tag + '.prototxt'
        deploy_proto_fileName = self.dir_strc.model_protofiles + '/deploy_' + run_tag + '.prototxt'


        self.logger.info('Generating training prototxt file: %s' % (train_proto_fileName))
        self.logger.info('Generating testing prototxt file: %s' % (test_proto_fileName))
        self.logger.info('Generating deploy prototxt file: %s' % (deploy_proto_fileName))


        gen_proto=gen_netProto.Gen_netProto(topology,params)

        print('--> Generating training prototxt file: %s') % (train_proto_fileName)
        gen_proto.assemble_net(lmdb=self.train_lmdb, mean_file=self.mean_file,
                               proto_filename=train_proto_fileName,
                               batch_size=batch_size, crop_size=self.crop_size, img_channels=self.img_channels,
                               phase='TRAIN')

        print('--> Generating testing prototxt file: %s') % (test_proto_fileName)
        gen_proto.assemble_net(lmdb=self.test_lmdb, mean_file=self.mean_file,
                               proto_filename=test_proto_fileName,
                               batch_size=batch_size, crop_size=self.crop_size, img_channels=self.img_channels,
                               phase='TEST')

        print('--> Generating deploy prototxt file: %s') % (deploy_proto_fileName)
        gen_proto.assemble_net(lmdb=[], mean_file=self.mean_file,
                               proto_filename=deploy_proto_fileName,
                               batch_size=batch_size, crop_size=self.crop_size, img_channels=self.img_channels,
                               phase='DEPLOY')
        return train_proto_fileName, test_proto_fileName, deploy_proto_fileName

    def set_archProtoFiles(self, train_proto_in,test_proto_in,deploy_proto_in, run_tag='base'):
        train_proto_fileName = self.dir_strc.model_protofiles + '/train_' + run_tag + '.prototxt'
        test_proto_fileName = self.dir_strc.model_protofiles + '/test_' + run_tag + '.prototxt'
        deploy_proto_fileName = self.dir_strc.model_protofiles + '/deploy_' + run_tag + '.prototxt'

        os.system('cp '+ train_proto_in +' '+train_proto_fileName)
        os.system('cp ' + test_proto_in+ ' ' + test_proto_fileName)
        os.system('cp ' + deploy_proto_in + ' ' + deploy_proto_fileName)
        return train_proto_fileName, test_proto_fileName, deploy_proto_fileName

    def gen_solverProtoFiles(self,train_proto_path,solver_params,test_proto_path='',run_tag=[]):
        dir_name = '/'.join(train_proto_path.split('/')[0:-1])
        if not run_tag:
            try:
                run_tag = train_proto_path.split('_')[-1].split('.')[0]
            except:
                run_tag='base'

        solver_proto_fileName=dir_name+'/solver_'+run_tag+'.prototxt'
        print('--> Generating solver prototext file: %s')%(solver_proto_fileName)
        self.logger.info('Generating solver prototext file: %s' % (solver_proto_fileName))

        weights_save_dir = self.dir_strc.weight_files
        model_prefix = weights_save_dir + '/weights_' + run_tag
        print('--> Weights_prefix: %s')%(model_prefix)
        self.logger.info('Weights_prefix set as : %s' %(model_prefix))

        gen_proto = gen_netProto.Gen_netProto(solver_params=solver_params)
        if test_proto_path:
            gen_proto.solverProto(solver_proto_fileName, train_proto_path, test_proto_path, model_prefix)
        else:
            gen_proto.solverProto_one_net(solver_proto_fileName, train_proto_path, model_prefix)

        return solver_proto_fileName

    def train_solver(self,solver_proto,niter=1000,weights_file=[],perf_layers=[],display_interval=[],weights_save_interval=[],run_tag=[]):

        if not run_tag:
            try:
                run_tag=solver_proto.split('/').split('-')[-1].split('.')[0]
            except:
                run_tag='base'
        slvr = train_solver.Train_solver(run_tag=self.dir_strc.weight_files+'/'+run_tag)

        print('--> Training using solver: %s')%(solver_proto)
        self.logger.info('Training using solver: %s'%(solver_proto))

        weights_save_dir=self.dir_strc.weight_files
        weights_save_prefix = weights_save_dir+'/weights_'+run_tag
        print('--> Weights being saved as %s')%(weights_save_prefix)
        self.logger.info('Weights being saved as %s' % (weights_save_prefix))
        slvr_perf_vec,final_weights,iter_cmpltd=slvr.train(solver_proto,niter,weights_file,perf_layers,display_interval,weights_save_interval,weights_save_prefix)

        self.logger.info('Final solver performace: %s' % (slvr_perf_vec[-1]))
        self.logger.info('Number of iterations: %s' % (iter_cmpltd))

        return slvr_perf_vec, final_weights, weights_save_dir

if __name__ == '__main__':
        model_id='Style'
        style_ml=ML_train(model_id)

        training='/home/karthik/caffe/data/flickr_style/train.txt'
        testing='/home/karthik/caffe/data/flickr_style/test.txt'
        img_width=256
        img_height=256
        train_lmdb=style_ml.create_img_lmdb(training,'train',img_width,img_height,img_channels=3)
        test_lmdb=style_ml.create_img_lmdb(testing,'test',img_width,img_height,img_channels=3)
        print(train_lmdb)
        print(test_lmdb)
