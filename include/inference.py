import sys
import cv2
import numpy as np
import pandas as pd
# caffe_root=os.en'/home/karthik/caffe'
# sys.path.insert(0, caffe_root + 'python')
# import caffe
#
from collections import defaultdict
import os
class Inference:

    def __init__(self,deploy_inputs,params=[],gpu_id=-1,caffe_root=''):
        if not caffe_root:
            caffe_root=os.environ["CAFFE_ROOT"]
        sys.path.insert(0, caffe_root + 'python')
        import caffe
        caffe.set_mode_gpu()
        self.caffe_root=caffe_root
        from caffe.proto import caffe_pb2

        self.net_proto = deploy_inputs['net']
        self.mean_file = deploy_inputs['mean']
        self.weights = deploy_inputs['model']
        self.mean_type=deploy_inputs['mean_type']
        # caffe initialization
        if gpu_id > -1:
            caffe.set_mode_gpu()
            # Select a GPU
            GPU_ID = gpu_id
            caffe.set_device(GPU_ID)
        if gpu_id == -1:
            caffe.set_mode_gpu()

        if gpu_id == -2:
            caffe.set_mode_cpu()

        ###
        print('----Running inference with---')
        print('Net protofile {0}'.format(self.net_proto))
        print('Weights file {0}'.format(self.weights))
        print('-------')

        self.net = caffe.Net(self.net_proto, self.weights, caffe.TEST)
        self.params = type('', (), {})()
        if not params:
            self.params.img_w = self.net.blobs['data'].data.shape[2]
            self.params.img_h = self.net.blobs['data'].data.shape[2]
        else:
            self.params.img_w = params['img_w']
            self.params.img_h = params['img_h']

        if self.mean_type=='binaryproto':
            mean_blob = caffe_pb2.BlobProto()
            with open(self.mean_file) as f:
                mean_blob.ParseFromString(f.read())
            mean_array = np.asarray(mean_blob.data, dtype=np.float32).reshape(
                    (mean_blob.channels, mean_blob.height, mean_blob.width))
            #mean_array = mean_array_temp[0, :, :]
            mu0, mu1, mu2 = np.mean(mean_array[0]), np.mean(mean_array[1]), np.mean(mean_array[2])
            mu = np.array([mu0, mu1, mu2])
            transformer = caffe.io.Transformer({'data': self.net.blobs['data'].data.shape})
            transformer.set_transpose('data', (2, 0, 1))  # channels, width, height
            transformer.set_mean('data', mu)

        if self.mean_type =='npy':
            mean_array=np.load(self.mean_file)
            mu=mean_array.mean(1).mean(1)
            transformer = caffe.io.Transformer({'data': self.net.blobs['data'].data.shape})
            transformer.set_transpose('data', (2, 0, 1))
            transformer.set_mean('data', mu)
            # Assumes BGR Image (opencv format), for rgb need to add code to translate
            # transformer.set_raw_scale('data', 255.0) #
            # self.net.blobs['data'].reshape(1, 3, self.params.img_w , self.params.img_w )# change later

        self.transformer=transformer

    def predict(self,img,_=[],layer=''):
        #print self.params.img_w
        img=cv2.resize(img, (self.params.img_h, self.params.img_w), interpolation=cv2.INTER_CUBIC)
        # Transform input
        self.net.blobs['data'].data[...] = self.transformer.preprocess('data', img)

        output=self.net.forward()
        if layer:
            embdings = self.net.blobs[layer].data.copy()
            return output, embdings
        else:
            return output

    def infer_many(self,data_txt_file, pred_key,layer='', header=None, sep=' '):
        analysis_data = pd.read_csv(data_txt_file, header=None, sep=' ')
        img_file_list = analysis_data[0]
        data_infr = defaultdict()
        data_embds = []

        for idx, img_file in enumerate(img_file_list):
            img = cv2.imread(img_file)
            if layer:
                out, embds = self.predict(img, layer=layer)  # Need to modify code to handle multiple embeddings
            else:
                out = self.predict(img, layer=layer)
                embds=[]

            pred_class = out[pred_key][0].argmax()
            pred_prob = out[pred_key][0].max()

            data_infr[idx] = [img_file, pred_class, pred_prob]
            data_embds.append(embds)

            print('Inferred image {0} of {1}').format(idx,len(img_file_list))


        data_infr=pd.DataFrame.from_dict(data_infr, orient='index')
        data_infr.columns=['filename','pred_class','pred_class_p']
        return data_infr,data_embds

    def get_arch_params(self,print_flag=0):
        arch_params=[]
        for layer_name, blob in self.net.blobs.iteritems():
            arch_params.append([layer_name,blob.data.shape])
            if print_flag==1:
                print layer_name + '\t' + str(blob.data.shape)
        return arch_params

    def get_embeddings(self,img,layers):
        if not isinstance(layers,list):
            layers=[layers]
        representations=[]
        for idx,layer in enumerate(layers):
            rep = self.net.blobs[layer].data
            representations.append(rep)
        return representations



if __name__=='__main__':
    deploy_inputs = {'net': trainer.dir_strc.model_protofiles + '/deploy.prototxt',
                     'mean': trainer.dir_strc.dbs + '/meanfile.binaryproto', 'model': deploy_weights,
                     'mean_type': 'binaryproto'}
    infer_params = {'img_w': self.params['crop_size'], 'img_h': self.params['crop_size']}
    infer = Inference(deploy_inputs, params=infer_params)







