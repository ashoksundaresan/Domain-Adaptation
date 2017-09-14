caffe_root = '/home/karthik/caffe/'  # this file should be run from {caffe_root}/examples (otherwise change this line)
import sys
import os
sys.path.insert(0, caffe_root + 'python')
import caffe
# caffe.set_device(0)
caffe.set_mode_gpu()
## Select a GPU
##GPU_ID = 0
##caffe.set_device(GPU_ID)
from caffe import layers as L, params as P
from caffe.proto import caffe_pb2

class Gen_netProto:
    def __init__(self, topology=[], params=[],solver_params=[]):
        self.params = params
        self.topology = topology
        self.solver_params = solver_params
        self.frozen = [dict(lr_mult=0)] * 2
        weight_param = dict(lr_mult=1, decay_mult=1)
        bias_param   = dict(lr_mult=2, decay_mult=0)
        self.learn = [weight_param, bias_param]
        # Write a function to loop throug params and create protos, and then solve

    def conv2d(self, bottom, layer_name):
        conv_params = self.params[layer_name]
        if layer_name in self.params['layers_to_learn']: param = self.frozen
        else: param = self.learn
        pre_activation = L.Convolution(bottom, param=param, convolution_param=conv_params, name=layer_name)
        conv = L.ReLU(pre_activation, in_place=True)
        return conv

    def fc(self, bottom, layer_name, dropout_flag_in):
        fc_params = self.params[layer_name]
        if 'dropout_ratio' in fc_params:
            dropout = dict(dropout_ratio=fc_params['dropout_ratio'])
            fc_params.pop('dropout_ratio')
            dropout_flag = True
        else:
            dropout_flag = False
        if layer_name in self.params['layers_to_learn']: param = self.frozen
        else: param = self.learn
        pre_activation = L.InnerProduct(bottom, param=param, inner_product_param=fc_params, name=layer_name)
        if layer_name=='output':  local = pre_activation
        else: local = L.ReLU(pre_activation, in_place=True)
        if dropout_flag_in and dropout_flag:
            return L.Dropout(local, in_place=True, dropout_param=dropout)
        else:
            return local

    def assemble_net(self, lmdb, mean_file, proto_filename, batch_size, crop_size,img_channels, phase):
        layer_dict = {}
        n = caffe.NetSpec()

        if phase in ['Train', 'TRAIN','train']:
            dropout_flag = True
            loss_layer_flag = True
            prob_layer_flag = False
            acc_layer_flag = True

        if phase in ['test', 'TEST', 'Test']:
            dropout_flag = False
            loss_layer_flag = True
            prob_layer_flag = False
            acc_layer_flag = True

        if phase in ['deploy', 'DEPLOY', 'Deploy']:
            dropout_flag = False
            loss_layer_flag = False
            prob_layer_flag = True
            acc_layer_flag = False

        if phase in ['deploy', 'DEPLOY', 'Deploy']:
            n.data = L.Input(input_param= {'shape':{'dim':[1,img_channels,crop_size,crop_size]}})
        else:
            n.data, n.label = L.Data(transform_param={'crop_size': crop_size, 'mean_file':mean_file}, data_param={'source':lmdb,'batch_size':batch_size, 'backend': P.Data.LMDB},
                                        ntop=2)

        for i, layer_name in enumerate(self.topology):
            #print layer_name
            if 'conv' in layer_name:
                if i==0:
                    n.conv = self.conv2d(n.data, layer_name)
                else:
                    n.conv = self.conv2d(layer_dict[str(i-1)], layer_name)
                layer_dict[str(i)] = n.conv

            elif 'pool' in layer_name:
                pool_params = {'pool': 0, 'kernel_size': 3, 'stride': 2}
                n.pool = L.Pooling(layer_dict[str(i - 1)], pooling_param=pool_params, name=layer_name)
                layer_dict[str(i)] = n.pool

            elif 'fc' in layer_name or 'output' in layer_name:
                n.fc = self.fc(layer_dict[str(i - 1)], layer_name, dropout_flag)
                layer_dict[str(i)] = n.fc

            elif 'lrn' in layer_name:
                n.norm = L.LRN(layer_dict[str(i - 1)], local_size=5, alpha=1e-4, beta=0.75, name=layer_name)
                layer_dict[str(i)] = n.norm

            if loss_layer_flag:
                n.loss = L.SoftmaxWithLoss(layer_dict[str(i)], n.label, name='loss')
            if prob_layer_flag:
                n.prob = L.Softmax(layer_dict[str(i)], name='loss')
            if acc_layer_flag:
                n.accuracy = L.Accuracy(layer_dict[str(i)], n.label, name='accuracy')

        print layer_dict
	with open(proto_filename, 'w') as f:
                f.write(str(n.to_proto()))

    def solverProto(self, solver_path, train_net_path, test_net_path, model_prefix):
        s = caffe_pb2.SolverParameter()
        s.train_net = train_net_path
        s.test_net.append(test_net_path)
        s.test_interval = self.solver_params['test_interval']
        s.test_iter.append(100)  # Test 250 "batches" each time we test.
        s.max_iter=self.solver_params['max_iter']
        s.iter_size=16#
        # s.max_iter = 100000     # # of times to update the net (training iterations)
        # Set the initial learning rate for stochastic gradient descent (SGD).
        s.lr_policy = self.solver_params['lr_policy']
        s.base_lr = self.solver_params['base_lr']
        s.gamma = self.solver_params['gamma']
        s.momentum = self.solver_params['momentum']
        # s.momentum2 = 0.999
        s.weight_decay = self.solver_params['weight_decay']
        s.stepsize = self.solver_params['step_size']
        s.display = self.solver_params['display']
        s.snapshot = self.solver_params['snapshot_interval']
        s.snapshot_prefix = model_prefix
        # s.type = "Adam"
        s.solver_mode = caffe_pb2.SolverParameter.GPU

        with open(solver_path, 'w') as f:
            f.write(str(s))

    def solverProto_one_net(self, solver_path, net_path, model_prefix):
        s = caffe_pb2.SolverParameter()
        s.net = net_path
        # s.test_net.append(test_net_path)
        s.test_interval = self.solver_params['test_interval']
        s.test_iter.append(100)  # Test 250 "batches" each time we test.
        s.max_iter=self.solver_params['max_iter']
        s.iter_size=16#
        # s.max_iter = 100000     # # of times to update the net (training iterations)
        # Set the initial learning rate for stochastic gradient descent (SGD).
        s.lr_policy = self.solver_params['lr_policy']
        s.base_lr = self.solver_params['base_lr']
        s.gamma = self.solver_params['gamma']
        s.momentum = self.solver_params['momentum']
        # s.momentum2 = 0.999
        s.weight_decay = self.solver_params['weight_decay']
        s.stepsize = self.solver_params['step_size']
        s.display = self.solver_params['display']
        s.snapshot = self.solver_params['snapshot_interval']
        s.snapshot_prefix = model_prefix
        # s.type = "Adam"
        s.solver_mode = caffe_pb2.SolverParameter.GPU

        with open(solver_path, 'w') as f:
            f.write(str(s))

