import sys
import numpy as np
import logging
import time
import os
caffe_root=os.environ["CAFFE_ROOT"]
sys.path.insert(0, caffe_root + 'python')
import caffe

class Train_solver:
    def __init__(self,gpu_id=-1,run_tag=[],convg_param=[],caffe_root=''):
        if not caffe_root:
            caffe_root=os.environ["CAFFE_ROOT"]
        sys.path.insert(0, caffe_root + 'python')
        import caffe

        if gpu_id > -1:
            caffe.set_mode_gpu()
            # Select a GPU
            GPU_ID = gpu_id
            # caffe.set_device(GPU_ID)
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
            print os.environ["CUDA_VISIBLE_DEVICES"]
        if gpu_id == -1:
            caffe.set_mode_gpu()
            os.environ["CUDA_VISIBLE_DEVICES"] = str(1)
            print 'CUDA_VISIBLE_DEVICES'
            print os.environ["CUDA_VISIBLE_DEVICES"]
        if gpu_id == -2:
            caffe.set_mode_cpu()
        # print '---------------------------------------------------------------'
        # print gpu_id

        if not run_tag:
            run_tag = 'Log_'+ time.strftime("%Y%m%d-%H%M%S")

        self.logger = logging.getLogger(run_tag)
        log_hdlr = logging.FileHandler(run_tag+'_train'+ '.log')
        log_formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
        log_hdlr.setFormatter(log_formatter)
        self.logger.addHandler(log_hdlr)
        self.logger.setLevel(logging.DEBUG)

        # parameters for convergence
        self.convg_exp_param=0.4
        self.convg_stop=False
        self.mav_acc=0

    def check_convg(self,accuracy):
        self.mav_test_acc=self.convg_exp_param*accuracy+(1-self.convg_exp_param)*self.mav_acc
        if self.mav_acc>0.99:
            self.convg_stop=True
        else:
            self.convg_stop=False



    def train(self,solver_proto,niter=1000,weights_file=[],perf_layers=[],display_interval=[],weights_save_interval=[],weights_save_prefix='weights'):
        print 'CUDA_VISIBLE_DEVICES'
        print os.environ["CUDA_VISIBLE_DEVICES"]

        solver=caffe.get_solver(solver_proto)

        if weights_file:
            solver.net.copy_from(weights_file)

        if not perf_layers:
            perf_layers=['loss']
        if not isinstance(perf_layers,list):
            perf_layers=[perf_layers]

        if not display_interval:
            display_interval=np.floor(niter/10) # show results 10 times

        if not weights_save_interval:
            weights_save_interval = np.floor(niter/100)

        perf_list=[]
        for it in range(niter):
            solver.step(1)
            perf=[solver.net.blobs[lyr].data for lyr in perf_layers]
            perf_list.append(perf)

            if it % display_interval == 0 or it % weights_save_interval ==0 or it== niter-1 or it == niter:
                print_str='Iteration '+str(it)+' of '+ str(niter)+' '
                for idx,perf_layer in enumerate(perf_layers):
                    print_str=print_str+' '+ perf_layer+ ': '+str(perf[idx])
                print print_str
	
            if it % weights_save_interval ==0 or it== niter-1 or it == niter:
                weights_save_fileName = weights_save_prefix+'_'+str(it)+'.caffemodel'
                self.logger.info("Weights: %s, Performance: %s"%(weights_save_fileName, print_str))
                solver.net.save(weights_save_fileName)

            #
            # self.check_convg(perf[0])
            # if self.convg_stop:
            #     break

        return perf_list, weights_save_fileName, it


if __name__=="__main__":
    train_slvr=Train_solver()
    solver_proto='/raid/karthik/online_learning/model_files/base_models/sns_model2/sns_solver_sq_test.prototxt'
    train_slvr.train(solver_proto)



