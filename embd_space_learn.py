import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
import include.low_dim_classifier as ld_clf
#from include.autoencoder_based_low_dim_reps import autoencode_reps
from sklearn.preprocessing import StandardScaler
from sklearn.semi_supervised import LabelSpreading
from include import samples_to_label
from sklearn.decomposition import PCA
import os
class Embeds_classify:
    def __init__(self,source_embds_file,target_embds_file,source_data_txt_file,target_data_txt_file,embds_param={'type':'convs','n_convs':2,'standardize':False},true_labels_col='1',index_col='',sep=',',source_test_frac=.2,target_test_frac=.2,serving_dir='',inter_save=True,dl_pred_key='dl_pred',save_suffix=''):
        # Load data/prediction files
        if index_col:
            self.source_data=pd.read_csv(source_data_txt_file,sep=sep,index_col=index_col)
            self.target_data=pd.read_csv(target_data_txt_file,sep=sep,index_col=index_col)
        else:
            self.source_data = pd.read_csv(source_data_txt_file, sep=sep)
            self.target_data = pd.read_csv(target_data_txt_file, sep=sep)

        self.source_embds=np.load(source_embds_file)
        self.target_embds=np.load(target_embds_file)

        self.n_source = self.source_embds.shape[0]
        self.n_target = self.target_embds.shape[0]
        # Assign true labels
        self.true_labels_col = true_labels_col
        self.source_labels = self.source_data[self.true_labels_col]
        self.target_labels = self.target_data[self.true_labels_col]

        # Initialize paramerter to enable control
        self.tsne_computed = 0
        self.embds_space_performance={}
        self.low_confidence_metric_cols=[]
        self.embds_space_grouping=[]
        if dl_pred_key in self.target_data.columns:
            self.embds_space_classifiers=[dl_pred_key]
        else:
            self.embds_space_classifiers = []
        self.low_dim_feat_cols={'tsne':[],'pca':[]}

        # Determine if target true labels are available
        if np.all(np.isnan(self.target_labels)):
            self.target_true_labels_available=0
        else:
            self.target_true_labels_available = 1

        # Extract embeddings based on the type of last layer
        if embds_param['type'] == 'convs' :# for
            source_feats=[]
            for row in self.source_embds:
                feat=[]
                for i_c in range(embds_param['n_convs']):
                    if embds_param['standardize']:
                        scaler = StandardScaler()
                        feats_in=scaler.fit_transform(row[i_c,:].reshape(1,-1)[0])
                    else:
                        feats_in = row[i_c, :].reshape(1, -1)[0]
                    feat=np.concatenate((feat,feats_in))
                source_feats.append(feat)
            self.source_embds_vec=np.array(source_feats)
            target_feats=[]
            for row in self.target_embds:
                feat = []
                for i_c in range(embds_param['n_convs']):
                    if embds_param['standardize']:
                        scaler = StandardScaler()
                        feats_in=scaler.fit_transform(row[i_c,:].reshape(1,-1)[0])
                    else:
                        feats_in = row[i_c, :].reshape(1, -1)[0]
                    feat = np.concatenate((feat, feats_in))
                target_feats.append(feat)
            self.target_embds_vec = np.array(target_feats)
        else:
            print('Only embds_param[type]=convs implemented')
            raise NotImplemented

        # Set test_train_split
        self.set_test_train_data(source_test_frac=source_test_frac,target_test_frac=target_test_frac)

        # Save data for visualization
        self.inter_save=inter_save
        self.save_suffix=save_suffix
        if not serving_dir:
            os.system('mkdir temp')
            serving_dir='temp/'
        self.serving_dir=serving_dir

        if self.inter_save:
            self.save_perforamance(self.serving_dir,suffix=self.save_suffix)

    def set_test_train_data(self,source_test_mask=[],target_test_mask=[],source_test_frac=.2,target_test_frac=.2):
        if not source_test_mask:
            source_test_mask = np.random.rand(self.n_source) < source_test_frac
        if not target_test_mask:
            target_test_mask = np.random.rand(self.n_target) < target_test_frac
        self.source_data['test_set_flag']=source_test_mask
        self.target_data['test_set_flag']=target_test_mask

        self.source_embds_test=self.source_embds_vec[source_test_mask,:]
        self.source_embds_train = self.source_embds_vec[~source_test_mask,:]
        self.target_embds_test=self.target_embds_vec[target_test_mask,:]
        self.target_embds_train=self.target_embds_vec[~target_test_mask,:]

        self.source_labels_test=self.source_labels[source_test_mask]
        self.source_labels_train = self.source_labels[~source_test_mask]
        self.target_labels_test=self.target_labels[target_test_mask]
        self.target_labels_train=self.target_labels[~target_test_mask]

    def low_dim_classifier(self,methods=['RandomForestClassifier'],train_feat_space='embeds',train_mode='source_and_target'):
        print('-----------------------------------------------------------------------')
        print('Computing target labels in {0} space using {1}'.format(train_feat_space,methods))
        key_prefix = train_feat_space + 'Space_' + train_mode + 'Mode'
        # Embeddings Space Features
        if train_feat_space=='embeds':
            source_data_feats = self.source_embds_vec
            target_data_feats = self.target_embds_vec
            source_test_feats = self.source_embds_test
            target_test_feats = self.target_embds_test
            if train_mode=='source_and_target':
                if self.target_true_labels_available==1:
                    trn_feats = np.concatenate((self.source_embds_train,self.target_embds_train))
                    trn_labels = np.concatenate((self.source_labels_train, self.target_labels_train))
                    test_feats = np.concatenate((self.source_embds_test,self.target_embds_test))
                    test_labels = np.concatenate((self.source_labels_test, self.target_labels_test))
                else:
                    print('Target labels not available, cannot train in this mode')
                    return 'Failed'

            elif train_mode=='source':
                trn_feats = self.source_embds_train
                trn_labels = np.array(self.source_labels_train)
                test_feats = self.source_embds_test
                test_labels =  np.array(self.source_labels_test)
            else:
                # raise NotImplemented
                return 'Not Implemented'

        # Emeds TSNE Space features
        elif train_feat_space == 'embeds_tsne':
            feat_cols = []
            for idx in range(self.n_tsne_components):
                feat_cols.append('embeds_tsne_' + str(idx))
            source_test_feats = self.source_data[self.source_data['test_set_flag'] == True][feat_cols].as_matrix()
            source_trn_feats = self.source_data[self.source_data['test_set_flag'] == False][feat_cols].as_matrix()
            target_test_feats = self.target_data[self.target_data['test_set_flag'] == True][feat_cols].as_matrix()
            target_trn_feats = self.target_data[self.target_data['test_set_flag'] == False][feat_cols].as_matrix()
            source_data_feats = self.source_data[feat_cols].as_matrix()
            target_data_feats = self.target_data[feat_cols].as_matrix()

            if train_mode=='source_and_target':
                if self.target_true_labels_available == 1:
                    trn_feats = np.concatenate((source_trn_feats, target_trn_feats))
                    trn_labels = np.concatenate((self.source_labels_train, self.target_labels_train))
                    test_feats = np.concatenate((source_test_feats, target_test_feats))
                    test_labels = np.concatenate((self.source_labels_test, self.target_labels_test))

                else:
                    print('Target labels not available, cannot train in this mode')
                    return 'Failed'

            elif train_mode=='source':
                trn_feats = source_trn_feats
                trn_labels = np.array(self.source_labels_train)
                test_feats = source_test_feats
                test_labels = np.array(self.source_labels_test)
            else:
                # raise NotImplemented
                return 'Not Implemented'
        else:
            # TODO ENCODINGS and ENCODINGS TSNE
            # raise NotImplemented
            return 'Not Implemented'

        # Shuffle and create create classifiers
        p = np.random.permutation(len(trn_feats))
        trn_feats_shuffle,trn_labels_shuffle=trn_feats[p],trn_labels[p]
        clf = ld_clf.Low_dim_classifier(trn_feats_shuffle, trn_labels_shuffle, methods=methods)
        clf.train(train_methods=methods)

        # Get predictions for source and data
        source_preds,source_cnf_mat=clf.get_performance(source_data_feats,self.source_labels,methods)
        self.embds_space_performance[key_prefix + '_source_data_cnf_matrix']=source_cnf_mat
        if self.target_true_labels_available:
            target_preds, target_cnf_mat = clf.get_performance(target_data_feats, self.target_labels, methods)
            self.embds_space_performance[key_prefix + '_target_data_cnf_matrix'] = target_cnf_mat
        else:
            target_preds = clf.predict(target_data_feats,methods)
        for key in source_preds.keys():
            self.source_data[key_prefix+'_'+key+'_pred'] =source_preds[key]
            self.target_data[key_prefix+'_'+key+'_pred'] =target_preds[key]

            self.embds_space_classifiers.append(key_prefix+'_'+key+'_pred')

        # Get predictions for test data
        source_test_preds, source_test_cnf_mats = clf.get_performance(source_test_feats,
                                                                            self.source_labels_test, methods)
        # Save performance matrices for test subset data for source and target
        self.embds_space_performance[key_prefix + '_source_test_cnf_matrix'] = source_test_cnf_mats
        if self.target_true_labels_available:
            target_test_preds, target_test_cnf_mats = clf.get_performance(target_test_feats,
                                                                          self.target_labels_test, methods)
            self.embds_space_performance[key_prefix + '_target_test_cnf_matrix'] = target_test_cnf_mats

        if self.inter_save:
            self.save_perforamance(self.serving_dir,suffix=self.save_suffix)
        return 'Success'

    def dim_red_autoencode(self,encode_dims=[128,64]):
        #self.source_encoded_reps,self.target_encoded_reps=autoencode_reps(self.source_embds_vec,self.target_embds_vec,encode_dims=encode_dims)
        raise NotImplemented

    def compute_tsne(self,perplexity=30, n_components=2, init='pca', n_iter=3500,save_data_viz=True,train_feat_space='embeds'):
        print('-----------------------------------------------------------------------')
        print('Computing TSNE Encodings')
        self.n_tsne_components=n_components
        if train_feat_space=='encoded':
            if not hasattr(self, 'source_encoded_reps'):
                self.dim_red_autoencode()
            concat_embs = np.concatenate((self.source_encoded_reps, self.target_encoded_reps))
        elif train_feat_space=='embeds':
            concat_embs=np.concatenate((self.source_embds_vec,self.target_embds_vec))
        else:
            raise NotImplemented

        tsne = TSNE(perplexity=perplexity, n_components=n_components, init=init, n_iter=n_iter,verbose=1)
        print('Computing Embeddings')
        tsne_feats = tsne.fit_transform(concat_embs)
        source_tsne=tsne_feats[:self.n_source,:]
        target_tsne=tsne_feats[self.n_source:,:]
        for idx in range(n_components):
            self.source_data[train_feat_space+'_tsne_'+str(idx)]=source_tsne[:,idx]
            self.target_data[train_feat_space+'_tsne_'+str(idx)]=target_tsne[:,idx]
            self.low_dim_feat_cols['tsne'].append(train_feat_space+'_tsne_'+str(idx))
        self.tsne_computed=1
        print('TSNE Encodings available')

        if self.inter_save:
            print('Saving TSNE for visualization')
            self.save_perforamance(self.serving_dir,suffix=self.save_suffix)

        print('-----------------------------------------------------------------------')

    def compute_pca(self,train_feat_space='embeds'):
        n_components=2
        if train_feat_space=='encoded':
            if not hasattr(self, 'source_encoded_reps'):
                self.dim_red_autoencode()
            concat_embs = np.concatenate((self.source_encoded_reps, self.target_encoded_reps))
        elif train_feat_space=='embeds':
            concat_embs=np.concatenate((self.source_embds_vec,self.target_embds_vec))
        else:
            raise NotImplemented

        pca = PCA()
        print('Computing PCA')
        pca_feats = pca.fit_transform(concat_embs)
        source_pca=pca_feats[:self.n_source,:]
        target_pca=pca_feats[self.n_source:,:]

        for idx in range(n_components):
            self.source_data[train_feat_space+'_pca_'+str(idx)]=source_pca[:,idx]
            self.target_data[train_feat_space+'_pca_'+str(idx)]=target_pca[:,idx]
            self.low_dim_feat_cols['pca'].append(train_feat_space+'_pca_'+str(idx))

        if self.inter_save:
            print('Saving PCA for visualization')
            self.save_perforamance(self.serving_dir,suffix=self.save_suffix)

    def source_to_target_label_prop(self,train_feat_space='embeds',kernel_param={'type':'rbf','gamma':20}):
        print('-----------------------------------------------------------------------')
        print('Propagating labels from source to target in {0} space'.format(train_feat_space))
        if train_feat_space == 'encoded':
            if not hasattr(self, 'source_encoded_reps'):
                self.dim_red_autoencode()
            concat_embs = np.concatenate((self.source_encoded_reps, self.target_encoded_reps))
        elif train_feat_space == 'embeds':
            concat_embs = np.concatenate((self.source_embds_vec,self.target_embds_vec))
        elif train_feat_space == 'embeds_tsne':
            if self.tsne_computed==0:
                self.compute_tsne()
            feat_cols = []
            for idx in range(self.n_tsne_components):
                feat_cols.append('embeds_tsne_' + str(idx))
            source_data_feats = self.source_data[feat_cols].as_matrix()
            target_data_feats = self.target_data[feat_cols].as_matrix()
            concat_embs = np.concatenate((source_data_feats,target_data_feats))
        else:
            raise NotImplemented
        unknown_labels=np.ones_like(self.target_labels )*-1
        label_prop_train_labels=np.concatenate((self.source_labels,unknown_labels))
        lp_model = LabelSpreading()
        lp_model.fit(concat_embs,label_prop_train_labels)
        transduction_labels = lp_model.transduction_
        label_distributions = lp_model.label_distributions_

        print(label_distributions[0:10,:])
        self.source_data[train_feat_space+'Space_prop_pred'] = transduction_labels[:self.n_source]
        self.target_data[train_feat_space + 'Space_prop_pred'] = transduction_labels[self.n_source:]
        # self.source_data[train_feat_space+'label_prop_groups'] = label_distributions[:self.n_source]
        # self.target_data[train_feat_space + 'label_prop_groups'] = label_distributions[self.n_source:]

        # self.embds_space_grouping.append(train_feat_space + 'label_prop_groups')
        # self.embds_space_classifiers.append(train_feat_space+'Space_prop_pred')
        if self.inter_save:
            print('Saving propagated labels')
            self.save_perforamance(self.serving_dir,suffix=self.save_suffix)

        print('Completed source to target label propagation in {0} space').format(train_feat_space)
        print('-----------------------------------------------------------------------')

    def get_samples_tolabel(self,dl_prob_label):
        dl_low_porb = samples_to_label.get_boundary_points(self.target_data[dl_prob_label])
        embed_space_disagree,embed_space_majority_lables = samples_to_label.get_disagreed_points(self.target_data[self.embds_space_classifiers].as_matrix())

        self.target_data['dl_low_prob_samples']=dl_low_porb
        self.target_data['embed_space_disagree_samples'] = embed_space_disagree
        self.source_data['dl_low_prob_samples'] = False
        self.source_data['embed_space_disagree_samples'] = False
        self.low_confidence_metric_cols.append(['dl_low_prob_samples','embed_space_disagree_samples'])
        self.target_data['voting_pred']=embed_space_majority_lables


        _,source_embeds_space_majority_lables = samples_to_label.get_disagreed_points(self.source_data[self.embds_space_classifiers].as_matrix())
        self.source_data['voting_pred'] = source_embeds_space_majority_lables

        self.embds_space_classifiers.append('voting_pred')
        if self.inter_save:
            self.save_perforamance(self.serving_dir,suffix=self.save_suffix)

    def save_perforamance(self,save_dir,class_key='all',print_flag=True,suffix=''):
        self.source_data.to_csv(save_dir+'/source_predictions_vis_'+suffix+'.csv',index=None,sep=',')
        self.target_data.to_csv(save_dir+'/target_predictions_vis_'+suffix+'.csv',index=None,sep=',')
        combined_data = pd.concat((self.source_data,self.target_data))
        combined_data['in_source_data_flag']=False
        combined_data['in_source_data_flag'].iloc[0:self.n_source]=True
        combined_data.to_csv(save_dir+'/visualization_data_'+suffix+'.csv',index=None,sep=',')

        for training_key in self.embds_space_performance.keys():
            for classifier_key in self.embds_space_performance[training_key]:
                if classifier_key==class_key or class_key=='all':
                    save_file_name=save_dir + '/' + training_key+'_'+classifier_key
                    print('Saving {0}'.format(save_file_name))
                    if print_flag==True:
                        print(self.embds_space_performance[training_key][classifier_key])
                    np.savetxt(save_file_name, self.embds_space_performance[training_key][classifier_key], '%f')

    def combine_output_files(self,old_file,new_file,save_dir,save_name):

        data_old = pd.read_csv(old_file)
        data_new = pd.read_csv(new_file)
        print(data_old.columns.tolist())
        print(data_new.columns.tolist())
        data_updated=data_new.copy()
        for column in data_old.columns.tolist():
            if column not in data_new.columns.tolist():
                data_updated[column] =data_old[column]

        print(data_updated.columns.tolist())
        # THis is hard-coded...modify later TODO
        pred_cols=[]
        for column in data_updated.columns:
            if ('pred' in column) and ('voting' not in column):
                pred_cols.append(column)
        print(pred_cols)
        embed_space_disagree, embed_space_majority_lables = samples_to_label.get_disagreed_points(data_updated[pred_cols].as_matrix())

        data_updated['voting_pred']=embed_space_majority_lables
        data_updated['embed_space_disagree_samples']=embed_space_disagree
        data_updated.to_csv(save_dir + '/'+save_name+'_conbined.csv', index=None, sep=',')






if __name__=='__main__':
    data_dir='/Users/karthikkappaganthu/Documents/online_learning/benckmarking/Performance_files/source_trained_model'
    source_embds_file=data_dir+'/source_data_files.txt_embeds.npy'
    target_embds_file=data_dir+'/target_data_files.txt_embeds.npy'

    source_predictions_file=data_dir+'/source_data_files.txt_predictions'
    target_predictions_file=data_dir+'/target_data_files.txt_predictions'
    ec = Embeds_classify(source_embds_file, target_embds_file, source_predictions_file, target_predictions_file,
                         embds_param={'type': 'convs', 'n_convs': 2,'standardize':False})

    ec.source_to_target_label_prop()

    # source_predictions_file = data_dir + '/source_predictions_vis_encoded_feats.csv'
    # target_predictions_file = data_dir + '/target_predictions_vis_encoded_feats.csv'
    # ec = Embeds_classify(source_embds_file, target_embds_file, source_predictions_file, target_predictions_file,
    #                      embds_param={'type': 'convs', 'n_convs': 2,'standardize':True},index_col=None,sep=',')
    # ec.n_tsne_components=2

    # ec.compute_tsne(train_feat_space='embeds')
    # ec.compute_tsne(train_feat_space='encoded')
    # ec.set_test_train_data()
    # ec.low_dim_classifier(train_feat_space='embeds',train_mode='source')
    # ec.low_dim_classifier(train_feat_space='embeds_tsne',train_mode='source')
    # ec.low_dim_classifier(train_feat_space='embeds',train_mode='source_and_target')
    # ec.low_dim_classifier(train_feat_space='embeds_tsne',train_mode='source_and_target')
    ec.save_perforamance(data_dir)