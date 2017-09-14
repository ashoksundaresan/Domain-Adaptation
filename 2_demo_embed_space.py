import embd_space_learn

#### Initialize the learning
data_dir = '/Users/karthikkappaganthu/Documents/online_learning/demo_testing_data/'
serving_dir=data_dir+'temp/'

# source_pred_loc = data_dir+'source_data.csv'
# target_pred_loc = data_dir+'target_data.csv'
# Precomputed
source_pred_loc = data_dir+'/source_predictions_tsne_precom.csv'
target_pred_loc = data_dir+'/target_predictions_tnse_precom.csv'

source_embeds_loc = data_dir+'source_data_embeds.npy'
target_embeds_loc = data_dir+'target_data_embeds.npy'
embds_param={'type': 'convs', 'n_convs': 2,'standardize':False}
true_labels_col='true_label'
em_learn=embd_space_learn.Embeds_classify(source_embeds_loc,target_embeds_loc,source_pred_loc,target_pred_loc,embds_param=embds_param,true_labels_col=true_labels_col,serving_dir=serving_dir)

#### Compute Low dimentional features
# em_learn.compute_pca(train_feat_space='embeds')
# em_learn.compute_tsne(train_feat_space='embeds')

#### Compute Labels

### Embedding space
## Label propagation
em_learn.source_to_target_label_prop()

## lowdim classifier
em_learn.low_dim_classifier(train_feat_space='embeds',train_mode='source')


### TSNE Space
## Label propagation
em_learn.source_to_target_label_prop(train_feat_space='embeds_tsne')

## lowdim classifier
em_learn.low_dim_classifier(train_feat_space='embeds_tsne',train_mode='source')



### Display
em_learn.n_tsne_components=2
em_learn.tsne_computed=1
em_learn.embds_space_classifiers=['dl_pred','embedsSpace_prop_pred', 'embedsSpace_sourceMode_RandomForestClassifier_pred', 'embeds_tsneSpace_prop_pred', 'embeds_tsneSpace_sourceMode_RandomForestClassifier_pred']
em_learn.low_dim_feat_cols={'tsne': ['embeds_tsne_0', 'embeds_tsne_1'], 'pca': ['embeds_pca_0', 'embeds_pca_1']}
em_learn.get_samples_tolabel(dl_pred_label='dl_prob')


