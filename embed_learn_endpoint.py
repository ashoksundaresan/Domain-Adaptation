from flask import Flask,request,jsonify
import os
import embd_space_learn
app = Flask(__name__)

@app.route('/embed_space_learn',methods=['POST','GET'])
def compute_embed_space():
    if request.method == 'POST':
        data_dir = request.form['data_dir']
        compute_tsne_flag=request.form['compute_tsne_flag']
    else:
        data_dir=''
        compute_tsne_flag=''

    if data_dir=='':
        data_dir='../domain_adaptation_demo_data'
        print('Using data in {0}'.format(data_dir))
    if compute_tsne_flag=='':
        compute_tsne_flag=False

    serving_dir = data_dir + '/embed_space_out/'
    os.system('mkdir '+serving_dir)

    # Input Data sources
    source_pred_loc = data_dir + '/embeddings_out/source_data.csv'
    target_pred_loc = data_dir + '/embeddings_out/target_data.csv'

    source_embeds_loc = data_dir + '/embeddings_out/source_data_embeds.npy'
    target_embeds_loc = data_dir + '/embeddings_out/target_data_embeds.npy'
    embds_param = {'type': 'convs', 'n_convs': 2, 'standardize': False}

    true_labels_col = 'true_label'
    em_learn = embd_space_learn.Embeds_classify(source_embeds_loc, target_embeds_loc, source_pred_loc, target_pred_loc,
                                                embds_param=embds_param, true_labels_col=true_labels_col,
                                                serving_dir=serving_dir)

    #### Compute Low dimentional features
    print('Computing PCA')
    em_learn.compute_pca(train_feat_space='embeds')

    ### Embedding space
    ## Label propagation
    print('Propagating labels: method {0} of {1}'.format(2,4))
    em_learn.source_to_target_label_prop()

    ## lowdim classifier
    print('Propagating labels: method {0} of {1}'.format(3, 4))
    em_learn.low_dim_classifier(train_feat_space='embeds', train_mode='source',methods=['RandomForestClassifier','ExtraTreesClassifier','MLPClassifier'])

    if compute_tsne_flag:
        ### TSNE BASED, Skipping for demo
        print('Computing TSNE')
        em_learn.compute_tsne(train_feat_space='embeds')

        ### TSNE Space
        ## Label propagation
        print('Propagating labels: method {0} of {1}', format(4, 5))
        em_learn.source_to_target_label_prop(train_feat_space='embeds_tsne')

        ## lowdim classifier
        print('Propagating labels: method {0} of {1}', format(5, 5))
        em_learn.low_dim_classifier(train_feat_space='embeds_tsne', train_mode='source')

    ### Display
    print('Computing low-confidence samples to request labels')
    em_learn.get_samples_tolabel(dl_prob_label='dl_prob')

    output_dict={'output_dir':data_dir + '/embeddings_out'}

    return jsonify(output_dict)

### A separate call for tsne--because it is slower
@app.route('/embed_space_learn_tsne',methods=['POST','GET'])
def compute_embed_space_tsne():
    if request.method == 'POST':
        data_dir = request.form['data_dir']
    else:
        data_dir=''

    if data_dir=='':
        data_dir='../domain_adaptation_demo_data/'
        print('Using data in {0}'.format(data_dir))
    serving_dir = data_dir + '/embed_space_out/'
    # Input Data sources
    source_pred_loc = data_dir + '/embed_space_out/source_predictions_vis_.csv'
    target_pred_loc = data_dir + '/embed_space_out/target_predictions_vis_.csv'

    source_embeds_loc = data_dir + '/embeddings_out/source_data_embeds.npy'
    target_embeds_loc = data_dir + '/embeddings_out/target_data_embeds.npy'
    embds_param = {'type': 'convs', 'n_convs': 2, 'standardize': False}

    true_labels_col = 'true_label'
    em_learn = embd_space_learn.Embeds_classify(source_embeds_loc, target_embeds_loc, source_pred_loc, target_pred_loc,
                                                embds_param=embds_param, true_labels_col=true_labels_col,
                                                serving_dir=serving_dir,save_suffix='tsne_')

    print('Computing TSNE')
    em_learn.compute_tsne(train_feat_space='embeds')

    ### TSNE Space
    ## Label propagation
    print('Propagating labels: method {0} of {1}', format(4, 5))
    em_learn.source_to_target_label_prop(train_feat_space='embeds_tsne')

    ## lowdim classifier
    print('Propagating labels: method {0} of {1}', format(5, 5))
    em_learn.low_dim_classifier(train_feat_space='embeds_tsne', train_mode='source')

    ### Display
    print('Computing low-confidence samples to request labels')
    em_learn.get_samples_tolabel(dl_prd_label='dl_prob')

    em_learn.combine_output_files(data_dir + '/embed_space_out/source_predictions_vis_.csv', data_dir + '/embed_space_out/source_predictions_vis_tsne_.csv')

if __name__ == '__main__':
   app.run('0.0.0.0',port=8080,debug=True)