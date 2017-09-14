from flask import Flask,redirect,url_for,request,jsonify,render_template
from include import four_group_plot
from threading import Lock
LOCK = Lock()
app = Flask(__name__)


PLOT_INPUT_PARAMAS={'filename':'/Users/karthikkappaganthu/Documents/online_learning/demo_testing_data/domain_adapt_output_files/visualization_data_.csv', \
                    'x_col':'embeds_tsne_0', 'y_col': 'embeds_tsne_1',\
                    'group1_col': 'in_source_data_flag', \
                    'group2_col': 'true_label', \
                    'group3_col': 'dl_pred', \
                    'show_only_col': 'dl_low_prob_samples'}

# LD_INPUT_PARAMAS={'source_data_filename':
#                   'target_data_filename':
#                    'source_embed_filename':
#                     'target_embed_filename':
#                     'serving_dir':
#                     'low_dim_feats':
#                     'semi_supervised_methods':
#                     'dl_feature_leaning':'Ran'}
# @app.route('/set_low_dim_param',methods=['POST','GET'])
# def set_low_dim_param():
#     if request.method == 'POST':
#



@app.route('/set_plot_param',methods=['POST','GET'])
def set_plot_param():
    if request.method=='POST':
        for key in PLOT_INPUT_PARAMAS.keys():
            val = request.form[key]
            if val:
                PLOT_INPUT_PARAMAS[key]=val

        status,val='OK',PLOT_INPUT_PARAMAS
        return jsonify(status=status,val=val)


@app.route('/show_plot',methods=['POST','GET'])
def show_plot():
    if request.method=='POST':
        four_group_plot.four_group_plot(PLOT_INPUT_PARAMAS['filename'], \
        PLOT_INPUT_PARAMAS['x_col'],PLOT_INPUT_PARAMAS['y_col'],\
        PLOT_INPUT_PARAMAS['group1_col'], \
        PLOT_INPUT_PARAMAS['group2_col'], \
        PLOT_INPUT_PARAMAS['group3_col'], \
        PLOT_INPUT_PARAMAS['show_only_col'],LOCK)

        return 'Plot opened in a new window'
        # return jsonify(plotting_parameters=PLOT_INPUT_PARAMAS)



if __name__ == '__main__':
   app.run('0.0.0.0',debug=True)