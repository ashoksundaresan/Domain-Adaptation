import numpy as np
import pandas as pd

def get_boundary_points(prob, n_classes=2,thresold=.25):
    if n_classes==2:
        return [abs(x-.5)<thresold for x in prob]
    else:
        raise NotImplemented


def get_disagreed_points(ensamble_classifications,class_labels=[0,1]):
    n_agree=[]
    majority_labels=[]
    n_classes =len(class_labels)
    n_classifiers=ensamble_classifications.shape[1]
    agree_threshold=.2
    if n_classes==2:
        for row in ensamble_classifications:
            n_c=[]
            for lab in class_labels:
                n_c.append(sum(row==lab))
            n_agree.append(abs(n_c[0]-n_c[1])*1./n_classifiers)
            majority_label=class_labels[np.argmax(n_c)]
            majority_labels.append(majority_label)

        ambigious_samples=[x <agree_threshold for x in n_agree]
        return ambigious_samples,majority_labels
    else:
        raise NotImplemented


if __name__=='__main__':
    data_dir = '/Users/karthikkappaganthu/Documents/online_learning/demo_testing_data/'
    target_pred_loc = data_dir + 'domain_adapt_output_files/target_predictions_vis_.csv'
    aa=pd.read_csv(target_pred_loc)
    cols=['dl_pred', 'embedsSpace_prop_pred', 'embedsSpace_sourceMode_RandomForestClassifier_pred',
     'embeds_tsneSpace_prop_pred', 'embeds_tsneSpace_sourceMode_RandomForestClassifier_pred']
    get_disagreed_points(aa[cols].as_matrix())