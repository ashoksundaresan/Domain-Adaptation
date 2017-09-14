from sklearn.model_selection import train_test_split
import pandas as pd
def mix_source_target(mix_pct,target_files,target_labels,source_files,source_labels):
    mix_train=[]
    mix_val=[]
    for pct in mix_pct:

        target_train_files_i,target_val_files_i,target_train_labels_i,target_val_labels_i=train_test_split(target_files,target_labels,train_size=pct[0]*1./100)
        source_train_files_i, source_val_files_i, source_train_labels_i, source_val_labels_i = train_test_split(
            source_files, source_labels, train_size=pct[1]*1./100)

        n_train_from_target_i=len(target_train_files_i)
        n_val_from_target_i = len(target_val_files_i)

        n_train_from_source_i = len(source_train_files_i)
        n_val_from_source_i = len(source_val_files_i)


        train_files_i=target_train_files_i+source_train_files_i
        train_labels_i = target_train_labels_i + source_train_labels_i
        train_from_flag_i=['target']*n_train_from_target_i+['source']*n_train_from_source_i

        val_files_i = target_val_files_i + source_val_files_i
        val_labels_i = target_val_labels_i + source_val_labels_i
        val_from_flag_i = ['target'] * n_val_from_target_i + ['source'] * n_val_from_source_i


        train_df_i=pd.DataFrame({'filename':train_files_i,'labels':train_labels_i,'from_flag':train_from_flag_i})
        val_df_i = pd.DataFrame({'filename': val_files_i, 'labels': val_labels_i, 'from_flag': val_from_flag_i})
        mix_train.append(train_df_i)
        mix_val.append(val_df_i)

    return  mix_train,mix_val
