import copy
def modify_solver_net_path(base_solver_path,net_path,save_path):
    with open(base_solver_path, 'r') as f:
        lines = f.readlines()
        for idx, line in enumerate(lines):
            if 'net' in line:
                net_idx = idx

    net_line_original=lines[net_idx].split(':')
    replace_line=':'.join([net_line_original[0],'\"'+ net_path+"\"\n"])
    lines[net_idx]=replace_line
    with open(save_path, 'w') as f:
        for line in lines:
            f.write(line)

def modify_arch_proto_data_paths(base_arch_path,mean_path,source_paths,save_path):
    if not isinstance(source_paths,list):
        source_paths=[source_paths]

    with open(base_arch_path, 'r') as f:
        lines = f.readlines()
        mean_idx = []
        source_idx=[]
        for idx, line in enumerate(lines):
            if 'mean_file' in line:
                mean_idx.append(idx)
            if 'source' in line:
                source_idx.append(idx)
    for idx in mean_idx:
        net_line_original = lines[idx].split(':')
        replace_line = ':'.join([net_line_original[0],'\"'+ mean_path+"\"\n"])
        lines[idx] = replace_line

    for idx,line_idx in enumerate(source_idx):
        net_line_original = lines[line_idx].split(':')
        replace_line = ':'.join([net_line_original[0],'\"'+ source_paths[idx]+"\"\n"])
        lines[line_idx] = replace_line

    with open(save_path, 'w') as f:
        for line in lines:
            f.write(line)


if __name__=='__main__':
    base_solver_path = '/Users/karthikkappaganthu/Documents/online_learning/model_files/base_models/sns_model2/sns_solver_sq.prototxt'
    net_path = "../../model_files/sqNet_domain_adap/mix_pct_h_10_t_100/model_protofiles/train.prototxt"
    save_path='test.proto'
    modify_solver_net_path(base_solver_path, net_path, save_path)


    base_arch_path='/Users/karthikkappaganthu/Documents/online_learning/model_files/base_models/sns_model2/sns_train_val_sq.prototxt'
    mean_path = "../../model_files/sqNet_domain_adap/mix_pct_h_10_t_100/dbs/mean.binaryproto"
    source_paths = ["../../model_files/sqNet_domain_adap/mix_pct_h_10_t_100/dbs/train_lmdb","../../model_files/sqNet_domain_adap/mix_pct_h_10_t_100/dbs/test_lmdb"]
    save_path = "test1.proto"

    modify_arch_proto_data_paths(base_arch_path,mean_path,source_paths,save_path)









