from keras.layers import Input, Dense
from keras.models import Model
import keras.callbacks
import numpy as np
from keras import regularizers

def autoencode_reps(reps_to_learn,reps_to_process=np.array([]),encode_dims=[128,64,32],params={}):
    ### Set parameters
    if 'optimizer' not in params.keys():
        optimizer = 'adadelta'
    else:
        optimizer = params['optimizer']

    if 'loss' not in params.keys():
        loss = 'mean_squared_error'#'binary_crossentropy'
    else:
        loss=params['loss']

    if 'epochs' not in params.keys():
        epochs = 100
    else:
        epochs=params['epochs']

    if 'batch_size' not in params.keys():
        batch_size = 256
    else:
        batch_size=params['batch_size']

    if 'shuffle' not in params.keys():
        shuffle = True
    else:
        shuffle=params['shuffle']

    if 'regularizer_val' not in params.keys():
        regularizer_val=10e-5
    else:
        regularizer_val=params['regularizer_val']

    if 'regularizer_type' not in params.keys():
        regularizer_type='l1'
    else:
        regularizer_type=params['regularizer_type']

    ### set regularizer
    if regularizer_type=='l1':
        regularize=regularizers.l1(regularizer_val)
    elif regularizer_type=='None':
        regularize=None
    else:
        raise NotImplemented


    #### Representations to process
    if reps_to_process.any():
        validation_data = (reps_to_process, reps_to_process)
    else:
        validation_data = ()

    #### Create the autoencoder model
    n_inputs,n_reps_dim=reps_to_learn.shape
    input_rep=Input(shape=(n_reps_dim,))
    # encoder definition
    for idx, dim in enumerate(encode_dims):
        if idx == 0:
            encoded = Dense(dim, activation='relu')(input_rep)
        else:
            if regularize:
                encoded = Dense(dim, activation='relu',activity_regularizer=regularize)(encoded)
            else:
                encoded = Dense(dim, activation='relu')(encoded)
    #decoder definition
    for idx, dim in enumerate(reversed(encode_dims)):
        if idx == 0:
            pass
        elif idx == 1:
            decoded = Dense(dim, activation='relu')(encoded)
        else:
            decoded = Dense(dim, activation='relu')(decoded)
    decoded = Dense(n_reps_dim, activation='sigmoid')(decoded)

    # encoded = Dense(128, activation='relu')(input_rep)
    # encoded = Dense(64, activation='relu')(encoded)
    # encoded = Dense(32, activation='relu')(encoded)
    #
    # decoded = Dense(64, activation='relu')(encoded)
    # decoded = Dense(128, activation='relu')(decoded)
    # decoded = Dense(n_reps_dim, activation='sigmoid')(decoded)

    autoencoder = Model(input_rep, decoded)
    encoder = Model(input_rep, encoded)  # This is to extrat the low dim em

    ### Display encoder properties
    print('Created Autoencoder')
    for layer in autoencoder.layers:
        print('Layer_name: '+layer.name,'  Dimensions:'+str(layer.input_shape))
    # Creat callbacks (for logging and display) #TBD
    tbCallback = keras.callbacks.TensorBoard(log_dir='./log_dir', histogram_freq=0, write_graph=True, write_images=True)
    tbCallback.set_model(autoencoder)

    #### Compile and learn autoencoder
    print('Learning autoencoder')


    autoencoder.compile(optimizer=optimizer, loss=loss)
    autoencoder.fit(reps_to_learn, reps_to_learn,
                    epochs=epochs,
                    batch_size=batch_size,
                    shuffle=shuffle,
                    validation_data=validation_data)


    ### Compute encoded dimensions and display
    low_dim_embds_learn=encoder.predict(reps_to_learn)
    if reps_to_process.any():
        low_dim_embds_processed_reps=encoder.predict(reps_to_process)
        return low_dim_embds_learn,low_dim_embds_processed_reps
    else:
        return low_dim_embds_learn


if __name__=='__main__':
    data_dir = '/Users/karthikkappaganthu/Documents/online_learning/benckmarking/Performance_files/source_trained_model'
    source_embds_file = data_dir + '/source_data_files.txt_embeds.npy'
    source_embds = np.load(source_embds_file)
    source_feats = []
    for row in source_embds:
        feat = []
        for i_c in range(2):
            feat = np.concatenate((feat, row[i_c, :].reshape(1, -1)[0]))
        source_feats.append(feat)
    source_embds_vec = np.array(source_feats)
    print(source_embds_vec.shape)
    a=autoencode_reps(source_embds_vec)
    print(a.shape)







