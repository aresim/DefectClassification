import numpy as np
import pandas as pd
import tensorflow as tf
import pathlib
from PIL import Image
from os.path import join
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score
import seaborn as sn
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import math

import tools                        # load own package


cfg = {
    'name': 'Stoat',                                    # codename of the model - used for output path and filenames
    'train_from': 'Stoat_V59',                        # either: 'imagenet' or full model name such as 'Stoat_V4'
    'input_size': 299,
    'pred_dataset': 'kiel_clean_5fold_000_valid',  # which dataset to predict
    'col_cl_mapping': 'cl_15_w404',                  # which column mapping to use between real classes and target classes
    'col_class': 'cl_manual',                           # column name in data .csv with class information
    'col_image': 'filename' ,                            # column name in data .csv with filename of images (no path)
    'col_id': 'defect_id'
}

path = {
    'models': '/data/Simone/models/',                                # directory where models are saved

    'csv_train': '/data/Simone/split_by_wafer/train.csv',   # path to .csv with training data 
    'csv_valid': '/data/Simone/split_by_wafer/valid.csv',   # path to .csv with validation data
    'csv_test': '/data/Simone/split_by_wafer/test.csv',     # path to .csv with test data 
    'csv_predict': '/data/DefectDensity/ostsee/kiel/kiel_clean_5fold_000_valid.csv', # path to csv or excel with data frame to images
    'csv_meta': '/data/DefectDensity/ostsee/kiel_clean.xlsx',                            # path to csv or excel with metadata to enrich predictions
    'csv_context': '/data/DefectDensity/contextdata/kiel_contextdata.csv',
    
    'data_train':'/images/kiel/' ,                              # path to image directory
    'data_predict': '/images/kiel/',                            # path to image directory

    'arch_overview': '/data/yury/ybtools/models/model_architectures.xlsx', # path to excel with architecture overview
    'class_map': '/data/DefectDensity/DefectClassMap_Metal.xlsx'    # path to excel with mapping of classes between
}

hyper = {
    'n_epochs_max': 50,                # maximum number of epochs for training, only reached if early stopping doesn't trigger
    'lr_min': 1e-5,                     # minimum value for lr search
    'lr_max': 1e-0,                     # maximum value for lr search
    'n_lr_search': 100,                 # how many steps to take in the lr search space (exponential by default)
    'lr_decay': 0,                      # learing rate decay parameter, suggested to use 0 if lr reduction callback is used
    'patience_stop': 10,                # epochs without improvement to wait for early stopping. 20 is better for "final" models
    'patience_lr': 6,                   # epochs without improvement to wait until learning rate reduction is triggered, 2 reductions within patience_stop occur
    'lr_factor': 0.33,                  # factor by which learning rate is multiplied when lr reduction is triggered. good experience with factor 3 -> 2 reductions lead to 1 order of magnitude
    'save_best_epoch_only': True,       # flag if only best epoch is saved or all epochs, default True
    'dropout': False,                   # flag if Mustilidae dropout layer should be included or not. Not implemented for Beaver
    'n_threshold_steps': 50,            # number of steps for softmax threshold simulation (O(n)) - default 20 equals 5% steps
    'target_acc': 0.8,                  # target accuracy for volume simulation
    'keras_preprocessing': True,        # use Keras preprocessing function or rescaling instead
    'tensorboard': 'none',              # write output files for tensorboard. Possible values: 'minimal', 'full' and 'none' (or anything else)
    'remove_to_404': False,             # Flag if remove-class images are actually supposed to be removed from prediction analysis. Does not influence training where remove is always removed
    'meta_columns': ["tech","route","op","tool"],  # which columns to add from csv_meta to enriched predictions                
    'context_columns': ["x(wafer)[um]","y(wafer)[um]","x(die)[um]","y(die)[um]","die_x","die_y"]
}


# DERIVE SOME HYPER-PARAMETERS FROM ARCHITECTURE 
d = pd.read_excel(path['arch_overview'], sheet_name=0)
hyper['batch_size'] = d.loc[d['codename']==cfg['name'],'batch_size'].values[0]
hyper['threshold_layer'] = d.loc[d['codename']==cfg['name'],'threshold_layer'].values[0]
hyper['lr_fb'] = d.loc[d['codename']==cfg['name'],'default_lr_fb'].values[0] # default learning rate if lr search is not used for frozen base model
hyper['lr_pt'] = d.loc[d['codename']==cfg['name'],'default_lr_pt'].values[0] # default learning rate if lr search is not used for pretrained top model


cl_names = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','U']
n_classes = len(cl_names)

d_map = pd.read_excel(path['class_map'],dtype=str)
d_train = tools.helper.load_dataframe(path['csv_train'],cfg,d_map)
d_valid = tools.helper.load_dataframe(path['csv_valid'],cfg,d_map)
d_test = tools.helper.load_dataframe(path['csv_test'],cfg,d_map)


# PREPARE TRAIN AND VALID DATA SUCH THAT IT CAN BE FED TO DATA-GENERATOR
train_labels,train_filenames = tools.helper.prepare_data_only_images(d_train,hyper)
valid_labels,valid_filenames = tools.helper.prepare_data_only_images(d_valid,hyper)
test_labels,test_filenames = tools.helper.prepare_data_only_images(d_test,hyper)


# CREATE TRAIN AND VALID DATA-GENERATOR
train_datagen = tools.helper.MyImageDataGenerator(img_files=train_filenames,labels=train_labels, batch_size=hyper['batch_size'] , n_classes=n_classes)
val_datagen = tools.helper.MyImageDataGenerator(img_files=valid_filenames,labels=valid_labels, batch_size=hyper['batch_size'], n_classes=n_classes,shuffle=False)
test_datagen = tools.helper.MyImageDataGenerator(img_files=test_filenames,labels=test_labels, batch_size=hyper['batch_size'],n_classes=n_classes,shuffle=False)


d_cl = tools.helper.summarize_train_valid_test_data(d_train,d_valid,d_test,cl_names)
# dictionary with weight for loss function adjustment
cl_weight_dict = dict(zip(d_cl['cl_id'], d_cl['inv_ratio_adj_train'])) 

# LOAD PRE-TRAINED MODEL
m = tf.keras.models.load_model('/data/Simone/models/Stoat_V55.hdf5')


# RESIZE LAST LAYER
# creates new model without last layer
m_base = tf.keras.models.Model(inputs=m.input, outputs=m.layers[-2].output) 
# add classification layer    
classification_layer = tf.keras.layers.Dense(n_classes, activation='softmax', name='classification')(m_base.output)
m_images = tf.keras.models.Model(inputs=m_base.input, outputs=classification_layer)



m_images.compile(optimizer=tf.keras.optimizers.Adam(lr=hyper['lr_pt'], decay=hyper['lr_decay']),
              loss=WeightedCategoricalCrossentropy(cost_matrix),
              metrics=['accuracy'])


# Derive some hyper-parameters for training
hyper['n_steps_train'] = int(np.ceil(train_datagen.__len__()/train_datagen.batch_size))
hyper['n_steps_valid'] = int(np.ceil(val_datagen.__len__()/val_datagen.batch_size))


# Adding callbacks
filepath = "path_to_directory_where_to_save_best_model"
checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath, monitor='val_acc', verbose=1,save_weights_only=False,save_best_only=True, mode='max')
check_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss',min_delta=0.0001,patience=hyper['patience_lr'],verbose=1)
check_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss',min_delta=0.0001,patience=hyper['patience_stop'],verbose=1,restore_best_weights=True)
callbacks_list = [checkpoint,check_lr,check_stop]


model_trained = m_images.fit_generator(train_datagen, 
                           steps_per_epoch=hyper['n_steps_train'], 
                           epochs=hyper['n_epochs_max'], 
                           class_weight = cl_weight_dict,
                           callbacks = callbacks_list, 
                           validation_data= val_datagen, 


# Plot training history
tools.helper.model_history_analysis(model_trained, '/data/Simone/only_images/plots/model_history.png', n_classes)


best_model = tf.keras.models.load_model(filepath)
best_model.compile(optimizer=tf.keras.optimizers.Adam(lr=hyper['lr_pt'],decay=hyper['lr_decay']),loss='categorical_crossentropy',metrics=['accuracy'])


probabilities = m_images.predict_generator(test_datagen, steps=len(test_datagen), verbose=1)
predictions = np.argmax(probabilities, axis=1)


# DERIVE NEW PREDICTIONS BY WEIGHTING OLD PREDICTIONS WITH BAYESIAN PRIORS
new_probabilities = np.ones(shape=probabilities.shape)
bayesian_priors = np.load('/data/Simone/predictions/bayesian_priors_1500.npy')
weights_for_priors = np.load('/data/Simone/predictions/weights_for_priors.npy')

for i in range(new_probabilities.shape[0]):
  
    weighted_bayesian_priors = np.multiply(bayesian_priors[i],weights_for_priors)
    weighted_bayesian_priors = weighted_bayesian_priors/np.sum(weighted_bayesian_priors)

    new_probabilities[i] = np.multiply(probabilities[i],weighted_bayesian_priors)
    N = np.sum(new_probabilities[i])
    new_probabilities[i] = new_probabilities[i]/N
    
new_predictions = np.argmax(new_probabilities, axis=1)


# DERIVE TRUE LABELS
test_values = np.array([t for t in test_labels.values()])
test_values = test_values[0:probabilities.shape[0]]

# Plot confusion matrix
tools.helper.plot_confusion_matrix(test_values , predictions , n_classes)


# Evaluate classification score
report = classification_report(test_values, new_predictions)
report_df = tools.helper.report_to_df(report,'path_to_directory_where_to_save_classification_score')

# Turns keras probabilities into a properly named pandas data frame
classes = []

for i in range(predictions.shape[0]):
    classes.append(tools.helper.assign_class(test_values[i]))

d_softmax_2 = tools.helper.prepare_base_table(new_probabilities, cl_names,classes,test_filenames,'path_to_directory_where_to_save_softmax_vector2')
d_softmax = tools.helper.prepare_base_table(probabilities, cl_names,classes,test_filenames,'path_to_directory_where_to_save_softmax_vector')


# Softmax threshold analysis 
hyper['threshold_range'] = np.array(range(hyper['n_threshold_steps']))/hyper['n_threshold_steps']
cfg['n_img'] = d_softmax.shape[0]
d_thresh, d_thresh_cl = tools.helper.simulate_softmax_thresholds(d_softmax,cfg,hyper)
d_thresh_2, d_thresh_cl_2 = tools.helper.simulate_softmax_thresholds(d_softmax_2,cfg,hyper)


# Plot softmax threshold simulation: comparison with/without priors
tools.helper.plot_accuracy_and_volume_vs_softmax_2(d_thresh,d_thresh_2,'path_to_directory_where_to_save_plot_by_class',cfg)


# Plot softmax threshold simulation by class: comparison with/without priors
tools.helper.plot_accuracy_and_volume_vs_softmax_by_class_2(d_thresh_cl,d_thresh_cl_2, 'path_to_directory_where_to_save_plot_by_class', cfg)



