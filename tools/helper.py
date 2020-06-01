import numpy as np
import pandas as pd
import tensorflow as tf
import pathlib
from PIL import Image
from os.path import join
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import seaborn as sn
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator 
import math
import random


class DataGenerator(tf.keras.utils.Sequence):
    """Generates data for Keras."""
    def __init__(self, img_files, context_info, labels, batch_size=32, dim=(299,299), n_channels=3,
                 n_classes=16, shuffle=True):
        """Initialization.
        
        Args:
            img_files: A list of path to image files.
            context_info: A dictionary of corresponding context variables.
            labels: A dictionary of corresponding labels.
        """
        self.img_files = img_files
        self.context_info = context_info
        self.labels = labels
        self.batch_size = batch_size
        self.dim = dim
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        """Denotes the number of batches per epoch."""
        return int(np.floor(len(self.img_files) / self.batch_size))

    def __getitem__(self, index):
        """Generate one batch of data."""
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        img_files_temp = [self.img_files[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(img_files_temp)

        return X, y

    def on_epoch_end(self):
        """Updates indexes after each epoch."""
        self.indexes = np.arange(len(self.img_files))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, img_files_temp):
        """Generates data containing batch_size samples."""

        X_img = []
        X_context = []
        y = np.empty((self.batch_size), dtype=int)

        # Generate data
        for i , img_file in enumerate(img_files_temp):
            # Read image
            img = Image.open(join('/images/kiel/',img_file)) 
            img = img.convert(mode='RGB') #convert to 3-channels
            if self.shuffle:
                img = img.rotate(random.uniform(-5,5))
            # Resize image
            im = np.array(img.resize(size=self.dim) , dtype=np.float32)
            # Rescale image
            im = im/255.0
            
            # Normalization
            #for ch in range(self.n_channels):
                #img[:, :, ch] = (img[:, :, ch] - self.ave[ch])/self.std[ch]

            X_img.append(im)
            X_context.append(self.context_info[img_file])
            y[i] = self.labels[img_file]
        X = [np.array(X_img), np.array(X_context)]
        return X, tf.keras.utils.to_categorical(y, num_classes=self.n_classes)
        
        
        
class MyImageDataGenerator(tf.keras.utils.Sequence):
    """Generates data for Keras."""
    def __init__(self, img_files, labels, batch_size=32, dim=(299,299), n_channels=3,
                 n_classes=16, shuffle=True):
        """Initialization.
        Args:
            img_files: A list of path to image files.
            labels: A dictionary of corresponding labels.
        """
        self.img_files = img_files
        self.labels = labels
        self.batch_size = batch_size
        self.dim = dim
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        """Denotes the number of batches per epoch."""
        return int(np.floor(len(self.img_files) / self.batch_size))

    def __getitem__(self, index):
        """Generate one batch of data."""
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        img_files_temp = [self.img_files[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(img_files_temp)

        return X, y

    def on_epoch_end(self):
        """Updates indexes after each epoch."""
        self.indexes = np.arange(len(self.img_files))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, img_files_temp):
        """Generates data containing batch_size samples."""

        X_img = []
        y = np.empty((self.batch_size), dtype=int)

        # Generate data
        for i , img_file in enumerate(img_files_temp):
            # Read image
            img = Image.open(join('/images/kiel/',img_file)) 
            #img = Image.open(join('/images/puttgraden/',img_file)) 
            img = img.convert(mode='RGB') #convert to 3-channels
            if self.shuffle:
                img = img.rotate(random.uniform(-5,5))
            # Resize image
            im = np.array(img.resize(size=self.dim) , dtype=np.float32)
            # Rescale image
            im = im/255.0
            
            # Normalization
            #for ch in range(self.n_channels):
                #img[:, :, ch] = (img[:, :, ch] - self.ave[ch])/self.std[ch]

            X_img.append(im)
            y[i] = self.labels[img_file]
            
        X = np.array(X_img)
        
        return X, tf.keras.utils.to_categorical(y, num_classes=self.n_classes)


class TryDataGenerator(tf.keras.utils.Sequence):
    """Generates data for Keras."""
    def __init__(self, img_files, labels, batch_size=32, dim=(299,299), n_channels=3,
                 n_classes=16, shuffle=True):
        """Initialization.
        Args:
            img_files: A list of path to image files.
            labels: A dictionary of corresponding labels.
        """
        self.img_files = img_files
        self.labels = labels
        self.batch_size = batch_size
        self.dim = dim
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()
        

    def __len__(self):
        """Denotes the number of batches per epoch."""
        return int(np.floor(len(self.img_files) / self.batch_size))

    def __getitem__(self, index):
        """Generate one batch of data."""
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        img_files_temp = [self.img_files[k] for k in indexes]

        # Generate data
        X, y, W = self.__data_generation(img_files_temp)

        return X, y, W

    def on_epoch_end(self):
        """Updates indexes after each epoch."""
        self.indexes = np.arange(len(self.img_files))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
            
    def __data_generation(self, img_files_temp):
        """Generates data containing batch_size samples."""

        X_img = []
        y = np.empty((self.batch_size), dtype=int)
        W = np.empty((self.batch_size,self.n_classes), dtype=float)

        # Generate data
        for i , img_file in enumerate(img_files_temp):
            # Read image
            img = Image.open(join('/images/kiel/',img_file)) 
            #img = Image.open(join('/images/puttgraden/',img_file)) 
            img = img.convert(mode='RGB') #convert to 3-channels
            if self.shuffle:
                img = img.rotate(random.uniform(-5,5))
            # Resize image
            im = np.array(img.resize(size=self.dim) , dtype=np.float32)
            # Rescale image
            im = im/255.0
            
            # Normalization
            #for ch in range(self.n_channels):
                #img[:, :, ch] = (img[:, :, ch] - self.ave[ch])/self.std[ch]

            X_img.append(im)
            y[i] = self.labels[img_file]
            W[i] = np.ones(self.n_classes)
            
        X = np.array(X_img)
        #X = np.reshape(X,newshape=(X.shape[0],X.shape[1],X.shape[2],3))
        Y = tf.keras.utils.to_categorical(y, num_classes=self.n_classes)
        #Y = np.reshape(Y,newshape=(Y.shape[0],Y.shape[1],1))
        
        return X,Y,W        
            
            
def check_gpu():
    ''' checks GPU availability and prints warning if it is not '''
    if tf.test.is_gpu_available():
        print('GPU device id: ', tf.test.gpu_device_name())
        print('Built with CUDA: ', tf.test.is_built_with_cuda())
    else:
        print('WARNING: GPU not available!')


def assign_label(cl_target):
    
    if cl_target == 'A':
        new_label = 0
    if cl_target == 'B':     
        new_label = 1
    if cl_target == 'C':
        new_label = 2 
    if cl_target == 'D':
        new_label = 3
    if cl_target == 'E':
        new_label = 4        
    if cl_target == 'F':
        new_label = 5        
    if cl_target == 'G':
        new_label = 6        
    if cl_target == 'H':
        new_label = 7
    if cl_target == 'I':
        new_label = 8  
    if cl_target == 'J':
        new_label = 9        
    if cl_target == 'K':
        new_label = 10                
    if cl_target == 'L':
        new_label = 11       
    if cl_target == 'M':
        new_label = 12       
    if cl_target == 'N':
        new_label = 13
    if cl_target == 'O':
        new_label = 14       
    if cl_target == 'U':
        new_label = 15
        
    return new_label
    
    
def assign_class(label):

    if label == 0:
        new_class = 'A'
    if label == 1:     
        new_class = 'B'
    if label == 2:
        new_class = 'C'
    if label == 3:
        new_class = 'D'
    if label == 4:
        new_class = 'E'        
    if label == 5:
        new_class = 'F'       
    if label == 6:
        new_class = 'G'        
    if label == 7:
        new_class = 'H'
    if label == 8:
        new_class = 'I' 
    if label == 9:
        new_class = 'J'        
    if label == 10:
        new_class = 'K'                
    if label == 11:
        new_class = 'L'       
    if label == 12:
        new_class = 'M'       
    if label == 13:
        new_class = 'N'
    if label == 14:
        new_class = 'O'      
    if label == 15:
        new_class = 'U'
        
    return new_class    
    
    
def report_to_df(report , save_name):
    report = [x.split(' ') for x in report.split('\n')]
    header = ['Class Name']+[x for x in report[0] if x!='']
    values = []
    for row in report[1:-5]:
        row = [value for value in row if value!='']
        if row!=[]:
            values.append(row)
    df = pd.DataFrame(data = values, columns = header)
    df.to_excel(save_name, index=False)
    
    return df
    
def plot_confusion_matrix(val,predictions,n_classes=15):
    
    cm = confusion_matrix(val, predictions)
    df_cm = pd.DataFrame(cm , range(n_classes) , range(n_classes))
    plt.figure(figsize=(10,10))
    sn.set(font_scale = 1.0)
    sn.heatmap(df_cm, annot=True)
    plt.savefig(path_plot, bbox_inches='tight')
    plt.show()
    
    
def load_dataframe(filename , cfg , d_map):

    d = pd.read_csv(filename, dtype=str)
    
    # create new defect_id column
    #ids = []
    #for i in range(len(d)):
        #ids.append(d['defect'][i].split('/')[3])
    #d[cfg['col_id']] = ids 
    #d[cfg['col_id']] = d[cfg['col_id']].astype(str) # convert to str
    
    # create dictionary between manual classification and target classification
    cl_dict = dict(zip(d_map[cfg['col_class']], d_map[cfg['col_cl_mapping']]))
    
    # create new target class column based on mapping
    d['cl_target'] = list(map(cl_dict.get, d[cfg['col_class']]))
    d['cl_target'] = d['cl_target'].astype(str)
    
    # filter out 'remove' class
    if 'remove' in cl_dict.values():   
        #n_pre_filter = d.shape[0]
        d = d[d['cl_target'] != 'remove']
        #n_removed = n_pre_filter - d.shape[0]
        #print('class "remove" was found, removed {} images from dataset'.format(n_removed))
    
    return d

def cart2pol(x, y):

    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    
    return rho, phi

def get_mean_and_std_context_features(d, hyper):

    arr_0 = np.array(d[hyper['context_columns'][0]], dtype=float)
    arr_1 = np.array(d[hyper['context_columns'][1]], dtype=float)
    #arr_2 = np.array(d[hyper['context_columns'][2]], dtype=float)
    #arr_3 = np.array(d[hyper['context_columns'][3]], dtype=float)
    #arr_4 = np.array(d[hyper['context_columns'][4]], dtype=float)
    #arr_5 = np.array(d[hyper['context_columns'][5]], dtype=float)
    
    m_0 , std_0 = np.mean(arr_0) , np.std(arr_0)
    m_1 , std_1 = np.mean(arr_1) , np.std(arr_1)
    #m_2 , std_2 = np.mean(arr_2) , np.std(arr_2)
    #m_3 , std_3 = np.mean(arr_3) , np.std(arr_3)
    #m_4 , std_4 = np.mean(arr_4) , np.std(arr_4)
    #m_5 , std_5 = np.mean(arr_5) , np.std(arr_5)
    
    return m_0,std_0,m_1,std_1 #,m_2,std_2,m_3,std_3,m_4,std_4,m_5,std_5
    
def get_die_size_by_route_dict(d, routes, hyper):
    # Create a dict 'route': (max_x_die,max_y_die)
    sizes_die = {}

    for route in routes:
        
        d_red = d[d['route'] == route]
        max_x_die = math.ceil(np.max(d_red[hyper['context_columns'][2]].astype(float)))
        max_y_die = math.ceil(np.max(d_red[hyper['context_columns'][3]].astype(float)))
        sizes_die[route] = (max_x_die,max_y_die)
        
    return sizes_die
   
def prepare_data(d, sizes_die, hyper, polar):

    context = {}
    labels = {}
    filenames = [] 
    
    m_0,std_0,m_1,std_1 = get_mean_and_std_context_features(d, hyper)

    for i in range(len(d)):
        img_file = d['filename_000'][i]
        route = d['route'][i]
        max_x_die, max_y_die = sizes_die[route][0], sizes_die[route][1]
        filenames.append(img_file)
        if polar:
            rho,phi = cart2pol(float(d[hyper['context_columns'][0]][i]),float(d[hyper['context_columns'][1]][i])) 
            rho = rho/50000
        else:
            x_wafer = (float(d[hyper['context_columns'][0]][i]) - m_0) / std_0
            y_wafer = (float(d[hyper['context_columns'][1]][i]) - m_1)/ std_1
        x_die = float(d[hyper['context_columns'][2]][i]) / max_x_die
        y_die = float(d[hyper['context_columns'][3]][i]) / max_y_die
        #x = (float(d[hyper['context_columns'][4]][i]) - m_4) / std_4
        #y = (float(d[hyper['context_columns'][5]][i]) - m_5) / std_5

        if polar: 
            context[img_file] = [rho,phi,x_die,y_die]
        else:
            context[img_file] = [x_wafer,y_wafer,x_die,y_die]
            
        labels[img_file] = assign_label(d['cl_target'][i]) 
        
    return context , labels , filenames
    
def prepare_data_only_images_2(d, hyper):

    labels = {}
    filenames = []   

    for i in list(d['cl_manual'].index):   
    
        img_file = d['filename'][i]
        filenames.append(img_file)    
        labels[img_file] = assign_label(d['cl_manual'][i])
        
    return labels , filenames
    
def prepare_data_only_images(d, hyper):

    labels = {}
    filenames = [] 

    for i in range(len(d)): 
        img_file = d['filename_000'][i]
        filenames.append(img_file)    
        labels[img_file] = assign_label(d['cl_target'][i]) 
        
    return labels , filenames
    
def summarize_train_valid_test_data(d_train, d_valid, d_test, cl_names):
    ''' Calculates some basic class distributions within for the train and valid datasets and saves them to file.
    d_train = training dataframe
    d_valid = validation dataframe
    d_test = test dataframe
    cl_names = classes' names'''
    lb_names, n_tr, n_v , n_te = [],[],[],[]

    for elem in cl_names:
        lb_names.append(assign_label(elem))
        n_tr.append(len(d_train[d_train['cl_target'] == elem]))
        n_v.append(len(d_valid[d_valid['cl_target'] == elem]))
        n_te.append(len(d_test[d_test['cl_target'] == elem]))

    d_cl = pd.DataFrame(columns = ['cl_label','cl_id'])
    d_cl['cl_label'] = cl_names
    d_cl['cl_id'] = lb_names

    d_cl['n_train'] = n_tr
    d_cl['inv_ratio_train'] = len(d_train) / d_cl['n_train']
    d_cl['inv_ratio_norm_train'] = d_cl['inv_ratio_train'] / np.sum(d_cl['inv_ratio_train'])
    d_cl['loss_adj_train'] = d_cl['inv_ratio_norm_train'] * d_cl['n_train']
    d_cl['inv_ratio_adj_train'] = d_cl['inv_ratio_norm_train'] * len(d_train) / np.sum(d_cl['loss_adj_train'])

    d_cl['n_valid'] = n_v
    d_cl['inv_ratio_valid'] = len(d_valid) / d_cl['n_valid']
    d_cl['inv_ratio_norm_valid'] = d_cl['inv_ratio_valid'] / np.sum(d_cl['inv_ratio_valid'])
    d_cl['loss_adj_valid'] = d_cl['inv_ratio_norm_valid'] * d_cl['n_valid']
    d_cl['inv_ratio_adj_valid'] = d_cl['inv_ratio_norm_valid'] * len(d_valid) / np.sum(d_cl['loss_adj_valid'])
    
    d_cl['n_test'] = n_te
    d_cl['inv_ratio_test'] = len(d_test) / d_cl['n_test']
    d_cl['inv_ratio_norm_test'] = d_cl['inv_ratio_test'] / np.sum(d_cl['inv_ratio_test'])
    d_cl['loss_adj_test'] = d_cl['inv_ratio_norm_test'] * d_cl['n_test']
    d_cl['inv_ratio_adj_test'] = d_cl['inv_ratio_norm_test'] * len(d_test) / np.sum(d_cl['loss_adj_test'])

    # add final column that needs both train and valid data
    #d_cl['ratio_valid_train'] = d_cl['n_valid']/d_cl['n_train']
    
    return d_cl
    
    
def calculate_class_level_metrics(d, threshold, d_n_true_classes_nothresh,cfg):
    ''' Calculates precision, recall, global recall etc. based on input dataframe and cfg values'''
    n_images_filtered = d.shape[0]
    
    d_correct = d[d['real_class'] == d['cl_max']]
    d_incorrect = d[d['real_class'] != d['cl_max']]

    d_n_true_classes_filtered = d['real_class'].value_counts().to_frame(name = 'n_true_classes_filtered')
    d_n_correct = d_correct['cl_max'].value_counts().to_frame(name = 'n_correct')
    d_n_incorrect = d_incorrect['cl_max'].value_counts().to_frame(name = 'n_incorrect')
    d_n_missed = d_incorrect['real_class'].value_counts().to_frame(name = 'n_missed')

    dm = d_n_true_classes_nothresh.join(d_n_correct).join(d_n_incorrect).join(d_n_missed).join(d_n_true_classes_filtered)
    dm.index.name = 'class'
    dm.reset_index(inplace=True)
    dm.sort_values('class', inplace=True)
    dm.fillna(value=0, inplace =True)

    dm['n_classified'] = dm['n_correct'] + dm['n_incorrect']
    dm['volume'] = dm['n_true_classes_filtered'] / dm['n_true_classes_nothresh']
    dm['n_others_nothresh'] = cfg['n_img'] - dm['n_true_classes_nothresh']
    dm['n_others_filtered'] = n_images_filtered - dm['n_true_classes_filtered']
    dm['recall_nothresh'] = dm['n_correct']/dm['n_true_classes_nothresh'] # this is correct
    dm['recall_filtered'] = dm['n_correct']/dm['n_true_classes_filtered'] # this is correct
    dm['false_positive_rate_nothresh'] = dm['n_incorrect']/dm['n_others_nothresh'] 
    dm['false_positive_rate_filtered'] = dm['n_incorrect']/dm['n_others_filtered'] 
    dm['precision'] = dm['n_correct']/(dm['n_correct']+dm['n_incorrect']) # this is correct
    dm['f1_nothresh'] = 2*dm['precision']*dm['recall_nothresh']/(dm['precision'] + dm['recall_nothresh'])
    dm['f1_filtered'] = 2*dm['precision']*dm['recall_filtered']/(dm['precision'] + dm['recall_filtered'])
    dm['sm_threshold'] = threshold

    return dm
    
    
def prepare_base_table(probabilities, cl_names , classes , valid_filenames , save_name):
    ''' Turns keras probabilities into a properly named pandas data frame'''
    print('building base data frame... ')
    d = pd.DataFrame(probabilities, columns=cl_names)             # get predicted softmax values into a dataframe. Every line is an image, every column the class
    d['cl_max'] = d[cl_names].idxmax(axis=1)                      # find column with maximum value
    d['prob_max'] = d[cl_names].max(axis=1)                       # get the corresponding softmax probability
    d['real_class'] = classes                                     # get real class from image_gen
    d['filename'] = valid_filenames[:probabilities.shape[0]]      # add filename column with image name
    d.to_csv(save_name, index=False)

    return d
    
    
def simulate_softmax_thresholds(d, cfg, hyper):
    ''' Softmax threshold loop. Returns 2 data frames: d_thresh, d_thresh_cl'''
    # prepare values
    print('running softmax threshold analysis... ')
    d_thresh = pd.DataFrame(columns=['sm_threshold','accuracy','pct_classified'])                               # data frame for accuracy vs volume vs softmax threshold
    d_thresh_cl = pd.DataFrame(columns=['cl_max','precision','recall','f1-score','support','sm_threshold'])     # data with class-fine precision, recall etc.

    d_n_true_classes_nothresh = d['real_class'].value_counts().to_frame(name = 'n_true_classes_nothresh')

    for threshold in hyper['threshold_range']:
        df = d[d['prob_max'] > threshold]
        if df.shape[0] > 0: # check if any predictions remain
            # accuracy
            d_thresh.loc[len(d_thresh)] = [
                threshold,
                accuracy_score(df['real_class'], df['cl_max']),
                df.shape[0]/cfg['n_img']
            ]

            # class level metrics
            d_temp = calculate_class_level_metrics(df, threshold, d_n_true_classes_nothresh, cfg)
            d_thresh_cl = pd.concat([d_thresh_cl, d_temp], sort=True)

    return d_thresh, d_thresh_cl
    
        
def plot_accuracy_and_volume_vs_softmax(d, save_name, cfg):
    ''' Standard plot, requires pandas dataframe with sm_threshold, accuracy and pct_classified columns.
    cfg and are used for axis labels and filename'''
    plt.plot(d['sm_threshold'], d['accuracy'],  label='Accuracy')
    plt.plot(d['sm_threshold'], d['pct_classified'], label='Volume')
    plt.title('Accuracy&Volume VS Softmax')
    plt.xlabel('Softmax filter threshold')
    plt.legend()
    plt.grid()
    plt.savefig(save_name, dpi=300)
    plt.show()
    plt.close()
    
def plot_accuracy_and_volume_vs_softmax_2(d1, d2, save_name, cfg):
    ''' Standard plot, requires pandas dataframe with sm_threshold, accuracy and pct_classified columns.
    cfg and are used for axis labels and filename'''
    plt.plot(d1['sm_threshold'], d1['accuracy'], '-.b', label='Accuracy')
    plt.plot(d1['sm_threshold'], d1['pct_classified'], '-.r', label='Volume')
    plt.plot(d2['sm_threshold'], d2['accuracy'], 'b', label='Accuracy with priors')
    plt.plot(d2['sm_threshold'], d2['pct_classified'], 'r', label='Volume with priors')
    plt.title('Accuracy&Volume VS Softmax')
    plt.xlabel('Softmax filter threshold')
    plt.legend()
    plt.grid()
    plt.savefig(save_name, dpi=300)
    plt.show()
    plt.close()


def plot_accuracy_and_volume_vs_softmax_by_class(d, save_name, cfg):
    ''' Standard plot, requires pandas dataframe from calculate_class_level_metrics
    cfg and path are used for axis labels and filename'''

    elements = pd.unique(d['class']) # same as cl_list_model + weighted average
    n_elements = len(elements)
    n_rows = math.ceil(n_elements/4) # calculate number of required rows in subplot

    fig,ax = plt.subplots(nrows=n_rows, ncols=4, sharex=True, sharey=True, figsize=(15,n_rows*3.5), constrained_layout=True, squeeze=False) # squeeze false makes sure the array does not collapse if n_rows = 1
        
    for i_element in range(0,n_elements):
            
        d_temp = d[d['class']==elements[i_element]]
            
        if max(d_temp['n_true_classes_nothresh'] > 0): # if at least one image per class exists, important if 
            ax[i_element//4][i_element%4].plot(d_temp['sm_threshold'],d_temp['precision'], label='Precision')
            ax[i_element//4][i_element%4].plot(d_temp['sm_threshold'],d_temp['recall_nothresh'], label='Recall')
            ax[i_element//4][i_element%4].plot(d_temp['sm_threshold'],d_temp['volume'], label='Volume')
            ax[i_element//4][i_element%4].legend()
            ax[i_element//4][i_element%4].set_title('Class {0}, pre={1:.1%}, rec={2:.1%}, n={3}'.format(elements[i_element], d_temp['precision'].iloc[0], d_temp['recall_filtered'].iloc[0],d_temp['n_true_classes_nothresh'].iloc[0]), fontsize=12)
        else:
            ax[i_element//4][i_element%4].set_title('Class {0} not found'.format(elements[i_element]), fontsize=12)
        ax[i_element//4][i_element%4].grid()
            
        if i_element//4 == n_rows-1: # -1 is required because indexing starts from 0
            ax[i_element//4][i_element%4].set_xlabel('Softmax threshold', fontsize=12) # add labels only to the bottom plots

    fig.savefig(save_name, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()
    
def plot_accuracy_and_volume_vs_softmax_by_class_2(d1, d2, save_name, cfg):
    ''' Standard plot, requires pandas dataframe from calculate_class_level_metrics
    cfg and path are used for axis labels and filename'''

    elements = pd.unique(d1['class']) # same as cl_list_model + weighted average
    n_elements = len(elements)
    n_rows = math.ceil(n_elements/4) # calculate number of required rows in subplot

    fig,ax = plt.subplots(nrows=n_rows, ncols=4, sharex=True, sharey=True, figsize=(15,n_rows*3.5), constrained_layout=True, squeeze=False) # squeeze false makes sure the array does not collapse if n_rows = 1
        
    for i_element in range(0,n_elements):
            
        d_temp = d1[d1['class']==elements[i_element]]
        d_temp2 = d2[d2['class']==elements[i_element]]
            
        if max(d_temp['n_true_classes_nothresh'] > 0): # if at least one image per class exists, important if 
            ax[i_element//4][i_element%4].plot(d_temp['sm_threshold'],d_temp['precision'], '-.b',label='Precision')
            ax[i_element//4][i_element%4].plot(d_temp['sm_threshold'],d_temp['recall_nothresh'], '-.r', label='Recall')
            ax[i_element//4][i_element%4].plot(d_temp['sm_threshold'],d_temp['volume'], '-.g', label='Volume')
            ax[i_element//4][i_element%4].plot(d_temp2['sm_threshold'],d_temp2['precision'], 'b', label='Precision with priors')
            ax[i_element//4][i_element%4].plot(d_temp2['sm_threshold'],d_temp2['recall_nothresh'], 'r', label='Recall with priors')
            ax[i_element//4][i_element%4].plot(d_temp2['sm_threshold'],d_temp2['volume'], 'g', label='Volume with priors')
            ax[i_element//4][i_element%4].legend() 
            ax[i_element//4][i_element%4].set_title('Class {0}'.format(elements[i_element]), fontsize=12)            
        else:
            ax[i_element//4][i_element%4].set_title('Class {0} not found'.format(elements[i_element]), fontsize=12)
        ax[i_element//4][i_element%4].grid()
            
        if i_element//4 == n_rows-1: # -1 is required because indexing starts from 0
            ax[i_element//4][i_element%4].set_xlabel('Softmax threshold', fontsize=12) # add labels only to the bottom plots

    fig.savefig(save_name, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()
    
    
def plot_recall_vs_precision_by_class(d, save_name):
    ''' Standard plot, requires pandas dataframe from calculate_class_level_metrics
    cfg and path are used for axis labels and filename'''

    elements = pd.unique(d['class']) # same as cl_list_model + weighted average
    n_elements = len(elements)
    n_rows = math.ceil(n_elements/4) # calculate number of required rows in subplot

    fig, ax = plt.subplots(nrows=n_rows, ncols=4, sharex=True, sharey=True, figsize=(15,n_rows*3.5), constrained_layout=True, squeeze=False) # squeeze false makes sure the array does not collapse if n_rows = 1

    for i_element in range(0,n_elements):
        d_temp = d[d['class']==elements[i_element]]
        if max(d_temp['n_true_classes_nothresh'] > 0): # if at least one image per class exists, important if 
            ax[i_element//4][i_element%4].plot(d_temp['recall_nothresh'],d_temp['precision'])
            ax[i_element//4][i_element%4].set_title('Class {0}, n={1}'.format(elements[i_element], d_temp['n_true_classes_nothresh'].iloc[0]), fontsize=12)
        else:
            ax[i_element//4][i_element%4].set_title('Class {0} not found'.format(elements[i_element]), fontsize=12)
        ax[i_element//4][i_element%4].grid()
        
        if i_element//4 == n_rows-1: # -1 is required because indexing starts from 0
            ax[i_element//4][i_element%4].set_xlabel('recall', fontsize=12) # add labels only to the bottom plots
            
        if i_element%4 == 0:
            ax[i_element//4][i_element%4].set_ylabel('precision', fontsize=12) # add labels only to the bottom plots

    fig.savefig(save_name, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()
    
    
def model_history_analysis(mdl, path_plot, n_classes):
    ''' Analyses history of model training and saves it both as .csv and a plot '''
    d_history = pd.DataFrame({'epoch':range(1, len(mdl.history['accuracy'])+1),
                   'loss_train':mdl.history['loss'],
                   'acc_train':mdl.history['accuracy'],
                   'loss_valid':mdl.history['val_loss'],
                   'acc_valid':mdl.history['val_accuracy']})
                   #'learning_rate':mdl.history['lr']})
    #d_history.to_csv(path_csv, index=False, mode='a', float_format='%g')

    # get best epochs for validation acc and loss
    best_epoch_acc = d_history[d_history['acc_valid']==max(d_history['acc_valid'])]['epoch'].iloc[0]
    best_epoch_loss = d_history[d_history['loss_valid']==min(d_history['loss_valid'])]['epoch'].iloc[0]

    # Plot accuracy and loss
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12,4), constrained_layout=True)

    ax[0].hlines(1/n_classes, min(d_history['epoch']), max(d_history['epoch']), linestyles='dashed')
    ax[0].plot(d_history['epoch'], d_history['acc_train'], label ='Training')
    ax[0].plot(d_history['epoch'], d_history['acc_valid'], label='Validation')
    ax[0].plot(best_epoch_acc, d_history['acc_valid'][best_epoch_acc-1], marker = 'o')
    ax[0].set_title('Best epoch={}\nacc_train={:.2%}, acc_valid={:.2%}'.format(
        best_epoch_acc,
        float(d_history['acc_train'][best_epoch_acc-1]),
        float(d_history['acc_valid'][best_epoch_acc-1])))
    ax[0].set_xlabel('Epoch')
    ax[0].set_ylabel('Classification Accuracy')
    ax[0].xaxis.set_major_locator(MaxNLocator(integer=True))
    ax[0].grid()
    ax[0].legend()

    #ax[1].hlines(np.log(n_classes), min(d_history['epoch']), max(d_history['epoch']), linestyles='dashed')
    ax[1].plot(d_history['epoch'], d_history['loss_train'], label ='Training')
    ax[1].plot(d_history['epoch'], d_history['loss_valid'], label='Validation')
    ax[1].plot(best_epoch_loss, d_history['loss_valid'][best_epoch_loss-1], marker = 'o')
    ax[1].set_title('Best epoch={}\nloss_train={:.2f}, loss_valid={:.2f}'.format(
        best_epoch_loss,
        float(d_history['loss_train'][best_epoch_loss-1]),
        float(d_history['loss_valid'][best_epoch_loss-1])))
    ax[1].set_xlabel('Epoch')
    ax[1].set_ylabel('Classification Loss')
    ax[1].xaxis.set_major_locator(MaxNLocator(integer=True))
    ax[1].grid()
    ax[1].legend()

    #ax[2].plot(d_history['epoch'], d_history['learning_rate'], c=colors[0])
    #ax[2].set_title('Learning Rate')
    #ax[2].set_xlabel('Epoch')
    #ax[2].set_ylabel('Learning Rate')
    #ax[2].xaxis.set_major_locator(MaxNLocator(integer=True))
    #ax[2].grid()

    fig.savefig(path_plot, bbox_inches='tight')
    plt.show()
    plt.close()
    
    
    
    
    
