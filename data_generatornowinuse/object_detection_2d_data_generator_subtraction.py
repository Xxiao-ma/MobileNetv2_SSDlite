from __future__ import division
import numpy as np
import inspect
from collections import defaultdict
import warnings
import sklearn.utils
from copy import deepcopy
from PIL import Image
import cv2
import csv
import os
import sys
from tqdm import tqdm, trange
try:
    import h5py
except ImportError:
    warnings.warn("'h5py' module is missing. The fast HDF5 dataset option will be unavailable.")
try:
    from bs4 import BeautifulSoup
except ImportError:
    warnings.warn("'BeautifulSoup' module is missing. The XML-parser will be unavailable.")
try:
    import pickle
except ImportError:
    warnings.warn("'pickle' module is missing. You won't be able to save parsed file lists and annotations as pickled files.")

from ssd_encoder_decoder.ssd_input_encoder import SSDInputEncoder
from data_generator.object_detection_2d_image_boxes_validation_utils import BoxFilter

'''
pipeline of this generator:
1) xml parser will parse the data category file and load the pictures' name and the content of label files in parameters
2) create hdf5 data base file to save the picture files and labels' content, which will accelerate the data loader in the training process
3) generate data in batch 
'''

class DataGenerator:

    def __init__(self,
                    hdf5_dataset_path=None,
                    filenames=None,
                    filenames_type='text',
                    images_dir=None,
                    labels=None,
                    image_ids=None,
                    labels_output_format=('class_id', 'xmin', 'ymin', 'xmax', 'ymax'),
                    verbose=True,
                    timeseries=None):
            self.subtractions=[]
            self.labels_output_format = labels_output_format
            self.labels_format={'class_id': labels_output_format.index('class_id'),
                                'xmin': labels_output_format.index('xmin'),
                                'ymin': labels_output_format.index('ymin'),
                                'xmax': labels_output_format.index('xmax'),
                                'ymax': labels_output_format.index('ymax')} # This dictionary is for internal use.
            self.timeseries = timeseries
            self.dataset_size = 0 # As long as we haven't loaded anything yet, the dataset size is zero.
            self.images = None # The only way that this list will not stay `None` is if `load_images_into_memory == True`.
            # `self.filenames` is a list containing all file names of the image samples (full paths).
            # Note that it does not contain the actual image files themselves. This list is one of the outputs of the parser methods.
            # In case you are loading an HDF5 dataset, this list will be `None`.
            if not filenames is None:
                if isinstance(filenames, (list, tuple)):
                    self.filenames = filenames
                elif isinstance(filenames, str):
                    with open(filenames, 'rb') as f:
                        if filenames_type == 'pickle':
                            self.filenames = pickle.load(f)
                        elif filenames_type == 'text':
                            self.filenames = [os.path.join(images_dir, line.strip()) for line in f]
                        else:
                            raise ValueError("`filenames_type` can be either 'text' or 'pickle'.")
                else:
                    raise ValueError("`filenames` must be either a Python list/tuple or a string representing a filepath (to a pickled or text file). The value you passed is neither of the two.")
                self.dataset_size = len(self.filenames)
                self.dataset_indices = np.arange(self.dataset_size, dtype=np.int32)
                            
            else:
                self.filenames = None

            # In case ground truth is available, `self.labels` is a list containing for each image a list (or NumPy array)
            # of ground truth bounding boxes for that image.
            if not labels is None:
                if isinstance(labels, str):
                    with open(labels, 'rb') as f:
                        self.labels = pickle.load(f)
                elif isinstance(labels, (list, tuple)):
                    self.labels = labels
                else:
                    raise ValueError("`labels` must be either a Python list/tuple or a string representing the path to a pickled file containing a list/tuple. The value you passed is neither of the two.")
            else:
                self.labels = None

            if not image_ids is None:
                if isinstance(image_ids, str):
                    with open(image_ids, 'rb') as f:
                        self.image_ids = pickle.load(f)
                elif isinstance(image_ids, (list, tuple)):
                    self.image_ids = image_ids
                else:
                    raise ValueError("`image_ids` must be either a Python list/tuple or a string representing the path to a pickled file containing a list/tuple. The value you passed is neither of the two.")
            else:
                self.image_ids = None





    def create_hdf5_dataset(self,
                                file_path='dataset.h5',
                                resize=False,
                                variable_image_size=True,
                                verbose=True):

            self.hdf5_dataset_path = file_path

            dataset_size = len(self.filenames) #  subtractions has the same length as filenames

            # Create the HDF5 file.
            hdf5_dataset = h5py.File(file_path, 'w')

            # Create a few attributes that tell us what this dataset contains.
            # The dataset will obviously always contain images, but maybe it will
            # also contain labels, image IDs, etc.
            hdf5_dataset.attrs.create(name='has_labels', data=False, shape=None, dtype=np.bool_)
            hdf5_dataset.attrs.create(name='has_image_ids', data=False, shape=None, dtype=np.bool_)
            # It's useful to be able to quickly check whether the images in a dataset all
            # have the same size or not, so add a boolean attribute for that.
            if variable_image_size and not resize:
                hdf5_dataset.attrs.create(name='variable_image_size', data=True, shape=None, dtype=np.bool_)
            else:
                hdf5_dataset.attrs.create(name='variable_image_size', data=False, shape=None, dtype=np.bool_)

            # Create the dataset in which the images will be stored as flattened arrays.
            # This allows us, among other things, to store images of variable size.
            hdf5_images = hdf5_dataset.create_dataset(name='images',
                                                    shape=(dataset_size,),
                                                    maxshape=(None),
                                                    dtype=h5py.special_dtype(vlen=np.uint8))

            # Create the dataset that will hold the image heights, widths and channels that
            # we need in order to reconstruct the images from the flattened arrays later.
            hdf5_image_shapes = hdf5_dataset.create_dataset(name='image_shapes',
                                                            shape=(dataset_size, 3),
                                                            maxshape=(None, 3),
                                                            dtype=np.int32)


            # Create the dataset in which the labels will be stored as flattened arrays.
            hdf5_subs = hdf5_dataset.create_dataset(name='subs',
                                                    shape=(dataset_size,),
                                                    maxshape=(None),
                                                    dtype=h5py.special_dtype(vlen=np.int32))

            # Create the dataset that will hold the dimensions of the labels arrays for
            # each image so that we can restore the labels from the flattened arrays later.
            hdf5_sub_shapes = hdf5_dataset.create_dataset(name='sub_shapes',
                                                            shape=(dataset_size, 3),
                                                            maxshape=(None, 3),
                                                            dtype=np.int32)


            if not (self.labels is None):

                # Create the dataset in which the labels will be stored as flattened arrays.
                hdf5_labels = hdf5_dataset.create_dataset(name='labels',
                                                        shape=(dataset_size,),
                                                        maxshape=(None),
                                                        dtype=h5py.special_dtype(vlen=np.int32))

                # Create the dataset that will hold the dimensions of the labels arrays for
                # each image so that we can restore the labels from the flattened arrays later.
                hdf5_label_shapes = hdf5_dataset.create_dataset(name='label_shapes',
                                                                shape=(dataset_size, 2),
                                                                maxshape=(None, 2),
                                                                dtype=np.int32)

                hdf5_dataset.attrs.modify(name='has_labels', value=True)

            if not (self.image_ids is None):

                hdf5_image_ids = hdf5_dataset.create_dataset(name='image_ids',
                                                            shape=(dataset_size,),
                                                            maxshape=(None),
                                                            dtype=h5py.special_dtype(vlen=str))

                hdf5_dataset.attrs.modify(name='has_image_ids', value=True)


            if verbose:
                tr = trange(dataset_size, desc='Creating HDF5 dataset', file=sys.stdout)
            else:
                tr = range(dataset_size)

            # Iterate over all images in the dataset.
            for i in tr:
                #remember to change the input picture as single channel
                # Store the image and subtraction mask.
                img = cv2.imread(self.filenames[i],0)
                img = np.asarray(img, dtype=np.uint8)
                img = np.stack([img] * 3, axis=-1)
                sub = cv2.imread(self.subtractions[i],0)
                sub = np.asarray(sub, dtype=np.uint8)
                sub = np.stack([sub] * 3, axis=-1)
                hdf5_images[i] = img.reshape(-1)
                hdf5_image_shapes[i] = img.shape
                hdf5_subs[i] = sub.reshape(-1)
                hdf5_sub_shapes[i] = sub.shape

                # Store the ground truth if we have any.
                if not (self.labels is None):

                    labels = np.asarray(self.labels[i])
                    # Flatten the labels array and write it to the labels dataset.
                    hdf5_labels[i] = labels.reshape(-1)
                    # Write the labels' shape to the label shapes dataset.
                    hdf5_label_shapes[i] = labels.shape

                # Store the image ID if we have one.
                if not (self.image_ids is None):

                    hdf5_image_ids[i] = self.image_ids[i]

            hdf5_dataset.close()
            self.hdf5_dataset = h5py.File(file_path, 'r')
            self.hdf5_dataset_path = file_path
            self.dataset_size = len(self.hdf5_dataset['images'])
            self.dataset_indices = np.arange(self.dataset_size, dtype=np.int32) # Instead of shuffling the HDF5 dataset, we will shuffle this index list.



    def parse_xml(self,
                    images_dirs,
                    image_set_filenames,
                    annotations_dirs=[],
                    classes=['background',
                            'aeroplane', 'bicycle', 'bird', 'boat',
                            'bottle', 'bus', 'car', 'cat',
                            'chair', 'cow', 'diningtable', 'dog',
                            'horse', 'motorbike', 'person', 'pottedplant',
                            'sheep', 'sofa', 'train', 'tvmonitor'],
                    include_classes = 'all',
                    exclude_truncated=False,
                    exclude_difficult=False,
                    ret=False,
                    verbose=True):

            # Set class members.
            self.images_dirs = images_dirs
            self.annotations_dirs = annotations_dirs
            self.image_set_filenames = image_set_filenames
            self.classes = classes
            self.include_classes = include_classes

            # Erase data that might have been parsed before.
            self.filenames = []
            self.image_ids = []
            self.labels = []
            self.subtractions = [] # add subtraction name list
            if not annotations_dirs:
                self.labels = None
                #self.eval_neutral = None
                annotations_dirs = [None] * len(images_dirs)
            names =set()

            

            for images_dir, image_set_filename, annotations_dir in zip(images_dirs, image_set_filenames, annotations_dirs):
                # Read the image set file that so that we know all the IDs of all the images to be included in the dataset.
                with open(image_set_filename) as f:
                    image_ids = [line.strip() for line in f] # Note: These are strings, not integers.
                    self.image_ids += image_ids

                if verbose:
                    it = tqdm(image_ids, desc="Processing image set '{}'".format(os.path.basename(image_set_filename)), file=sys.stdout)
                else: 
                    it = image_ids

                # Loop over all images in this dataset.
                for image_id in it:

                    filename = '{}'.format(image_id) + '.pgm' #改变为pgm
                    subtraction = '{}'.format(image_id) + '_sub.pgm' # set id of subtraction masks
                    self.filenames.append(os.path.join(images_dir, filename))
                    self.subtractions.append(os.path.join(images_dir, subtraction))  # add background subtraction image
                    
                    # Parse the XML file for this image.
                    if not annotations_dir is None:
                        with open(os.path.join(annotations_dir, image_id + '.xml')) as f:
                            soup = BeautifulSoup(f, 'xml')

                        folder = soup.folder.text # In case we want to return the folder in addition to the image file name. Relevant for determining which dataset an image belongs to.
                        #filename = soup.filename.text

                        boxes = [] # We'll store all boxes for this image here.
                        eval_neutr = [] # We'll store whether a box is annotated as "difficult" here.
                        objects = soup.find_all('object') # Get a list of all objects in this image.
                        # Parse the data for each object.
                        for obj in objects:
                            class_name = obj.find('name', recursive=False).text
                            names.add(class_name)
                            #print(class_name)
                            class_id = self.classes.index(class_name)
                            # Check whether this class is supposed to be included in the dataset.
                            if (not self.include_classes == 'all') and (not class_id in self.include_classes): continue
                            pose = obj.find('pose', recursive=False).text
                            truncated = int(obj.find('truncated', recursive=False).text)
                            if exclude_truncated and (truncated == 1): continue
                            difficult = int(obj.find('difficult', recursive=False).text)
                            if exclude_difficult and (difficult == 1): continue
                            # Get the bounding box coordinates.
                            bndbox = obj.find('bndbox', recursive=False)
                            xmin = int(bndbox.xmin.text)
                            ymin = int(bndbox.ymin.text)
                            xmax = int(bndbox.xmax.text)
                            ymax = int(bndbox.ymax.text)
                            item_dict = {'folder': folder,
                                        'image_name': filename,
                                        'image_id': image_id,
                                        'class_name': class_name,
                                        'class_id': class_id,
                                        'pose': pose,
                                        'truncated': truncated,
                                        'difficult': difficult,
                                        'xmin': xmin,
                                        'ymin': ymin,
                                        'xmax': xmax,
                                        'ymax': ymax}
                            box = []
                            for item in self.labels_output_format:
                                box.append(item_dict[item])
                            boxes.append(box)
                            if difficult: eval_neutr.append(True)
                            else: eval_neutr.append(False)

                        self.labels.append(boxes)

            self.dataset_size = len(self.filenames)
            self.dataset_indices = np.arange(self.dataset_size, dtype=np.int32)
            
            if ret:
                return self.images, self.filenames, self.labels, self.image_ids, self.eval_neutral, self.subtractions # the subtractions locate at the end of the return information


    def generate(self,
                 batch_size=32,
                 shuffle=False,
                 transformations=[],
                 label_encoder=None,
                 returns={'processed_images', 'encoded_labels'},
                 keep_images_without_gt=False,
                 mode='train',
                 degenerate_box_handling='remove'):
        
        if self.dataset_size == 0:
            raise DatasetError("Cannot generate batches because you did not load a dataset.")

        #############################################################################################
        # Warn if any of the set returns aren't possible.
        #############################################################################################

        if self.labels is None:
            if any([ret in returns for ret in ['original_labels', 'processed_labels', 'encoded_labels', 'matched_anchors', 'evaluation-neutral']]):
                warnings.warn("Since no labels were given, none of 'original_labels', 'processed_labels', 'evaluation-neutral', 'encoded_labels', and 'matched_anchors' " +
                              "are possible returns, but you set `returns = {}`. The impossible returns will be `None`.".format(returns))
        elif label_encoder is None:
            if any([ret in returns for ret in ['encoded_labels', 'matched_anchors']]):
                warnings.warn("Since no label encoder was given, 'encoded_labels' and 'matched_anchors' aren't possible returns, " +
                              "but you set `returns = {}`. The impossible returns will be `None`.".format(returns))
        elif not isinstance(label_encoder, SSDInputEncoder):
            if 'matched_anchors' in returns:
                warnings.warn("`label_encoder` is not an `SSDInputEncoder` object, therefore 'matched_anchors' is not a possible return, " +
                              "but you set `returns = {}`. The impossible returns will be `None`.".format(returns))
        elif self.subs is None:
            if any([ret in returns for ret in ['subtractions']]):
                warnings.warn("Since no subtraction mask was given, 'subtractions' isn't possible returns, " +
                              "but you set `returns = {}`. The impossible returns will be `None`.".format(returns))

        #############################################################################################
        # Do a few preparatory things like maybe shuffling the dataset initially.
        #############################################################################################
        #
        if shuffle:
            objects_to_shuffle = [self.dataset_indices]
            if not (self.filenames is None):
                objects_to_shuffle.append(self.filenames)
            if not (self.labels is None):
                objects_to_shuffle.append(self.labels)
            if not (self.image_ids is None):
                objects_to_shuffle.append(self.image_ids) 
            if not (self.subtractions is None):
                objects_to_shuffle.append(self.subtractions)
            shuffled_objects = sklearn.utils.shuffle(*objects_to_shuffle)
            for i in range(len(objects_to_shuffle)):
                objects_to_shuffle[i][:] = shuffled_objects[i]

        if degenerate_box_handling == 'remove':
            box_filter = BoxFilter(check_overlap=False,
                                   check_min_area=False,
                                   check_degenerate=True,
                                   labels_format=self.labels_format)

        # Override the labels formats of all the transformations to make sure they are set correctly.
        if not (self.labels is None):
            for transform in transformations:
                transform.labels_format = self.labels_format

        #############################################################################################
        # Generate mini batches.
        #############################################################################################

        current = 0

        while True:

            batch_X, batch_subs, batch_y = [], [], []

            if current >= self.dataset_size:
                current = 0

            #########################################################################################
            # Maybe shuffle the dataset if a full pass over the dataset has finished.
            #########################################################################################

                if shuffle:
                    objects_to_shuffle = [self.dataset_indices]
                    if not (self.filenames is None):
                        objects_to_shuffle.append(self.filenames)
                    if not (self.subtractions is None):
                        objects_to_shuffle.append(self.subtractions)
                    if not (self.labels is None):
                        objects_to_shuffle.append(self.labels)
                    if not (self.image_ids is None):
                        objects_to_shuffle.append(self.image_ids)
                    shuffled_objects = sklearn.utils.shuffle(*objects_to_shuffle)
                    for i in range(len(objects_to_shuffle)):
                        objects_to_shuffle[i][:] = shuffled_objects[i]

            #########################################################################################
            # Get the images, (maybe) image IDs, (maybe) labels, etc. for this batch.
            #########################################################################################

            # We prioritize our options in the following order:
            # 1) If we have the images already loaded in memory, get them from there.
            # 2) Else, if we have an HDF5 dataset, get the images from there.
            # 3) Else, if we have neither of the above, we'll have to load the individual image
            #    files from disk.
            batch_indices = self.dataset_indices[current:current+batch_size]
#
            if not (self.images is None):#in memory
                #print(self.images.shape)
                for i in batch_indices:
                    batch_X.append(self.images[i])
                    batch_subs.append(self.subs[i])
                if not (self.filenames is None):
                    batch_filenames = self.filenames[current:current+batch_size]
                else:
                    batch_filenames = None
            if not (self.hdf5_dataset is None): # basically in use
                for i in batch_indices:
                    batch_X.append(self.hdf5_dataset['images'][i].reshape(self.hdf5_dataset['image_shapes'][i]))
                #if not (self.subs is None): #append subtractions
                    batch_subs.append(self.hdf5_dataset['subs'][i].reshape(self.hdf5_dataset['sub_shapes'][i]))
                if not (self.filenames is None):
                    batch_filenames = self.filenames[current:current+batch_size]
                else:
                    batch_filenames = None

            #print("This is image:"+str(batch_filenames))
            #print("This is subtraction:"+str(batch_filenames)+'_sub')
            # Get the labels for this batch (if there are any).
            if not (self.labels is None):
                batch_y = deepcopy(self.labels[current:current+batch_size])
            else:
                batch_y = None
            temp_batch_y = batch_y # Placeholder for labels in later transformation

            # Get the image IDs for this batch (if there are any).
            if not (self.image_ids is None):
                batch_image_ids = self.image_ids[current:current+batch_size]
            else:
                batch_image_ids = None

            if 'original_images' in returns:
                batch_original_images = deepcopy(batch_X) # The original, unaltered images
            if 'original_labels' in returns:
                batch_original_labels = deepcopy(batch_y) # The original, unaltered labels

            current += batch_size
            
            #########################################################################################
            # Maybe perform image transformations.
            #########################################################################################

            batch_items_to_remove = [] # In case we need to remove any images from the batch, store their indices in this list.
            batch_inverse_transforms = []

            for i in range(len(batch_X)):
                
                # if self.labels is None:
                    #print("this image "+str(i)+"has no labels")
                if not (self.labels is None):
                    
                    # Convert the labels for this image to an array (in case they aren't already).
                    batch_y[i] = np.array(batch_y[i])
                    # If this image has no ground truth boxes, maybe we don't want to keep it in the batch.
                    if (batch_y[i].size == 0) and not keep_images_without_gt:
                        #print("the image "+str(i)+" has no ground truth")
                        batch_items_to_remove.append(i)
                        batch_inverse_transforms.append([])

                        continue

                # Apply any image transformations we may have received.
                if transformations:

                    inverse_transforms = []

                    for transform in transformations:

                        if not (self.labels is None):

                            if ('inverse_transform' in returns) and ('return_inverter' in inspect.signature(transform).parameters):
                                batch_X[i], batch_y[i], inverse_transform = transform(batch_X[i], batch_y[i], return_inverter=True)
                                print(self.subtractions[i])
                                print(batch_subs[i])
                                print('------------------')
                                print(batch_X[i])
                                batch_subs[i], temp = transform(batch_subs[i], batch_y[i])
                                inverse_transforms.append(inverse_transform)
                            else:
                                batch_X[i], batch_y[i] = transform(batch_X[i], batch_y[i])
                                #print(self.subtractions[i])
                                #print(batch_subs[i])
                                batch_subs[i], temp = transform(batch_subs[i], batch_y[i])

                            if batch_X[i] is None: # In case the transform failed to produce an output image, which is possible for some random transforms.
                                print("the image "+str(i)+" in batch is empty")
                                batch_items_to_remove.append(i)
                                batch_inverse_transforms.append([])
                                continue

                        else:

                            if ('inverse_transform' in returns) and ('return_inverter' in inspect.signature(transform).parameters):
                                batch_X[i], inverse_transform = transform(batch_X[i], return_inverter=True)
                                batch_subs[i],temp = transform(batch_subs[i], return_inverter=True)
                                inverse_transforms.append(inverse_transform)
                            else:
                                batch_X[i] = transform(batch_X[i])
                                batch_subs[i] = transform(batch_subs[i])

                    batch_inverse_transforms.append(inverse_transforms[::-1])

                #########################################################################################
                # Check for degenerate boxes in this batch item.
                #########################################################################################

                if not (self.labels is None):

                    xmin = self.labels_format['xmin']
                    ymin = self.labels_format['ymin']
                    xmax = self.labels_format['xmax']
                    ymax = self.labels_format['ymax']

                    if np.any(batch_y[i][:,xmax] - batch_y[i][:,xmin] <= 0) or np.any(batch_y[i][:,ymax] - batch_y[i][:,ymin] <= 0):
                        if degenerate_box_handling == 'warn':
                            warnings.warn("Detected degenerate ground truth bounding boxes for batch item {} with bounding boxes {}, ".format(i, batch_y[i]) +
                                          "i.e. bounding boxes where xmax <= xmin and/or ymax <= ymin. " +
                                          "This could mean that your dataset contains degenerate ground truth boxes, or that any image transformations you may apply might " +
                                          "result in degenerate ground truth boxes, or that you are parsing the ground truth in the wrong coordinate format." +
                                          "Degenerate ground truth bounding boxes may lead to NaN errors during the training.")
                        elif degenerate_box_handling == 'remove':
                            batch_y[i] = box_filter(batch_y[i])
                            if (batch_y[i].size == 0) and not keep_images_without_gt:
                                print("the image "+str(i)+" has label , whose length is 0")

                                batch_items_to_remove.append(i)

            #########################################################################################
            # Remove any items we might not want to keep from the batch.
            #########################################################################################

            if batch_items_to_remove:
                for j in sorted(batch_items_to_remove, reverse=True):
                    # This isn't efficient, but it hopefully shouldn't need to be done often anyway.
                    batch_X.pop(j)
                    batch_filenames.pop(j)
                    batch_subs.pop(j)
                    if batch_inverse_transforms: batch_inverse_transforms.pop(j)
                    if not (self.labels is None): batch_y.pop(j)
                    if not (self.image_ids is None): batch_image_ids.pop(j)
                    if 'original_images' in returns: batch_original_images.pop(j)
                    if 'original_labels' in returns and not (self.labels is None): batch_original_labels.pop(j)

            #########################################################################################

            # CAUTION: Converting `batch_X` into an array will result in an empty batch if the images have varying sizes
            #          or varying numbers of channels. At this point, all images must have the same size and the same
            #          number of channels.
            #print("this is the batch size & sub size")
            #print(len(batch_X))
            #print(len(batch_subs))
            #print(batch_X)
            batch_X = np.array(batch_X)
            batch_subs = np.array(batch_subs)
            if (batch_X.size == 0):
                raise DegenerateBatchError("You produced an empty batch. This might be because the images in the batch vary " +
                                           "in their size and/or number of channels. Note that after all transformations " +
                                           "(if any were given) have been applied to all images in the batch, all images " +
                                           "must be homogenous in size along all axes.")
            # check if this is used as a timeseries input
            if not self.timeseries == None:
                batch_X = batch_X.reshape(batch_X.size, 1, 3)
                batch_subs = batch_subs.reshape(batch_subs.size, 1, 3)

            #########################################################################################
            # If we have a label encoder, encode our labels.
            #########################################################################################

            if not (label_encoder is None or self.labels is None):

                if ('matched_anchors' in returns) and isinstance(label_encoder, SSDInputEncoder):
                    batch_y_encoded, batch_matched_anchors = label_encoder(batch_y, diagnostics=True)
                else:
                    batch_y_encoded = label_encoder(batch_y, diagnostics=False)
                    batch_matched_anchors = None

            else:
                batch_y_encoded = None
                batch_matched_anchors = None

            #########################################################################################
            # Compose the output.
            #########################################################################################

            # print(" batch_X.shape is :")
            # print(batch_X.shape)
            # print(" batch_subs.shape is :")
            # print(batch_subs[0].shape)
            # print(" batch_y.shape is :")
            # print(batch_y.shape)
            if mode == 'inference':
                yield [batch_X, batch_subs], batch_filenames, batch_inverse_transforms
            else:
                yield [batch_X, batch_subs], batch_y_encoded
# yield gnext[0], [gnext[1], gnext[1], gnext[1]]