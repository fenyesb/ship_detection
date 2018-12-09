#!/usr/bin/env python
# coding: utf-8

#Based on the 'weight_sampling_tutorial.ipynb'

#We are planning to do transfer learning based on the MS COCO model
#But that one was trained on 80 classes and we only need 1 therefor for compatibility reasons we have to downsample it

#The model can be downloaded from here: https://drive.google.com/file/d/1vmEF7FUsWfHquXyCqO17UaXOPpRbwsdj/view

#Import libraries
import h5py
import numpy as np
import shutil

from misc_utils.tensor_sampling_utils import sample_tensors


# ## 1. Load the trained weights file and make a copy
# 
# First, we'll load the HDF5 file that contains the trained weights that we need (the source file). In our case this is "`VGG_coco_SSD_300x300_iter_400000.h5`, which are the weights of the original SSD300 model that was trained on MS COCO.
# 
# Then, we'll make a copy of that weights file. That copy will be our output file (the destination file).

weights_source_path = 'VGG_coco_SSD_300x300_iter_400000.h5'

weights_destination_path = 'ShipDetection_trasfered_from_VGG_coco_SSD_300x300_iter_400000.h5'

# Make a copy of the weights file.
shutil.copy(weights_source_path, weights_destination_path)


# Load both the source weights file and the copy we made.
# We will load the original weights file in read-only mode so that we can't mess up anything.
weights_source_file = h5py.File(weights_source_path, 'r')
weights_destination_file = h5py.File(weights_destination_path)


# ## 2. Figure out which weight tensors we need to sub-sample
# 
# Next, we need to figure out exactly which weight tensors we need to sub-sample. As mentioned above, the weights for all layers except the classification layers are fine, we don't need to change anything about those.
# 
# So which are the classification layers in SSD300? Their names are:

classifier_names = ['conv4_3_norm_mbox_conf',
                    'fc7_mbox_conf',
                    'conv6_2_mbox_conf',
                    'conv7_2_mbox_conf',
                    'conv8_2_mbox_conf',
                    'conv9_2_mbox_conf']


# ## 3. Figure out which slices to pick
# 
# The following section is optional. I'll look at one classification layer and explain what we want to do, just for your understanding. If you don't care about that, just skip ahead to the next section.
# 
# We know which weight tensors we want to sub-sample, but we still need to decide which (or at least how many) elements of those tensors we want to keep. Let's take a look at the first of the classifier layers, "`conv4_3_norm_mbox_conf`". Its two weight tensors, the kernel and the bias, have the following shapes:

conv4_3_norm_mbox_conf_kernel = weights_source_file[classifier_names[0]][classifier_names[0]]['kernel:0']
conv4_3_norm_mbox_conf_bias = weights_source_file[classifier_names[0]][classifier_names[0]]['bias:0']

print("Shape of the '{}' weights:".format(classifier_names[0]))
print()
print("kernel:\t", conv4_3_norm_mbox_conf_kernel.shape)
print("bias:\t", conv4_3_norm_mbox_conf_bias.shape)


# So the last axis has 324 elements. Why is that?
# 
# - MS COCO has 80 classes, but the model also has one 'backgroud' class, so that makes 81 classes effectively.
# - The 'conv4_3_norm_mbox_loc' layer predicts 4 boxes for each spatial position, so the 'conv4_3_norm_mbox_conf' layer has to predict one of the 81 classes for each of those 4 boxes.
# 
# That's why the last axis has 4 * 81 = 324 elements.
# 
# So how many elements do we want in the last axis for this layer?
# 
# Let's do the same calculation as above:
# 
# - Our dataset has 1 class, but our model will also have a 'background' class, so that makes 2 classes effectively.
# - We need to predict one of those 2 classes for each of the four boxes at each spatial position.
# 
# That makes 4 * 2 = 8 elements.
# 
# Now we know that we want to keep 8 elements in the last axis and leave all other axes unchanged. But which 8 elements out of the original 324 elements do we want?
# 
# Should we just pick them randomly? If the object classes in our dataset had absolutely nothing to do with the classes in MS COCO, then choosing those 8 elements randomly would be fine (and the next section covers this case, too)


n_classes_source = 81
classes_of_interest = [0, 37] #0: background, 37: randomly selected

subsampling_indices = []
for i in range(int(324/n_classes_source)):
    indices = np.array(classes_of_interest) + i * n_classes_source
    subsampling_indices.append(indices)
subsampling_indices = list(np.concatenate(subsampling_indices))

print(subsampling_indices)


# These are the indices of the 8 elements that we want to pick from both the bias vector and from the last axis of the kernel tensor.
# 
# This was the detailed example for the '`conv4_3_norm_mbox_conf`' layer. And of course we haven't actually sub-sampled the weights for this layer yet, we have only figured out which elements we want to keep. The piece of code in the next section will perform the sub-sampling for all the classifier layers.

# ## 4. Sub-sample the classifier weights
# 
# The code in this section iterates over all the classifier layers of the source weights file and performs the following steps for each classifier layer:
# 
# 1. Get the kernel and bias tensors from the source weights file.
# 2. Compute the sub-sampling indices for the last axis. The first three axes of the kernel remain unchanged.
# 3. Overwrite the corresponding kernel and bias tensors in the destination weights file with our newly created sub-sampled kernel and bias tensors.
# 
# The second step does what was explained in the previous section.
# 
# In case you want to **up-sample** the last axis rather than sub-sample it, simply set the `classes_of_interest` variable below to the length you want it to have. The added elements will be initialized either randomly or optionally with zeros. Check out the documentation of `sample_tensors()` for details.


n_classes_source = 81 # 80+1 for MS COCO
classes_of_interest = [0, 37]

for name in classifier_names:
    # Get the trained weights for this layer from the source HDF5 weights file.
    kernel = weights_source_file[name][name]['kernel:0'].value
    bias = weights_source_file[name][name]['bias:0'].value

    # Get the shape of the kernel. We're interested in sub-sampling
    # the last dimension, 'o'.
    height, width, in_channels, out_channels = kernel.shape
    
    # Compute the indices of the elements we want to sub-sample.
    # Keep in mind that each classification predictor layer predicts multiple
    # bounding boxes for every spatial location, so we want to sub-sample
    # the relevant classes for each of these boxes.
    if isinstance(classes_of_interest, (list, tuple)):
        subsampling_indices = []
        for i in range(int(out_channels/n_classes_source)):
            indices = np.array(classes_of_interest) + i * n_classes_source
            subsampling_indices.append(indices)
        subsampling_indices = list(np.concatenate(subsampling_indices))
    elif isinstance(classes_of_interest, int):
        subsampling_indices = int(classes_of_interest * (out_channels/n_classes_source))
    else:
        raise ValueError("`classes_of_interest` must be either an integer or a list/tuple.")
    
    # Sub-sample the kernel and bias.
    # The `sample_tensors()` function used below provides extensive
    # documentation, so don't hesitate to read it if you want to know
    # what exactly is going on here.
    new_kernel, new_bias = sample_tensors(weights_list=[kernel, bias],
                                          sampling_instructions=[height, width, in_channels, subsampling_indices],
                                          axes=[[3]], # The one bias dimension corresponds to the last kernel dimension.
                                          init=['gaussian', 'zeros'],
                                          mean=0.0,
                                          stddev=0.005)
    
    # Delete the old weights from the destination file.
    del weights_destination_file[name][name]['kernel:0']
    del weights_destination_file[name][name]['bias:0']
    # Create new datasets for the sub-sampled weights.
    weights_destination_file[name][name].create_dataset(name='kernel:0', data=new_kernel)
    weights_destination_file[name][name].create_dataset(name='bias:0', data=new_bias)

# Make sure all data is written to our output file before this sub-routine exits.
weights_destination_file.flush()


# That's it, we're done.
# 
# Let's just quickly inspect the shapes of the weights of the '`conv4_3_norm_mbox_conf`' layer in the destination weights file:


conv4_3_norm_mbox_conf_kernel = weights_destination_file[classifier_names[0]][classifier_names[0]]['kernel:0']
conv4_3_norm_mbox_conf_bias = weights_destination_file[classifier_names[0]][classifier_names[0]]['bias:0']

print("Shape of the '{}' weights:".format(classifier_names[0]))
print()
print("kernel:\t", conv4_3_norm_mbox_conf_kernel.shape)
print("bias:\t", conv4_3_norm_mbox_conf_bias.shape)