from __future__ import division
import numpy as np
from keras.models import Model
from keras.layers import Input, Lambda, Activation, Conv2D, \
    DepthwiseConv2D, Reshape, Concatenate, BatchNormalization, ReLU
import keras.backend as K
from layers.AnchorBoxesLayer import AnchorBoxes
from layers.DecodeDetectionsLayer import DecodeDetections
from layers.DecodeDetectionsFastLayer import DecodeDetectionsFast
from models.graphs.mobilenet_v2_ssdlite_praph_modi import mobilenet_v2_ssdlite


def predict_block(inputs, out_channel, sym, id):
    name = 'ssd_' + sym + '{}'.format(id)
    x = DepthwiseConv2D(kernel_size=3, strides=1,
                           activation=None, use_bias=False, padding='same', name=name + '_dw_conv')(inputs)
    x = BatchNormalization(epsilon=1e-3, momentum=0.999, name=name + '_dw_bn')(x)
    x = ReLU(6., name=name + '_dw_relu')(x)

    x = Conv2D(out_channel, kernel_size=1, padding='same', use_bias=False,
                  activation=None, name=name + 'conv2')(x)
    x = BatchNormalization(epsilon=1e-3, momentum=0.999, name=name + 'conv2_bn')(x)
    return x


def mobilenet_v2_ssd(image_size,
            n_classes,
            mode='training',
            l2_regularization=0.0005,
            min_scale=None,
            max_scale=None,
            scales=None,
            aspect_ratios_global=None,
            aspect_ratios_per_layer=[[1.0, 2.0, 0.5],
                                     [1.0, 2.0, 0.5, 3.0, 1.0 / 3.0],
                                     [1.0, 2.0, 0.5, 3.0, 1.0 / 3.0],
                                     [1.0, 2.0, 0.5, 3.0, 1.0 / 3.0],
                                     [1.0, 2.0, 0.5],
                                     [1.0, 2.0, 0.5]],
            two_boxes_for_ar1=True,
            steps=[8, 16, 32, 64, 100, 300],
            offsets=None,
            clip_boxes=False,
            variances=[0.1, 0.1, 0.2, 0.2],
            coords='centroids',
            normalize_coords=True,
            divide_by_stddev=None,
            confidence_thresh=0.01,
            iou_threshold=0.45,
            top_k=200,
            nms_max_output_size=400,
            return_predictor_sizes=False):
  

    n_predictor_layers = 6  # The number of predictor conv layers in the network is 6 for the original SSD300.
    n_classes += 1  # Account for the background class.
    l2_reg = l2_regularization  # Make the internal name shorter.
    img_height, img_width, img_channels = image_size[0], image_size[1], image_size[2]

    ############################################################################
    # Get a few exceptions out of the way.
    ############################################################################

    if aspect_ratios_global is None and aspect_ratios_per_layer is None:
        raise ValueError(
            "`aspect_ratios_global` and `aspect_ratios_per_layer` \
            cannot both be None. At least one needs to be specified.")
    if aspect_ratios_per_layer:
        if len(aspect_ratios_per_layer) != n_predictor_layers:
            raise ValueError(
                "It must be either aspect_ratios_per_layer is None or len(aspect_ratios_per_layer) == {}, \
                but len(aspect_ratios_per_layer) == {}.".format(
                    n_predictor_layers, len(aspect_ratios_per_layer)))

    if (min_scale is None or max_scale is None) and scales is None:
        raise ValueError("Either `min_scale` and `max_scale` or `scales` need to be specified.")
    if scales:
        if len(scales) != n_predictor_layers + 1:
            raise ValueError("It must be either scales is None or len(scales) == {}, but len(scales) == {}.".format(
                n_predictor_layers + 1, len(scales)))
    # If no explicit list of scaling factors was passed,
    # compute the list of scaling factors from `min_scale` and `max_scale`
    else:
        scales = np.linspace(min_scale, max_scale, n_predictor_layers + 1)

    if len(variances) != 4:
        raise ValueError("4 variance values must be pased, but {} values were received.".format(len(variances)))
    variances = np.array(variances)
    if np.any(variances <= 0):
        raise ValueError("All variances must be >0, but the variances given are {}".format(variances))

    if (not (steps is None)) and (len(steps) != n_predictor_layers):
        raise ValueError("You must provide at least one step value per predictor layer.")

    if (not (offsets is None)) and (len(offsets) != n_predictor_layers):
        raise ValueError("You must provide at least one offset value per predictor layer.")

    ############################################################################
    # Compute the anchor box parameters.
    ############################################################################

    # Set the aspect ratios for each predictor layer. These are only needed for the anchor box layers.
    if aspect_ratios_per_layer:
        aspect_ratios = aspect_ratios_per_layer
    else:
        aspect_ratios = [aspect_ratios_global] * n_predictor_layers

    # Compute the number of boxes to be predicted per cell for each predictor layer.
    # We need this so that we know how many channels the predictor layers need to have.
    if aspect_ratios_per_layer:
        n_boxes = []
        for ar in aspect_ratios_per_layer:
            if (1 in ar) & two_boxes_for_ar1:
                n_boxes.append(len(ar) + 1)  # +1 for the second box for aspect ratio 1
            else:
                n_boxes.append(len(ar))
    # If only a global aspect ratio list was passed,
    # then the number of boxes is the same for each predictor layer
    else:
        if (1 in aspect_ratios_global) & two_boxes_for_ar1:
            n_boxes = len(aspect_ratios_global) + 1
        else:
            n_boxes = len(aspect_ratios_global)
        n_boxes = [n_boxes] * n_predictor_layers

    if steps is None:
        steps = [None] * n_predictor_layers
    if offsets is None:
        offsets = [None] * n_predictor_layers

    ############################################################################
    # Define functions for the Lambda layers below.
    ############################################################################

    def identity_layer(tensor):
        return tensor

    def input_stddev_normalization(tensor):
        return tensor / np.array(divide_by_stddev)



    ############################################################################
    # Build the network.
    ############################################################################

    x = Input(shape=(img_height, img_width, img_channels),name = 'img_input')
    x_sub = Input(shape=(img_height, img_width, img_channels),name = 'sub_input')
    # The following identity layer is only needed so that the subsequent lambda layers can be optional.
    x1 = Lambda(identity_layer, output_shape=(img_height, img_width, img_channels), name='identity_layer')(x)
    x2 = Lambda(identity_layer, output_shape=(img_height, img_width, img_channels), name='identity_layer2')(x_sub)
    if not (divide_by_stddev is None):
        x1 = Lambda(input_stddev_normalization, output_shape=(img_height, img_width, img_channels),
                    name='input_stddev_normalization')(x1)

    links = mobilenet_v2_ssdlite(x1,x2)

    link1_cls = predict_block(links[0], n_boxes[0] * n_classes, 'cls', 1)
    link2_cls = predict_block(links[1], n_boxes[1] * n_classes, 'cls', 2)
    link3_cls = predict_block(links[2], n_boxes[2] * n_classes, 'cls', 3)
    link4_cls = predict_block(links[3], n_boxes[3] * n_classes, 'cls', 4)
    link5_cls = predict_block(links[4], n_boxes[4] * n_classes, 'cls', 5)
    link6_cls = predict_block(links[5], n_boxes[5] * n_classes, 'cls', 6)

    link1_box = predict_block(links[0], n_boxes[0] * 4, 'box', 1)
    link2_box = predict_block(links[1], n_boxes[1] * 4, 'box', 2)
    link3_box = predict_block(links[2], n_boxes[2] * 4, 'box', 3)
    link4_box = predict_block(links[3], n_boxes[3] * 4, 'box', 4)
    link5_box = predict_block(links[4], n_boxes[4] * 4, 'box', 5)
    link6_box = predict_block(links[5], n_boxes[5] * 4, 'box', 6)

    priorbox1 = AnchorBoxes(img_height, img_width, this_scale=scales[0], next_scale=scales[1],
                                             aspect_ratios=aspect_ratios[0],
                                             two_boxes_for_ar1=two_boxes_for_ar1, this_steps=steps[0],
                                             this_offsets=offsets[0], clip_boxes=clip_boxes,
                                             variances=variances, coords=coords, normalize_coords=normalize_coords,
                                             name='ssd_priorbox_1')(link1_box)
    priorbox2 = AnchorBoxes(img_height, img_width, this_scale=scales[1], next_scale=scales[2],
                                    aspect_ratios=aspect_ratios[1],
                                    two_boxes_for_ar1=two_boxes_for_ar1, this_steps=steps[1], this_offsets=offsets[1],
                                    clip_boxes=clip_boxes,
                                    variances=variances, coords=coords, normalize_coords=normalize_coords,
                                    name='ssd_priorbox_2')(link2_box)
    priorbox3 = AnchorBoxes(img_height, img_width, this_scale=scales[2], next_scale=scales[3],
                                        aspect_ratios=aspect_ratios[2],
                                        two_boxes_for_ar1=two_boxes_for_ar1, this_steps=steps[2],
                                        this_offsets=offsets[2], clip_boxes=clip_boxes,
                                        variances=variances, coords=coords, normalize_coords=normalize_coords,
                                        name='ssd_priorbox_3')(link3_box)
    priorbox4 = AnchorBoxes(img_height, img_width, this_scale=scales[3], next_scale=scales[4],
                                        aspect_ratios=aspect_ratios[3],
                                        two_boxes_for_ar1=two_boxes_for_ar1, this_steps=steps[3],
                                        this_offsets=offsets[3], clip_boxes=clip_boxes,
                                        variances=variances, coords=coords, normalize_coords=normalize_coords,
                                        name='ssd_priorbox_4')(link4_box)
    priorbox5 = AnchorBoxes(img_height, img_width, this_scale=scales[4], next_scale=scales[5],
                                        aspect_ratios=aspect_ratios[4],
                                        two_boxes_for_ar1=two_boxes_for_ar1, this_steps=steps[4],
                                        this_offsets=offsets[4], clip_boxes=clip_boxes,
                                        variances=variances, coords=coords, normalize_coords=normalize_coords,
                                        name='ssd_priorbox_5')(link5_box)
    priorbox6 = AnchorBoxes(img_height, img_width, this_scale=scales[5], next_scale=scales[6],
                                        aspect_ratios=aspect_ratios[5],
                                        two_boxes_for_ar1=two_boxes_for_ar1, this_steps=steps[5],
                                        this_offsets=offsets[5], clip_boxes=clip_boxes,
                                        variances=variances, coords=coords, normalize_coords=normalize_coords,
                                        name='ssd_priorbox_6')(link6_box)

    # Reshape
    cls1_reshape = Reshape((-1, n_classes), name='ssd_cls1_reshape')(link1_cls)
    cls2_reshape = Reshape((-1, n_classes), name='ssd_cls2_reshape')(link2_cls)
    cls3_reshape = Reshape((-1, n_classes), name='ssd_cls3_reshape')(link3_cls)
    cls4_reshape = Reshape((-1, n_classes), name='ssd_cls4_reshape')(link4_cls)
    cls5_reshape = Reshape((-1, n_classes), name='ssd_cls5_reshape')(link5_cls)
    cls6_reshape = Reshape((-1, n_classes), name='ssd_cls6_reshape')(link6_cls)

    box1_reshape = Reshape((-1, 4), name='ssd_box1_reshape')(link1_box)
    box2_reshape = Reshape((-1, 4), name='ssd_box2_reshape')(link2_box)
    box3_reshape = Reshape((-1, 4), name='ssd_box3_reshape')(link3_box)
    box4_reshape = Reshape((-1, 4), name='ssd_box4_reshape')(link4_box)
    box5_reshape = Reshape((-1, 4), name='ssd_box5_reshape')(link5_box)
    box6_reshape = Reshape((-1, 4), name='ssd_box6_reshape')(link6_box)

    priorbox1_reshape = Reshape((-1, 8), name='ssd_priorbox1_reshape')(priorbox1)
    priorbox2_reshape = Reshape((-1, 8), name='ssd_priorbox2_reshape')(priorbox2)
    priorbox3_reshape = Reshape((-1, 8), name='ssd_priorbox3_reshape')(priorbox3)
    priorbox4_reshape = Reshape((-1, 8), name='ssd_priorbox4_reshape')(priorbox4)
    priorbox5_reshape = Reshape((-1, 8), name='ssd_priorbox5_reshape')(priorbox5)
    priorbox6_reshape = Reshape((-1, 8), name='ssd_priorbox6_reshape')(priorbox6)

    cls = Concatenate(axis=1, name='ssd_cls')(
        [cls1_reshape, cls2_reshape, cls3_reshape, cls4_reshape, cls5_reshape, cls6_reshape])

    box = Concatenate(axis=1, name='ssd_box')(
        [box1_reshape, box2_reshape, box3_reshape, box4_reshape, box5_reshape, box6_reshape])

    priorbox = Concatenate(axis=1, name='ssd_priorbox')(
        [priorbox1_reshape, priorbox2_reshape, priorbox3_reshape,
         priorbox4_reshape, priorbox5_reshape, priorbox6_reshape])

    cls = Activation('softmax', name='ssd_mbox_conf_softmax')(cls)

    predictions = Concatenate(axis=2, name='ssd_predictions')([cls, box, priorbox])

    # define input output formats
    if mode == 'training':
        model = Model(inputs=[x,x_sub], outputs=predictions)
    elif mode == 'inference':
        decoded_predictions = DecodeDetections(confidence_thresh=confidence_thresh,
                                               iou_threshold=iou_threshold,
                                               top_k=top_k,
                                               nms_max_output_size=nms_max_output_size,
                                               coords=coords,
                                               normalize_coords=normalize_coords,
                                               img_height=img_height,
                                               img_width=img_width,
                                               name='ssd_decoded_predictions')(predictions)
        model = Model(inputs=[x,x_sub], outputs=decoded_predictions)
    elif mode == 'inference_fast':
        decoded_predictions = DecodeDetectionsFast(confidence_thresh=confidence_thresh,
                                                   iou_threshold=iou_threshold,
                                                   top_k=top_k,
                                                   nms_max_output_size=nms_max_output_size,
                                                   coords=coords,
                                                   normalize_coords=normalize_coords,
                                                   img_height=img_height,
                                                   img_width=img_width,
                                                   name='ssd_decoded_predictions')(predictions)
        model = Model(inputs=[x,x_sub], outputs=decoded_predictions)
    else:
        raise ValueError(
            "`mode` must be one of 'training', 'inference' or 'inference_fast', but received '{}'.".format(mode))

    if return_predictor_sizes:
        predictor_sizes = np.array(
            [cls[0]._keras_shape[1:3], cls[1]._keras_shape[1:3], cls[2]._keras_shape[1:3],
             cls[3]._keras_shape[1:3], cls[4]._keras_shape[1:3], cls[5]._keras_shape[1:3]])
        return model, predictor_sizes
    else:
        return model