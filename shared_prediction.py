def shared_prediction(pred_semseg, pred_depth, pred_semseg_img_names, pred_depth_img_names, ground_truth_images):
    """
    The shared_prediction function takes five parameters: pred_semseg is a list of the binarized predicted images
    predicted by the semantic segmentation network; pred_depth is a list of the predicted images predicted by depth
    net network; pred_semseg_img_names is a list that contains the names of the predicted images by the semantic
    segmentation net; pred_depth_img_names is a list that contains the names of the predicted images by the depth net;
    ground_truth_images is a list that contains the names of the ground truth images. The shared_prediction function
    return two parameters: shared_pred is a list of the images received by the shared prediction; shared_ground_truth is
    a list of the ground Truth images so that the order of that list is matching with the order of shared_pred. The
    shared prediction is created in such a manner that for each pixel of a given predicted image, this pixel is assigned
    to the class foreground whenever the semantic segmentation net or the depth net is assigned that pixel to
    foreground.
    """
    shared_pred = []  # List that contains the shared prediction images
    shared_ground_truth = []  # List that contains the corresponding ground truth images

    for i in range(0, len(pred_semseg)):
        for j in range(0, len(pred_depth)):

            # Check if the prediction image matches the right ground truth image
            if pred_semseg_img_names[i].replace("labelTrainIds", "leftImg8bit") == pred_depth_img_names[j]:
                prediction_shared = pred_semseg[i].copy()

                # Generate the shared prediction and add the shared preddiction it to a list
                prediction_shared[pred_depth[j] == 1] = 1
                shared_pred.append(prediction_shared)

                # Keep track of the matching order between the prediction image and the corresponding ground truth image
                for k in range(0, len(pred_semseg)):
                    if pred_semseg_img_names[i] == ground_truth_images[k][1].replace("gtFine_", ""):
                        shared_ground_truth.append((ground_truth_images[k][0], ground_truth_images[k][1]))

    return shared_pred, shared_ground_truth
