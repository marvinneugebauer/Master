import os
import cv2
import numpy as np
import argparse
from shared_prediction import *

# Argument Parser
parser = argparse.ArgumentParser(description='Pathes to the image folders')

parser.add_argument("path_to_pred_depth", help="Path to the predicted depth estimation images.", type=str)

parser.add_argument("path_to_depth_gt", help="Path to the ground truth depth estimation images.", type=str)

parser.add_argument("path_to_pred_semseg", help="Path to the predicted semantic segmentation images.", type=str)

parser.add_argument("path_to_semseg_gt", help="Path to the ground truth semantic sementation images.", type=str)

parser.add_argument("dataset",
                    help="Enter the name of the dataset you are using, which is Cityscapes or LostandFound",
                    type=str)

parser.add_argument("approach", help="Enter if the approach is restrictive or nonrestrictive", type=str)

args = parser.parse_args()


class Dataloader:
    def __init__(self, dataset, binarization):
        self.dataset = dataset
        self.binarization = binarization

        self.prediction_images = []
        self.prediction_images_names = []
        self.gt_images = []
        self.gt_images_names = []

        print(self.dataset)

    def get_label(self):
        """ Set the label according to the considered dataset which could be Cityscapes or Lost and Found. Within the
        dataset the function makes a further distinction by setting the labels depending on the binarization that was
        chosen by the user.
        """
        if self.dataset == "Cityscapes" and self.binarization == "restrictive":
            foreground = {1, 2, 3, 4, 5, 6, 7, 8, 11, 12, 13, 14, 15, 16, 17, 18}
            background = {0, 9, 10, 255}
        elif self.dataset == "Cityscapes" and self.binarization == "nonrestrictive":
            foreground = {2, 3, 4, 5, 6, 7, 8, 11, 12, 13, 14, 15, 16, 17, 18}
            background = {0, 1, 2, 8, 9, 10, 255}
        elif self.dataset == "LostandFound" and self.binarization == "restrictive":
            foreground = {2}
            background = {1, 255}
        elif self.dataset == "LostandFound" and self.binarization == "nonrestrictive":
            foreground = {2}
            background = {1, 255}
        return foreground, background

    def get_binarization(self, image_non_bin):
        """
        For a given image, dataset and chosen kind of binarization (e.i. restrictive or nonrestrictive) this function
        returns binarized image.
        """
        labels_image = np.unique(image_non_bin)
        for trainId in labels_image:
            if trainId in self.get_label()[0]:
                image_non_bin[image_non_bin == trainId] = 1
            elif trainId == 255:
                image_non_bin[image_non_bin == trainId] = 255
            else:
                image_non_bin[image_non_bin == trainId] = 0
        return image_non_bin

    def bin_check(self, gt_image):
        """
        For a given image this function checks what kind of binarization is necessary and applies this approach to the
        image. Then this function returns a binarized image.
        """
        if self.dataset == "LostandFound" and args.approach == "restrictive":
            gt_image = self.get_binarization(gt_image)
        elif self.dataset == "LostandFound" and args.approach == "nonrestrictive":
            gt_image = self.get_binarization(gt_image)
        elif self.dataset == "Cityscapes" and args.approach == "restrictive":
            gt_image = self.get_binarization(gt_image)
        elif self.dataset == "Cityscapes" and args.approach == "restrictive":
            gt_image = self.get_binarization(gt_image)
        return gt_image

    def image_loader(self, path_to_pred, path_to_gt):
        """
         The data_loader function has two arguments: the path to the folder with the predicted images and the path to
         the folder with the ground truth images. The purpose of this function is to load the images and bring the
         images in the right form. The right form is achieved by conducting a binarization on each image. The data
         loader function updates 4 lists: The first list self.prediction_images_names contains the names of the
         binarized prediction images. The second list self.prediction_images contains binarized prediction images.
         The third list self.gt_images contains the binarized ground truth images. The fourth list
         self.gt_images_names contains the names of the binarized ground truth images.
        """

        for image_name in os.listdir(path_to_pred):

            # Read the image and check if a binarization is necessary
            if path_to_pred == args.path_to_pred_semseg:
                image = cv2.imread(os.path.join(path_to_pred, image_name), 0)
                image = self.get_binarization(image)
            else:
                image = cv2.imread(os.path.join(path_to_pred, image_name), 0)

            for gt_image_name in os.listdir(path_to_gt):
                if path_to_pred == args.path_to_pred_semseg:
                    if gt_image_name.replace("gtFine_", "") == image_name:
                        gt_image = cv2.imread(os.path.join(path_to_gt, gt_image_name), 0)
                        gt_image = self.bin_check(gt_image)
                        image[gt_image == 255] = 255  # Set ignore regions of gt_image to 255
                        self.prediction_images_names.append(image_name)
                        self.prediction_images.append(image)
                        self.gt_images.append((gt_image, gt_image_name))
                        self.gt_images_names.append(gt_image_name)
                else:
                    if gt_image_name.replace("gtFine_labelTrainIds", "leftImg8bit") == image_name:
                        gt_image = cv2.imread(os.path.join(path_to_gt, gt_image_name), 0)
                        gt_image = self.bin_check(gt_image)
                        image[gt_image == 255] = 255  # Set ignore regions of gt_image to 255
                        self.prediction_images_names.append(image_name)
                        self.prediction_images.append(image)
                        self.gt_images.append((gt_image, gt_image_name))
                        self.gt_images_names.append(gt_image_name)


class Evaluate:
    def __init__(self, pred_semseg, pred_gt):
        """
        We initialize all instances that we need for our calculation of each metric.
        """
        self.pred_semseg = pred_semseg
        self.pred_gt = pred_gt
        self.length = len(pred_gt)

        self.iou_add = 0
        self.iou_foreground_add = 0
        self.iou_background_add = 0
        self.mean_iou_add = 0
        self.pixel_acc = 0
        self.intersection_foreground_total = 0
        self.intersection_background_total = 0
        self.gt_foreground_total = 0
        self.gt_background_total = 0
        self.union_foreground_total = 0
        self.union_background_total = 0
        self.false_pos_foreground_total = 0
        self.false_pos_background_total = 0
        self.false_neg_foreground_total = 0
        self.false_neg_background_total = 0
        self.ignore_pred_total = 0
        self.ignore_gt_total = 0

        self.intersection_foreground = 0
        self.intersection_background = 0
        self.union_sem_foreground = 0
        self.union_sem_background = 0
        self.gt_bin_foreground = 0
        self.gt_bin_background = 0

        self.iou_average = 0
        self.mean_iou_average = 0
        self.iou_foreground_average = 0
        self.iou_background_average = 0
        self.iou_total = 0
        self.mean_iou_aLL_total = 0
        self.iou_foreground_total = 0
        self.iou_background_total = 0
        self.pixel_accuracy_average = 0
        self.pixel_accuracy_total = 0
        self.true_pos_foreground_average = 0
        self.true_pos_background_average = 0
        self.true_pos_average = 0
        self.true_pos_total = 0
        self.false_pos_foreground_average = 0
        self.false_pos_background_average = 0
        self.false_pos_average = 0
        self.false_pos_total = 0
        self.false_neg_foreground_average = 0
        self.false_neg_background_average = 0
        self.false_neg_average = 0
        self.false_neg_total = 0

    @staticmethod
    def get_pixels_foreground(image_foreground):
        """
        For a given image this method computes all pixels with the label 1 and returns a set of tuples, which consists of
        a pixel, and it's assigned label.
        """
        x = np.where(image_foreground == 1)[0]
        y = np.where(image_foreground == 1)[1]
        S = set()
        class_foreground_array = np.array((x, y)).T
        for z in class_foreground_array:
            z = tuple(z.tolist())
            S.add(z)
        return S

    @staticmethod
    def get_pixels_background(image_background):
        """
        For a given image this method computes all pixels with the label 0 and returns a set of tuples, which consists of
        a pixel, and it's assigned label.
        """
        a = np.where(image_background == 0)[0]
        b = np.where(image_background == 0)[1]
        z = set()
        class_foreground_array = np.array((a, b)).T
        for c in class_foreground_array:
            c = tuple(c.tolist())
            z.add(c)
        return z

    @staticmethod
    def get_ignore(image_c):
        """
        For a given image this method counts all pixels with ignore regions (e.i. label of 255) and returns their number.
        """
        labels_image2 = np.unique(image_c, return_counts=True)
        if 255 in labels_image2[0]:
            a = np.where(labels_image2[0] == 255)
            b = (a[0][0])
            return labels_image2[1][b]
        else:
            return 0

    def calculate_iou(self):
        """
        Returns the calculated IoU (Intersection over Union) for a single image.
        """
        if len(self.union_sem_foreground) + len(self.union_sem_background) == 0:
            return 0
        else:
            iou = (len(self.intersection_foreground) + len(self.intersection_background)) /\
                  (len(self.union_sem_foreground) + len(self.union_sem_background))
            return iou

    def calculate_iou_foreground(self):
        """
        Returns the calculated IoU (Intersection over Union) of the class foreground for a single image.
        """
        if len(self.union_sem_foreground) == 0:
            return 0
        else:
            iou_foreground = (len(self.intersection_foreground) / len(self.union_sem_foreground))
            return iou_foreground

    def calculate_iou_background(self):
        """
        Returns the calculated IoU (Intersection over Union) of the class background for a single image.
        """
        if len(self.union_sem_background) == 0:
            return 0
        else:
            iou_background = (len(self.intersection_background) / len(self.union_sem_background))
            return iou_background

    def calculate_iou_all_images(self):
        """
        Returns the calculated IoU (Intersection over Union) for all images.
        """
        iou_all_images = ((self.intersection_foreground_total + self.intersection_background_total) /
                          (self.union_foreground_total + self.union_background_total))
        return iou_all_images

    def calculate_iou_foreground_all_images(self):
        """
        Returns the calculated IoU (Intersection over Union) of the class foreground for all images.
        """
        iou_foreground_all_images = (self.intersection_foreground_total / self.union_foreground_total)
        return iou_foreground_all_images

    def calculate_iou_background_all_images(self):
        """
        Returns the calculated IoU (Intersection over Union) of the class background for all images.
        """
        iou_background = (self.intersection_background_total / self.union_background_total)
        return iou_background

    def calculate_mean_iou(self):
        """
        Returns the calculated mean IoU ( mean Intersection over Union) for a single image.
        """

        if len(self.union_sem_foreground) == 0 or len(self.union_sem_background) == 0:
            return 0
        else:
            mean_iou = (1 / 2) * ((len(self.intersection_foreground) / len(self.union_sem_foreground)) +
                                  (len(self.intersection_background) / len(self.union_sem_background)))
            return mean_iou

    def calculate_mean_iou_all_images(self):
        """
        Returns the calculated mean IoU ( mean Intersection over Union) for all images.
        """

        mean_iou_all_im = (1 / 2) * ((self.intersection_foreground_total / self.union_foreground_total) +
                                     (self.intersection_background_total / self.union_background_total))
        return mean_iou_all_im

    def calculate_pixel_accuracy(self):
        """
        Returns the calculated pixel accuracy for a single image.
        """
        if len(self.gt_bin_foreground) + len(self.gt_bin_background) == 0:
            return 0
        else:
            pixel_accuracy = (len(self.intersection_foreground) + len(self.intersection_background)) / (
                    len(self.gt_bin_foreground) + len(self.gt_bin_background))
            return pixel_accuracy

    def calculate_pixel_accuracy_all_images(self):
        """
        Returns the calculated mean pixel accuracy for all images.
        """
        pixel_accuracy = (self.intersection_foreground_total + self.intersection_background_total) / (
                self.gt_foreground_total + self.gt_background_total)
        return pixel_accuracy

    def prepare_eval(self):
        """
        The purpose of this method is to update all variables that we need in order to calculate the metrics.
        """

        # Iterate over all images and update all variables that are needed
        for i in range(0, len(self.pred_semseg)):
            # Calculate the image wise number of pixels that belong to class foreground for the predicted image
            pred_class_1 = self.get_pixels_foreground(self.pred_semseg[i])

            # Calculate the image wise number of pixels that belong to class background for the predicted image
            pred_class_0 = self.get_pixels_background(self.pred_semseg[i])

            # Calculate the image wise number of pixels that belong to class foreground for the ground truth image
            self.gt_bin_foreground = self.get_pixels_foreground(self.pred_gt [i][0])
            self.gt_foreground_total += len(self.gt_bin_foreground)

            # Calculate the image wise number of pixels that belong to class background for the ground truth image
            self.gt_bin_background= self.get_pixels_background(self.pred_gt[i][0])
            self.gt_background_total+= len(self.gt_bin_background)

            # Calculate the image wise number of pixels that belong to ignore region for the prediction image
            ignore_pred = self.get_ignore(self.pred_semseg)  # zaehlt die ignore regionen je pered-bild
            self.ignore_pred_total += ignore_pred

            # Calculate the image wise number of pixels that belong to ignore region for the ground truth image
            ignore_gt = self.get_ignore(self.pred_gt [i][0])
            self.ignore_gt_total += ignore_gt

            # Calculate the image wise number of pixels that are false positives for the class foreground
            false_positives_class_1 = pred_class_1.difference(self.gt_bin_foreground)
            self.false_pos_foreground_total += len(false_positives_class_1)

            # Calculate the image wise number of pixels that are false positives for the class background
            false_positives_class_0 = pred_class_0.difference(self.gt_bin_background)
            self.false_pos_background_total += len(false_positives_class_0)

            # Calculate the image wise number of pixels that are false negatives for the class foreground
            false_negatives_class_1 = self.gt_bin_foreground.difference(pred_class_1)
            self.false_neg_foreground_total += len(false_negatives_class_1)

            # Calculate the image wise number of pixels that are false negatives for the class background
            false_negatives_class_0 = self.gt_bin_background.difference(pred_class_0)
            self.false_neg_background_total += len(false_negatives_class_0)

            # Calculate the image wise IoU for the class foreground
            self.intersection_foreground = pred_class_1.intersection(self.gt_bin_foreground)
            self.intersection_foreground_total += len(self.intersection_foreground)

            # Calculate the image wise IoU for the class background
            self.intersection_background = pred_class_0.intersection(self.gt_bin_background)
            self.intersection_background_total += len(self.intersection_background)

            # Calculate the image wise Union for the class foreground
            self.union_sem_foreground = pred_class_1.union(self.gt_bin_foreground)
            self.union_foreground_total += len(self.union_sem_foreground)

            # Calculate the image wise Union for the class background
            self.union_sem_background = pred_class_0.union(self.gt_bin_background)
            self.union_background_total += len(self.union_sem_background)

            # Calculate the image wise Union for the class background
            self.iou_add += self.calculate_iou()
            self.iou_foreground_add += self.calculate_iou_foreground()
            self.iou_background_add += self.calculate_iou_background()
            self.mean_iou_add += self.calculate_mean_iou()
            self.pixel_acc += self.calculate_pixel_accuracy()

    def eval(self):
        """
        This method receives the variable the path to folder with the images as an input and calculates all values
        for each considered metric.
        """
        print("self.length: ", self.length)
        self.iou_average = self.iou_add / self.length
        self.mean_iou_average = self.mean_iou_add / self.length

        # Compute the average IoU of the classes foreground and background
        self.iou_foreground_average = self.iou_foreground_add  / self.length  # IoU_foreground_total
        self.iou_background_average = self.iou_background_add  / self.length

        # Compute IoU and mean IoU for all images
        self.iou_total = self.calculate_iou_all_images()
        self.mean_iou_aLL_total = self.calculate_mean_iou_all_images()

        # Compute the IoU for the classes foreground and background
        self.iou_foreground_total = self.calculate_iou_foreground_all_images()
        self.iou_background_total = self.calculate_iou_background_all_images()

        # Compute the average pixel accuracy per image
        self.pixel_accuracy_average = self.pixel_acc / self.length

        # Compute the total pixel accuracy for all images
        self.pixel_accuracy_total = self.calculate_pixel_accuracy_all_images()

        # Compute the average number of true positives for the classes foreground and background as well as their sum
        self.true_pos_foreground_average = self.intersection_foreground_total / self.length
        self.true_pos_background_average = self.intersection_background_total / self.length
        self.true_pos_average = self.true_pos_foreground_average + self.true_pos_background_average

        # Compute the average number of true positives for all images
        self.true_pos_total = self.intersection_foreground_total + self.intersection_background_total

        # Compute the average number of false positives for the classes foreground and background as well as their sum
        self.false_pos_foreground_average = self.false_pos_foreground_total / self.length
        self.false_pos_background_average = self.false_pos_foreground_total / self.length
        self.false_pos_average = self.false_pos_foreground_average + self.false_pos_background_average

        # Compute the total number of false positives for all images
        self.false_pos_total = self.false_pos_foreground_total + self.false_pos_background_total

        # Compute the average number of false negatives for the classes foreground and background as well as their sum
        self.false_neg_foreground_average = self.false_neg_foreground_total / self.length
        self.false_neg_background_average = self.false_neg_background_total / self.length
        self.false_neg_average = self.false_neg_foreground_average + self.false_neg_background_average

        # Compute the total number of false negatives for all images
        self.false_neg_total = self.false_neg_foreground_total + self.false_neg_background_total

    def print_iou(self):
        """
        This method prints all values for each different kind of IoU we are considering.
        """

        report_iou = f'Result for the avarage IoU per image: IoU_average={np.round(self.iou_average, decimals=4)} \n ' \
                     f'Result for the average mean IoU per image: ' \
                     f'IoU_average={np.round(self.mean_iou_average, decimals=4)}  \n ' \
                     f'Result for the average mean IoU per image for the class foreground: ' \
                     f'IoU_foreground_average: ={np.round(self.iou_foreground_average , decimals=4)}  \n ' \
                     f'Result for the average mean IoU per image for the class background:' \
                     f' IoU_background_average={np.round(self.iou_background_average, decimals=4)}  \n ' \
                     f'Result for the IoU over all images: IoU_average={np.round(self.iou_total, decimals=4)}  \n ' \
                     f'Result for the mean IoU over all images: ' \
                     f'mean_IoU_average={np.round(self.mean_iou_aLL_total, decimals=4)} ' \
                     f'Result for the IoU over all images for the class foreground: ' \
                     f'IoU_foreground_average={np.round(self.iou_foreground_total, decimals=4)}  \n ' \
                     f'Result for the IoU over all images for the class background: ' \
                     f'IoU_background_average={np.round(self.iou_background_total, decimals=4)}  \n ' \

        print(report_iou)

    def print_pixel_accuracy(self):
        """
        This method prints all values for each different kind of pixel accuracy we are considering.
        """

        report_pixel_accuracy = f'Result for the avarage pixel accuracy per image: ' \
                                f'pixel_accuracy_average={np.round(self.pixel_accuracy_average, decimals=4)}  \n ' \
                                f'Result for the average mean IoU per image: ' \
                                f'IoU_average={np.round(self.pixel_accuracy_total, decimals=4)} \n ' \

        print(report_pixel_accuracy)

    def print_true_positives(self):
        """
        This method prints all values for each different kind of true positives we are considering.
        """
        true_positives_total = np.round(self.intersection_foreground_total + self.intersection_background_total,
                                        decimals=4)

        report_true_pos = f'Result for the average number of true positives per image for the class foreground :' \
                          f'true_pos_foreground_average ={np.round(self.true_pos_foreground_average, decimals=4)}  \n '\
                          f'Result for the average number of true positives per image for the class background :' \
                          f'true_pos_background_average ={np.round(self.true_pos_background_average, decimals=4)} \n '\
                          f'Result for the average number of true positives per image :' \
                          f'true_pos_average ={np.round(self.true_pos_average, decimals=4)}  \n ' \
                          f'Result for the total number of true positives for the class foreground :' \
                          f'true_positives_foreground_total =' \
                          f'{np.round(self.intersection_foreground_total, decimals=4)}  \n ' \
                          f'Result for the total number of true positives for the class background :' \
                          f'true_positives_background_total =' \
                          f'{np.round(self.intersection_background_total,decimals =4)}  \n ' \
                          f'Result for the total number of true positives:' \
                          f'true_positives_total = {true_positives_total}  \n ' \

        print(report_true_pos)

    def print_false_positives(self):
        """
        This method prints all values for each different kind of false positives we are considering.
        """
        false_positives_total = np.round(self.false_pos_foreground_total + self.false_pos_background_total, decimals=4)

        report_false_pos = f'Result for the average number of false positives per image for the class foreground :' \
                           f'false_pos_foreground_average =' \
                           f'{np.round(self.false_pos_foreground_average, decimals=4)}  \n ' \
                           f'Result for the average number of false positives per image for the class background :' \
                           f'false_pos_backgroundd_average =' \
                           f'{np.round(self.false_pos_background_average, decimals=4)}  \n ' \
                           f'Result for the average number of false positives per image :' \
                           f'false_pos_average ={np.round(self.false_pos_average, decimals=4)}  \n ' \
                           f'Result for the total number of false positives for the class foreground :' \
                           f'false_positives_foreground_total =' \
                           f'{np.round(self.false_pos_foreground_total, decimals=4)}  \n ' \
                           f'Result for the total number of false positives for the class background :' \
                           f'false_positives_background_total =' \
                           f'{np.round(self.false_pos_background_total, decimals=4)}  \n ' \
                           f'Result for the total number of false positives:' \
                           f'false_positives_total = {false_positives_total}  \n ' \

        print(report_false_pos)

    def print_false_negatives(self):
        """
        This method prints all values for each different kind of false negatives we are considering.
        """
        false_negatives_total = np.round(self.false_neg_foreground_total + self.false_neg_background_total, decimals=4)

        report_false_neg = f'Result for the average number of false negatives per image for the class foreground :' \
                           f'false_neg_foreground_average =' \
                           f'{np.round(self.false_neg_foreground_average , decimals=4)}  \n ' \
                           f'Result for the average number of false negatives per image for the class background :' \
                           f'false_neg_background_average =' \
                           f'{np.round(self.false_neg_background_average , decimals=4)}  \n ' \
                           f'Result for the average number of false negatives per image :' \
                           f'false_neg_average = {np.round(self.false_neg_average, decimals=4)}  \n ' \
                           f'Result for the total number of false positives for the class foreground :' \
                           f'false_negatives_foreground_total =' \
                           f'{np.round(self.false_neg_foreground_total, decimals=4)}  \n ' \
                           f'Result for the total number of false positives for the class background :' \
                           f'false_negatives_background_total =' \
                           f'{np.round(self.false_neg_background_total, decimals=4)}  \n ' \
                           f'Result for the total number of false negatives:' \
                           f'false_negatives_total = {false_negatives_total}  \n ' \

        print(report_false_neg)

    def print_additional_information(self):

        add_inf = f'Additional Information: '\
                  f'Result for the total number of pixels for all ground truth images that belong to the class ' \
                  f'foreground: gt_foreground_total = {np.round(self.gt_foreground_total, decimals=4)}  \n ' \
                  f'Result for the total number of pixels for all ground truth images that belong to the class ' \
                  f'background: gt_background_total = {np.round(self.gt_background_total, decimals=4)}  \n ' \
                  f'Result for the total number of pixels for all predicted images that belong to the ignore regions:' \
                  f'ignore_pred_total= {np.round(self.ignore_pred_total, decimals=4)}  \n ' \
                  f'Result for the total number of pixels for all ground truth images that belong to the ignore ' \
                  f'regions: ignore_pred_total= {np.round(self.ignore_gt_total, decimals=4)}  \n ' \

        print(add_inf)


def main():
    """
    Main Function
    """
    print("__________________________________________________________________________________________")
    print("__________________________________________________________________________________________")
    print("____ Compute the Accuracy of the Semantic Segmentation Net____")
    print("__________________________________________________________________________________________")
    print("__________________________________________________________________________________________")

    # Load the binarized images of the semantic segmentation net
    semseg = Dataloader(args.dataset, args.approach)
    semseg.image_loader(args.path_to_pred_semseg, args.path_to_semseg_gt)

    # Generate a list that contains all the names of predicted images by the semantic segmentation net.
    pred_semseg_img_names = semseg.prediction_images_names

    # Generate a list that contains all the predicted images by the semantic segmentation net.
    pred_semseg_img = semseg.prediction_images

    print("Number of images that are used: ", len(pred_semseg_img))

    # Generate a list that contains all the ground truth images.
    pred_gt = semseg.gt_images

    # Generate a list that contains all the names of the ground truth images.
    gt_array_names = semseg.gt_images_names

    # Generate the evaluation for the semantic segmentation net and print the results
    semseg_eval = Evaluate(pred_semseg_img, pred_gt)
    semseg_eval.prepare_eval()
    semseg_eval.eval()
    semseg_eval.print_iou()
    semseg_eval.print_pixel_accuracy()
    semseg_eval.print_true_positives()
    semseg_eval.print_false_positives()
    semseg_eval.print_false_negatives()
    semseg_eval.print_additional_information()

    print("                                                                                             ")
    print("                                                                                             ")
    print("                                                                                             ")
    print("                                                                                             ")

    print("__________________________________________________________________________________________")
    print("__________________________________________________________________________________________")
    print("___ Compute the Accuracy of the Depth Estimation Net___")
    print("__________________________________________________________________________________________")
    print("__________________________________________________________________________________________")

    # Load the binarized images of the depth net
    depth_net = Dataloader(args.dataset, args.approach)
    depth_net.image_loader(args.path_to_pred_depth, args.path_to_depth_gt)

    # Generate a list that contains all the names of predicted images by the depth estimation net.
    pred_depth_img_names = depth_net.prediction_images_names

    # Generate a list that contains all the predicted images by the depth estimation net.
    pred_depth_images = depth_net.prediction_images

    # Generate a list that contains all the ground truth images that are used by the depth estimation net.
    pred_gt = depth_net.gt_images

    # Generate the evaluation for the depth estimation net and print the results
    depth_net_eval = Evaluate(pred_depth_images, pred_gt)
    depth_net_eval.prepare_eval()
    depth_net_eval.eval()
    depth_net_eval.print_iou()
    depth_net_eval.print_pixel_accuracy()
    depth_net_eval.print_true_positives()
    depth_net_eval.print_false_positives()
    depth_net_eval.print_false_negatives()
    depth_net_eval.print_additional_information()

    print("                                                                                             ")
    print("                                                                                             ")
    print("                                                                                             ")
    print("                                                                                             ")
    print("__________________________________________________________________________________________")
    print("__________________________________________________________________________________________")
    print("____________ Compute the Accuracy of the Common Prediction of both Nets____________________________")

    # Generate the shared prediction images by applying the shared prediction function
    shared_pred = shared_prediction(pred_semseg_img, pred_depth_images, pred_semseg_img_names,
                                    pred_depth_img_names, pred_gt)[0]

    # Ensure that the array of ground truth images has the same matching order as the array of shared prediction images
    ground_truth = shared_prediction(pred_semseg_img, pred_depth_images, pred_semseg_img_names,
                                     pred_depth_img_names, pred_gt)[1]

    # Generate the evaluation for the common prediction and print the results
    common_pred_eval = Evaluate(shared_pred, ground_truth)
    common_pred_eval.prepare_eval()
    common_pred_eval.eval()
    common_pred_eval.print_iou()
    common_pred_eval.print_pixel_accuracy()
    common_pred_eval.print_true_positives()
    common_pred_eval.print_false_negatives()
    common_pred_eval.print_false_negatives()
    common_pred_eval.print_additional_information()


if __name__ == '__main__':
    main()
