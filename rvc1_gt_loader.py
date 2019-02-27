from __future__ import absolute_import, division, print_function, unicode_literals

import os
import os.path
import warnings
import json
import cv2
import data_holders
import rvc1_class_list


def read_ground_truth(directory):
    """
    Read all the ground truth from all the sequences in the given folder.
    Each sequence is a folder containing a json file and some number of mask images

    Folder structure is is:
    000000/
        labels.json
        0.png
        1.png
        ...
    000001/
        labels.json
        0.png
        1.png
        ...
    ...

    :param directory: location of root directory where folders containing sequence's gt data are located
    :return: sequences: dictionary of sequence gt generators
    """
    sequences = []
    for sequence_dir in sorted(os.listdir(directory)):
        sequence_path = os.path.join(directory, sequence_dir)
        if os.path.isdir(sequence_path) and os.path.isfile(os.path.join(sequence_path, 'labels.json')):
            sequences.append(SequenceGTLoader(sequence_path))
    return sequences


class SequenceGTLoader:
    """
    Create a generator to read all the ground truth information for a particular sequence.
    Each iteration of the generator returns another generator over the ground truth instances
    for that image.
    Given that the image ids are integers, it is guaranteed to return the ground truth in
    numerical frame order.

    Ground truth format is a labels.json file, and some number of images containing
    the segmentation masks.
    labels.json should be:
    {
      <image_id>: {
        "_metadata": {"mask_name":<mask_image_filename>, "mask_channel": <mask_image_channel>},
        <instance_id>: {
            "class": <class_name>, "mask_id": <mask_id>, "bounding_box": <bounding_box>
            }
        ...
      }
      ...
    }

    <image_id> : The id of the particular image, a 6-digit id in ascending frame order.
    The first frame is "000000", the second "000001", etc.

    <mask_image_filename> : Path of the mask image containing the masks for this image,
    which should exist in the same folder as labels.json (e.g. "0.png")
    Note that the masks are contained in one channel of this RGB image.

    <mask_image_channel> : channel from the mask image containing the masks for this image.
    As OpenCV is in BGR order, the 0th channel will be the Blue channel of the mask image.

    <instance_id> : id given to the object itself (not just the current visualised instance thereof).
    These ids are consistent between frames, and can be used for instance tracking.

    <class_name> : string name of the given object's class

    <mask_id> : the value of the mask image pixels where this object appears.
    That is, if <mask_id> is 10, <mask_image_filename> is "0.png", and <mask_image_channel> is 0,
    then the pixels for this object are all the places the blue channel of "0.png" is 10.

    <bounding_box> : bounding box encapsulating instance mask in format [left, top, right, bottom]


    :param sequence_directory: The ground truth directory, containing labels.json
    :return: sequence_generator: generator for the gt of a given sequence over all images in that sequence.
    Note that the sequence_generator produces an image gt generator over all gt instances in that image.
    """
    def __init__(self, sequence_directory):
        self._sequence_directory = sequence_directory
        with open(os.path.join(sequence_directory, 'labels.json'), 'r') as fp:
            self._labels = json.load(fp)

    def __len__(self):
        return len(self._labels.keys())

    def __iter__(self):
        for image_id, image_name in sorted((int(l), l) for l in self._labels.keys()):
            if '_metadata' in self._labels[image_name]:
                im_mask_name = self._labels[image_name]['_metadata']['mask_name']
                mask_im = cv2.imread(os.path.join(self._sequence_directory, im_mask_name))

                yield ImageGTLoader(
                    image_data=self._labels[image_name],
                    masks=mask_im[:, :, 0]
                )
            else:
                yield []


class ImageGTLoader:
    def __init__(self, image_data, masks):
        self._image_data = image_data
        self._masks = masks

    def __len__(self):
        return len(self._image_data)

    def __iter__(self):
        """
        Read ground truth for a particular image
        :param image_data: The image data from the labels json
        :param masks: A greyscale image containing all the masks in the image
        :return: image_generator: generator of GroundTruthInstances objects for each gt instance present in
        the given image.
        """
        if len(self._image_data) > 0:
            for instance_name in sorted(self._image_data.keys()):
                if not instance_name.startswith('_'):  # Ignore metadata attributes
                    detection_data = self._image_data[instance_name]
                    class_id = rvc1_class_list.get_class_id(detection_data['class'])
                    if class_id is not None:
                        mask_id = int(detection_data['mask_id'])
                        yield data_holders.GroundTruthInstance(
                            true_class_label=class_id,
                            segmentation_mask=(self._masks == mask_id),
                            bounding_box=[int(v) for v in detection_data.get('bounding_box', [])],
                            num_pixels=int(detection_data.get('num_pixels', -1))
                        )