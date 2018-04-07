'''	==================================================================================================

    ==================================================================================================  '''
import os
import sys
import shutil
import datetime
import logging
import queue
import statistics
from statistics import StatisticsError
import threading
import time
import traceback
from collections import ChainMap
from io import BytesIO
from multiprocessing import Queue, Process

import exiftool
import cv2
import numpy as np
# import libraw
import rawpy
# from rawkit.raw import LibRaw
# from rawkit.raw import Raw

from PyQt5.QtCore import QByteArray, QIODevice, QDataStream
from PyQt5.QtGui import QImage, QPixmap

# set-up logger before anything - two  handlers : one on console, the other one on file
formatter = \
    logging.Formatter("%(asctime)s :: %(funcName)s :: %(levelname)s :: %(message)s")

handler_file = logging.FileHandler("photo1.log", mode="a", encoding="utf-8")
handler_console = logging.StreamHandler()

handler_file.setFormatter(formatter)
handler_console.setFormatter(formatter)

handler_file.setLevel(logging.DEBUG)
handler_console.setLevel(logging.DEBUG)

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)  # A D A P T   LOGGING LEVEL        H E R E
logger.addHandler(handler_file)
logger.addHandler(handler_console)


class StoreQPixmap:
    """
    This class is not to be instantiated and is there to provide classes that will inherits from it the ability to store
    picture formated in different ways

    It is needed because QT Qpixmap are not natively pickable and thereroe it implements a way to pickling qpixmap
    into a Qbytearray. Else a simple dictionnary would have been enough
    """

    def __init__(self):
        # store QPixmap computed image - It is a dictionary as several images can be stored for the same picture and
        # the dictionary key is the way to distinguish them
        self._qpixmap = {}

    def set_qpixmap(self, hash_, qpixmap):
        self._qpixmap[hash_] = qpixmap

    def get_qpixmap(self, hash_):
        try:
            return self._qpixmap[hash_]
        except KeyError:
            return None

    def del_qpixmap(self, hash_):
        try:
            del self._qpixmap[hash_]
            return True
        except KeyError:
            return False

    def __getstate__(self):
        return self._pickle_stored_qpixmap()

    def _pickle_stored_qpixmap(self):
        """
                        This class is a mock class so when used in an subclass that implement __getstate__ __setstate__
                        customize pickling it's own customized __getstate__  __setstate__ won't be called
                        Therefore the  subclass mist call this method from within it's own __getstate__
                        :return:
                        """

        # QPixmap is not pickable so let's transform it into QByteArray that does support pickle
        state = {}
        for key, value in self._qpixmap.items():
            qbyte_array = QByteArray()
            stream = QDataStream(qbyte_array, QIODevice.WriteOnly)
            stream << value
            state[key] = qbyte_array
        return state

    def __setstate__(self, state):
        return self._unpickle_stored_qpixmap(state)

    def _unpickle_stored_qpixmap(self, state):
        """
                        This class is a mock class so when used in an subclass that implement __setstate__
                        customize pickling it's own customized __setstate__ won't be called
                        Therefore the  subclass mist call this method from within it's own __setstate__
                        :return:
                        """
        self._qpixmap = {}
        # retrieve a QByteArray and transform it into QPixmap
        for key, buffer in state.items():
            qpixmap = QPixmap()
            stream = QDataStream(buffer, QIODevice.ReadOnly)
            stream >> qpixmap
            self._qpixmap[key] = qpixmap

    def raz_qpixmap_dict(self):
        self._qpixmap = {}


class Photo(StoreQPixmap):

    _matplotlib_image_preview_size = (600, 400)

    @classmethod
    def set_matplotlib_image_preview_size(cls, width, height):
        __class__._matplotlib_image_preview_size = (int(abs(width)), int(abs(height)))

    @classmethod
    def get_matplotlib_image_preview_size(cls):
        return cls._matplotlib_image_preview_size

    @staticmethod
    def exif_info_2_time(ts):
        """changes EXIF date ('2005:10:20 23:22:28') to number of seconds since 1970-01-01"""
        tpl = time.strptime(ts + 'UTC', '%Y:%m:%d %H:%M:%S%Z')
        return time.mktime(tpl)

    def __init__(self, file_name, metadata):
        super().__init__()
        self.file_name = file_name
        # store all existing metadata in the file as return by exiftool
        # Dictionary of tag/values
        self.exif_metadata = metadata
        # compute capture time in numeric time format out of exif string format - avoid to compute multiple times later
        self._shot_time = __class__.exif_info_2_time(self.exif_metadata["EXIF:CreateDate"])
        # store matploblib ready image preview (numpy RGB)  so as to speed up display in interface
        self._matplotlib_image_preview = None
        # store full size image in opencv format (numpy BGR)
        self._opencv_image_fullsize = None
        # store reference to the CloneSet for clone picture following this picture
        self.clone_set_with_next = None
        # store reference to the CloneSet for clone picture preceeding this picture
        self.clone_set_with_previous = None

    @property
    def shot_timestamp(self):
        return self._shot_time

    def get_matplotlib_image_preview(self):
        logger.error("NotImplementedError")
        raise NotImplementedError

    def get_opencv_image_fullsize(self):
        """
        Return the full definition image in an opencv mumpy BGR format if the mime/type is implemented
        else it returns False

        :return:
        """
        logger.error("NotImplementedError")
        raise NotImplementedError

    def get_tag_value(self, list_of_tag_synonyms=[]):
        '''
        return value for the first tag that matches in the list

        :param list_of_tag_synonyms: a list containing tags that will be interpreted in order. If no value for first
                                     one then value for second is searched, and so on and so forth till end of list
                                     This is implemented to accommodate at least the case of low ISO (lower than 100)
                                     on Nikons where EXIF:ISO is not provided and replaced by MakerNotes:ISO. This is
                                     odd but that's the way it is.

        :return: value for the first tag in the list that is valued in the PhotoWithMetadata object or "NA" if tags not found
        '''
        # TODO migth consider to raise an exception and abort rather than returning NA for instance if EXIF:CaptureDate
        # TODO is not there, interval will not be computable and the whole thing will crash
        result = "NA"
        for i in range(len(list_of_tag_synonyms)):
            try:
                result = self.exif_metadata[list_of_tag_synonyms[i]]
            except KeyError:  # no tag for the first tag value - try other values if more alternatives are provided
                pass

        return result

    def __lt__(self, other):
        logger.error("NotImplementedError")
        raise NotImplementedError

    def __getstate__(self):
        logger.error("NotImplementedError")
        raise NotImplementedError

    def __setstate__(self, state):
        logger.error("NotImplementedError")
        raise NotImplementedError


class PhotoFromVideo(Photo):

    def __init__(self, file_name, metadata, video_file_path, index_in_video):
        super().__init__(file_name, metadata)
        self._video_file_path = video_file_path
        self._index_in_video = index_in_video

        if __debug__:
            logger.debug('PhotoFromVideo Object istanciation done for %s at index %s',
                         str(file_name), str(index_in_video))
        return

    def get_matplotlib_image_preview(self):

        if not isinstance(self._matplotlib_image_preview, np.ndarray):
            cap = cv2.VideoCapture(self._video_file_path)
            cap.set(cv2.CAP_PROP_POS_FRAMES, self._index_in_video)
            i = 0
            while True:
                ret, frame = cap.read()
                if ret:
                    break
                else:
                    i += 1
                    if i > 10:
                        logger.error("cv2.VideoCapture keep not delivering image after 10 attempt")
                        raise ValueError

            cap.release()
            width, heigth = __class__._matplotlib_image_preview_size
            image_resized = cv2.resize(frame, (width, heigth),
                                       interpolation=cv2.INTER_AREA)  # /!\ THIS IS CPU INTENSIVE IF image_cv IS BIG
            # TODO THIS IS NOT PROTECTING THE ORIGINAL IMAGE RATIO
            # transform opencv BGR in RGB as supported by Qt images
            self._matplotlib_image_preview = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)

        return self._matplotlib_image_preview

    def get_opencv_image_fullsize(self):
        """
        Return the full definition image in an opencv mumpy BGR format
        else it returns False

        :return:
        """
        if self._opencv_image_fullsize is None:
            cap = cv2.VideoCapture(self._video_file_path)
            cap.set(cv2.CAP_PROP_POS_FRAMES, self._index_in_video)
            ret, self._opencv_image_fullsize = cap.read()
            if not ret:
                return False
            cap.release()

        return self._opencv_image_fullsize


    def __getstate__(self):

        # TODO DEBUG REMOVE REMOVE
        # for key in self.__dict__.keys():
        #     print(key)

        state = {}

        state["file_name"] = self.file_name
        state["exif_metadata"] = self.exif_metadata
        state["_shot_time"] = self._shot_time
        state["_matplotlib_image_preview"] = self._matplotlib_image_preview
        state["_video_file_path"] = self._video_file_path
        state["_index_in_video"] = self._index_in_video
        state["clone_set_with_next"] = self.clone_set_with_next
        state["clone_set_with_previous"] = self.clone_set_with_previous

        # then call pickle for StoreQpixmap moke class
        store_qpixmap_state = self._pickle_stored_qpixmap()
        for key, value in store_qpixmap_state.items():
            state[key] = value

        return state

    def __setstate__(self, state):

        self.file_name = state["file_name"]
        del state["file_name"]
        self.exif_metadata = state["exif_metadata"]
        del state["exif_metadata"]
        self._shot_time = state["_shot_time"]
        del state["_shot_time"]
        self._matplotlib_image_preview = state["_matplotlib_image_preview"]
        del state["_matplotlib_image_preview"]
        self._video_file_path = state["_video_file_path"]
        del state["_video_file_path"]
        self._index_in_video = state["_index_in_video"]
        del state["_index_in_video"]
        self.clone_set_with_next = state["clone_set_with_next"]
        del state["clone_set_with_next"]
        self.clone_set_with_previous = state["clone_set_with_previous"]
        del state["clone_set_with_previous"]

        # then call unpickle for StoreQpixmap moke class - what remains in state is for StoreQpixmap
        self._unpickle_stored_qpixmap(state)

        return


class PhotoWithMetadata(Photo):
    _SUPPORTED_MIME_TYPES = {
        "image/x-nikon-nef",  # NIKON NEF files
        "image/jpeg"  # JPEG
    }

    _EXIF_KEYWORD_REFERENCING_OFFSET_AND_LENGTH_OF_PREVIEW_PER_MIME_FILE_TYPE_DICT = {
        # "image/x-nikon-nef": ["EXIF:JpgFromRawStart", "EXIF:JpgFromRawLength"],  # heavy one - resize takes too long
        "image/x-nikon-nef": ["EXIF:OtherImageStart", "EXIF:OtherImageLength"],  # lighter - preferred
        "image/jpeg": ["EXIF:ThumbnailOffset", "EXIF:ThumbnailLength"]
    }


    # # TODO no longer used kept for future code improvment
    # RAW_FILE_TYPES = {
    #     "NEF",  # NIKON
    #     "CR2",  # CANON
    #     "DNG",  # LEICA
    #     "RAF",  # FUJI
    #     "SR2"  # SONY
    #     #  "3FR"   HASSELBLAD
    # }
    # # TODO no longer used kept for future code improvment
    # EXIF_KEYWORD_REFERENCING_PREVIEW_IMAGE_PER_RAW_FILE_TYPE_DICT = {
    #     "NEF": "EXIF:JpgFromRaw",
    #     "CR2": "EXIF:PreviewImage",
    #     "DNG": "EXIF:PreviewImage",
    #     "RAF": "EXIF:ThumbnailImage",
    #     "SR2": "EXIF:PreviewImage"
    #     #  "3FR" : "EXIF:ThumbnailTIFF" Tiff doesn't work
    # }
    # # TODO no longer used kept for future code improvment
    # JPEG_FILE_TYPES = {"JPEG"}


    @classmethod
    def get_set_of_supported_mime_types(cls):
        return cls._SUPPORTED_MIME_TYPES

    @staticmethod
    def _create_opencv_image_from_bytesio(img_stream, cv2_img_flag=0):
        """
        return an opencv image from a JPEG binary data
        :param img_stream: a io.BytesIO stream on the binary data extracted from the EXI:JpgFromRaw metadata
                           of the RAW file
        :param cv2_img_flag: -> 0: gray -1(negative):BGR color
        :return: an opencv3 image that is a numpy array in BGR format
        """
        img_stream.seek(0)
        img_array = np.asarray(bytearray(img_stream.read()), dtype=np.uint8)
        return cv2.imdecode(img_array, cv2_img_flag)  # cv2_imb_flag -> 0: gray | -1(negative):BGR color

    def __init__(self, file_name, metadata):
        super().__init__(file_name, metadata)

        logger.debug('PhotoWithMetadata Object istanciation started for %s', str(file_name))

        # stores binary image preview whatever the file type so size can be very different (NEF vs JPEG Thumbnail)
        self._binary_image_preview = None
        # to be used in any method that updates values so that multi threading is safe
        # self._lock = threading.Lock()
        # for tag, value in self.exif_metadata.items():
        #     logger.debug(' tag %s has value %s', tag, value)
        # logger.debug('PhotoWithMetadata Object istanciation completed  for %s', file_name)
        # TODO implement some minimal check that mandatory tags are presents else fail - how to fail an __init__ ?
        return

    # def set_shot_timestamp(self,
    #                        shot_time):  # TODO NOT USED MIGHT BE REMOVED as timestamp for real photos should not change
    #     self._shot_time = shot_time

    def get_binary_image_preview(self):
        """

        :return: binary image or False if MIMEType not supported
        """
        if self.exif_metadata["File:MIMEType"] not in __class__._SUPPORTED_MIME_TYPES:
            return False

        if self._binary_image_preview is None:
            start_position_in_file = \
                __class__._EXIF_KEYWORD_REFERENCING_OFFSET_AND_LENGTH_OF_PREVIEW_PER_MIME_FILE_TYPE_DICT[
                    self.exif_metadata["File:MIMEType"]
                ][0]
            length_of_preview_image = \
                __class__._EXIF_KEYWORD_REFERENCING_OFFSET_AND_LENGTH_OF_PREVIEW_PER_MIME_FILE_TYPE_DICT[
                    self.exif_metadata["File:MIMEType"]
                ][1]

            with open(self.file_name, 'rb') as f:
                f.seek(self.exif_metadata[start_position_in_file])
                self._binary_image_preview = f.read(self.exif_metadata[length_of_preview_image])

        return self._binary_image_preview

    def get_matplotlib_image_preview(self):

        # return False if binary image preview can't be loaded
        if not self.get_binary_image_preview():
            return False

        if not isinstance(self._matplotlib_image_preview, np.ndarray):
            img_cv2 = __class__._create_opencv_image_from_bytesio(BytesIO(self.get_binary_image_preview()), -1)
            width, heigth = __class__._matplotlib_image_preview_size
            image_resized = cv2.resize(img_cv2, (width, heigth),
                                       interpolation=cv2.INTER_AREA)  # /!\ THIS IS CPU INTENSIVE IF image_cv IS BIG
            # TODO THIS IS NOT PROTECTING THE ORIGINAL IMAGE RATIO
            # transform opencv BGR in RGB as supported by Qt images
            self._matplotlib_image_preview = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)

        return self._matplotlib_image_preview

    def get_opencv_image_fullsize(self):
        """
        Return the full definition image in an opencv mumpy BGR format if the mime/type is implemented
        else it returns False

        :return:
        """
        # TODO implemntation with NEF file is returning the preview 120x160 image, not the full size one
        # TODO opencv doesn't seem to support raw file reading - has to be implemented with some libraw python module
        def downsample(m):
            """
            Simple demosaicing by taking red and blue pixel and average of green ones. Simple but loss of pixels
            """
            r = m[0::2, 0::2]
            g = np.clip(m[0::2, 1::2] // 2 + m[1::2, 0::2] // 2,
                        0, 2 ** 14 - 1)
            b = m[1::2, 1::2]

            return np.dstack([r, g, b])

        if self._opencv_image_fullsize is None:

            if self.exif_metadata["File:MIMEType"] not in __class__._SUPPORTED_MIME_TYPES:
                return False

            elif self.exif_metadata["File:MIMEType"] == "image/jpeg":
                self._opencv_image_fullsize = cv2.imread(self.file_name, cv2.IMREAD_COLOR)

            elif self.exif_metadata["File:MIMEType"] == "image/x-nikon-nef":

                with rawpy.imread(self.file_name) as raw:
                    # rgb = raw.postprocess(use_auto_wb=True, no_auto_bright=True)
                    rgb = raw.postprocess(no_auto_bright=True, use_camera_wb=True)

                    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

                    if __debug__:
                        print(logger.debug("shape of bdr image is %s", str(bgr.shape)))

                    # try:
                    #     # rgb = raw.postprocess(use_camera_wb=True, output_color=(sRGB=1), gamma=(2.222, 4.5))
                    #     rgb = raw.postprocess(use_camera_wb=True, gamma=(2.222, 4.5))
                    # except Exception as e:
                    #     print(exception_to_string(e))
                    #
                    # print(rgb.shape)

                    self._opencv_image_fullsize = bgr

                # proc = libraw.LibRaw()
                # proc.open_file(self.file_name)
                # proc.unpack()
                # mosaic = proc.imgdata.rawdata.raw_image
                #
                # # mapping to linear values
                # lin_lut = proc.imgdata.color.curve  # linearisation look up table
                # mosaic = lin_lut[mosaic]
                #
                # black = proc.imgdata.color.black
                # saturation = proc.imgdata.color.maximum
                # mosaic -= black    # black substraction
                #
                # uint14_max = 2**14 - 1
                # mosaic *= int(uint14_max / (saturation - black))
                # mosaic = np.clip(mosaic, 0, uint14_max)      #clip to range
                #
                # # white balancing
                # assert(proc.imgdata.idata.cdesc == b"RGBG")
                #
                # cam_mul = proc.imgdata.color.cam_mul  # RGB camera multiplier
                # cam_mul /= cam_mul[1]                 # scale green to 1
                # mosaic[0::2, 0::2] *= cam_mul[0]      # scale reds
                # mosaic[1::2, 1::2] *= cam_mul[2]      # scale blues
                # mosaic = np.clip(mosaic, 0, uint14_max)      # clip to range
                #
                # # demosaicing
                # img = downsample(mosaic)
                #
                # # colour space conversion
                # cam2sgrb = proc.imgdata.color.rgb_cam[:, 0:3]
                # cam2sgrb = np.round(cam2sgrb * 255).astype(np.int16)
                # img = img // 2**6   # reduce dynamic range to 8bpp from 14bpp - 14 - 8 = 6
                # shape = img.shape
                # pixels = img.reshape(-1, 3).T    # 3xN array of picels
                # pixels = cam2sgrb.dot(pixels) // 255
                # img = pixels.T.reshape(shape)
                # img = np.clip(img, 0, 255).astype(np.uint8)
                #
                # # gamma correction
                # gcurve = [(i/255) ** (1/2.2) for i in range(256)]
                # gcurve = np.array(gcurve * 255, dtype=np.uint8)
                # img = gcurve[img]
                #
                # img_cv2 = img

        return self._opencv_image_fullsize

    # TODO should this sort be maintained ? if i want the container class to be orderd on different tag the
    # TODO ordering should be managed in the container PhotoCollection and not in thePhoto itself that should
    # TODO remain agnostic vis-a-vis the sorting criteria
    def __lt__(self, other):  # implementation required so that sort() can be used
        return self._shot_time < other.shotTime

    def __repr__(self):
        _string = ("FILE NAME = " + str(self.file_name) + "\n")
        _string += ("PICTURE capture timestamp ="
                    + str(datetime.datetime.fromtimestamp(self._shot_time).strftime('%Y-%m-%d %H:%M:%S')) + "\n")
        _string += ("PICTURE TAG VALUES : \n")
        for tag in self.exif_metadata.keys():
            _string += ("    {} = {}".format(tag, self.exif_metadata[tag]))
            _string += "\n"
        return _string

    def __getstate__(self):

        # TODO DEBUG REMOVE REMOVE
        # for key in self.__dict__.keys():
        #     print(key)

        state = {}

        state["file_name"] = self.file_name
        state["exif_metadata"] = self.exif_metadata
        state["_shot_time"] = self._shot_time
        state["_binary_image_preview"] = self._binary_image_preview
        state["_matplotlib_image_preview"] = self._matplotlib_image_preview
        state["clone_set_with_next"] = self.clone_set_with_next
        state["clone_set_with_previous"] = self.clone_set_with_previous

        # then call pickle for StoreQpixmap moke class
        store_qpixmap_state = self._pickle_stored_qpixmap()
        for key, value in store_qpixmap_state.items():
            state[key] = value

        return state

    def __setstate__(self, state):

        self.file_name = state["file_name"]
        del state["file_name"]
        self.exif_metadata = state["exif_metadata"]
        del state["exif_metadata"]
        self._shot_time = state["_shot_time"]
        del state["_shot_time"]
        self._binary_image_preview = state["_binary_image_preview"]
        del state["_binary_image_preview"]
        self._matplotlib_image_preview = state["_matplotlib_image_preview"]
        del state["_matplotlib_image_preview"]
        self.clone_set_with_next = state["clone_set_with_next"]
        del state["clone_set_with_next"]
        self.clone_set_with_previous = state["clone_set_with_previous"]
        del state["clone_set_with_previous"]

        # then call unpickle for StoreQpixmap moke class - what remains in state is for StoreQpixmap
        self._unpickle_stored_qpixmap(state)

        return


class CloneSet:
    """
    This class supports the data describing a set of cloned picture between two real picture. It has reference to
    previous and next real picture, list of clone picture and parameters and function to be used to generate images
    from clones at a later stage. These latter parameters are manage via a link to an instanceof DuplicateMethod
    subclasses.

    """

    def __init__(self, previous_picture, next_picture, duplicate_method):
        self.previous_picture = previous_picture  # reference of previous picture by order of shooting time
        self.next_picture = next_picture  # reference of next picture by order of shooting time
        self.list_of_clones = []  # list of clone pictures created out of the previous and next
        # create an instance of duplicate_methode_set an link it to the clone_set
        self.duplicate_method_set = DuplicateMethod.create_duplicate_method_subclass(duplicate_method)

    def __len__(self):
        return len(self.list_of_clones)

    @property
    def duplicate_method(self):
        return self.duplicate_method_set.duplicate_method

    @property
    def computation_parameters(self):
        return self.duplicate_method_set.computation_parameters

    def __repr__(self):
        _string = ("PREVIOUS PICTURE FILE NAME = " + str(self.previous_picture.file_name) + "\n")
        _string += ("PREVIOUS PICTURE capture timestamp ="
                    + str(datetime.datetime.fromtimestamp(
                    self.previous_picture._shot_time
                ).strftime('%Y-%m-%d %H:%M:%S')) + "\n")
        _string += ("NEXT PICTURE FILE NAME = " + str(self.next_picture.file_name) + "\n")
        _string += ("NEXT PICTURE capture timestamp ="
                    + str(datetime.datetime.fromtimestamp(self.next_picture._shot_time).strftime('%Y-%m-%d %H:%M:%S'))
                    + "\n")
        _string += ("LIST OF CLONES : \n")
        for picture in self.list_of_clones:
            _string += (" {}".format(str(picture.file_name)))
            _string += "\n"
        return _string


class DuplicateMethod:
    """
    This is an abstract class that aims at being subclassed to implement the various duplicated method supported
    by the program
    Every instance of a subclass will provide method to duplicate images but also the set of parameters to be used
    for the computation. It will also provide the
    It also hosts the reference list of supported methods as a class attribute

    TODO it will replace the cloneset per duplicate method as a generic pointer to
    """

    supported_duplicate_method_dict = {"gunner_farnerback": "Dense optical flow Two-Frame Motion Estimation \
                                                           Based on Polynomial Expansion",
                                       "lucas_kanade": "blabla...",
                                       "interpolate": "blabla...",
                                       "simple_copy": "blabla..."}

    @classmethod
    def get_duplicate_method_supported_list(cls):
        return __class__.supported_duplicate_method_dict.keys()

    @classmethod
    def create_duplicate_method_subclass(cls, duplicate_method):
        if duplicate_method == "simple_copy":
            return DuplicateSimpleCopy()
        if duplicate_method == "interpolate":
            return DuplicateInterpolate()
        if duplicate_method == "lucas_kanade":
            return DuplicateLucasKanade()
        if duplicate_method == "gunner_farnerback":
            return DuplicateGunnerFarnerback()

    def __init__(self, ):
        self._method_name = None

    @property
    def duplicate_method(self):
        return self._method_name

    @property
    def computation_parameters(self):
        logger.error("NotImplementedError")
        raise NotImplementedError

    def create_transition_image(self, img_before, img_after, nb_intervals=None, interval_rank=None):
        logger.error("NotImplementedError")
        raise NotImplementedError


class DuplicateSimpleCopy(DuplicateMethod):
    COMPUTATION_PARAMETERS_SET = {"duplicate", "successor"}

    def __init__(self):
        super().__init__()

        self._method_name = "simple_copy"
        self._copy_from = "previous"

    @property
    def computation_parameters(self):
        return self._copy_from

    def create_transition_image(self, img_before, img_after, nb_intervals=None, interval_rank=None):

        if self._copy_from == "previous":
            return img_before
        elif self._copy_from == "successor":
            return img_after
        else:
            raise ValueError


class DuplicateInterpolate(DuplicateMethod):

    def __init__(self, ):
        super().__init__()

        self._method_name = "interpolate"

    @property
    def computation_parameters(self):
        return None

    def create_transition_image(self, img_before, img_after, nb_intervals, interval_rank):
        alpha = interval_rank / nb_intervals
        return cv2.addWeighted(img_before, 1 - alpha, img_after, alpha, 0)


class DuplicateLucasKanade(DuplicateMethod):

    def _move_blob(image_src, image_dest, coord_src, coord_end, nb_intervals, interval_rank,
                   length=150, radius_max=20):
        """
        receives two images taken in sequence and generate an intermediate image based on Shi-Tomasi Corner Detector and
        Lukas-Kanade Optical Flow method implemented in opencv
        (see https://docs.opencv.org/3.4.0/d7/d8b/tutorial_py_lucas_kanade.html)


        :param image_src: an numpy array image as generated by opencv2
        :param image_dest:an numpy array image as generated by opencv2
        :param coord_src: a numpy array as generated by the opencv function cv2.goodFeaturesToTrack which value have
                          been converted to integers via np.int0
        :param coord_end: a numpy array as generated by the opencv function cv2.calcOpticalFlowPyrLK which value have
                          been converted to integers via np.int0
        :nb_intervals: number of interval between the two real pictures
        :interval_rank: position of the current interval in the sequence of intervals between the two real pictures
                        Starts at 1 not at 0 as usual in Python position
        :param length: nb of pixel to move around the good features coordinates
        :param radius_max : maximum distance between the coord_src and coord_end within which we copy the blob.
                            if distance is > radius_max no copy is made
        :return: a new image for which the feature point that moved fof more than 1 pixels are moved. New image is based
                 on src image. the place of initial feature is set to the value in dest image and vice versa
        """

        # TODO ignore data input validation for time being
        if image_src.shape != image_dest.shape:
            logger.error("image_src and image_dest do not have same dimension")
            raise ValueError

        # record image size
        x_max, y_max, c = image_src.shape

        # initialize new image with image source
        image_new = np.copy(image_src)

        # compute coordinates of intervals points. Thank you numpy !
        corners_midpoint = np.int0((coord_src * (nb_intervals - interval_rank) + coord_end * interval_rank)
                                   / nb_intervals)

        nb_of_points = len(coord_src)
        nb_of_moves = 0
        distance_list = []
        for (src, end, target) in zip(coord_src, coord_end, corners_midpoint):

            y_src, x_src = src.ravel()
            y_end, x_end = end.ravel()
            y_target, x_target = target.ravel()

            # points with a move of less than 1 and located in border (of length width) are not treated
            distance = ((x_src - x_end) ** 2 + (y_src - y_end) ** 2) ** 0.5
            distance_list.append(distance)

            if distance <= 1 \
                    or distance > radius_max \
                    or (x_src < length) \
                    or (y_src < length) \
                    or (x_max - x_src < length) \
                    or (y_max - y_src < length) \
                    or (x_end < length) \
                    or (y_end < length) \
                    or (x_max - x_end < length) \
                    or (y_max - y_end < length) \
                    :
                continue
            nb_of_moves += 1

            # generate_computed_pictures blob to be moved on source image at location of source corner
            blob = np.zeros((2 * length, 2 * length, 3), np.uint8)
            # blob[:, :] = image_src[x_src - length:x_src + length, y_src - length:y_src + length]
            blob[:, :] = image_dest[x_end - length:x_end + length, y_end - length:y_end + length]

            # replace blob at source location by it's footprint on destination image
            image_new[x_src - length:x_src + length, y_src - length:y_src + length] = \
                image_dest[x_src - length:x_src + length, y_src - length:y_src + length]

            # copy blob at target interpolated point
            image_new[x_target - length:x_target + length, y_target - length:y_target + length] = blob[:, :]

        logger.debug(" %s blobs copied - nb intervals = %s - interval rank = %s", str(nb_of_moves),
                    str(nb_intervals), str(interval_rank))
        # distance_list = sorted(distance_list, reverse=True)
        # print(str(nb_of_moves) + " point moved out of " + str(nb_of_points) + " detected")
        # for i in range(50):
        #     print(distance_list[i])

        # print("size of img_new", image_new.shape)
        return image_new
    def __init__(self,
                 ShiTomasi_maxCorners=200000,
                 ShiTomasi_qualityLevel=0.01,
                 ShiTomasi_minDistance=7,
                 ShiTomasi_blockSize=7,
                 LukasKanade_winSize=(15, 15),
                 LukasKanade_maxLevel=2,
                 LukasKanade_criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
                 ):
        super().__init__()
        self._method_name = "lucas_kanade"

        self.FEATURES_PARAMS = dict(maxCorners=ShiTomasi_maxCorners,
                                    qualityLevel=ShiTomasi_qualityLevel,  # minimum quality level between 0 and 1
                                    minDistance=ShiTomasi_minDistance,  # minimum distance between corners
                                    blockSize=ShiTomasi_blockSize)

        self.LK_PARAMS = dict(winSize=LukasKanade_winSize,
                              maxLevel=LukasKanade_maxLevel,
                              criteria=LukasKanade_criteria)

    @property
    def computation_parameters(self):
        return (self.FEATURES_PARAMS, self.LK_PARAMS)

    def create_transition_image(self, img_before, img_after, nb_intervals, interval_rank):

        # Detect Shi-Tomasi corners as per current parameters
        img_before_gray = cv2.cvtColor(img_before, cv2.COLOR_BGR2GRAY)
        corners = cv2.goodFeaturesToTrack(img_before_gray, mask=None, **self.FEATURES_PARAMS)
        corners_ = np.int0(corners)

        img_after_gray = cv2.cvtColor(img_after, cv2.COLOR_BGR2GRAY)
        corners_after, st, err = cv2.calcOpticalFlowPyrLK(img_before_gray, img_after_gray, corners, None,
                                                          **self.LK_PARAMS)
        corners_after_ = np.int0(corners_after)

        # select only corners which flow can be found in img_after_gray - st==1 in that case
        good_corners_before = corners_[st == 1]
        good_corners_after = corners_after_[st == 1]

        try:
            img_with_blobs_updated = __class__._move_blob(img_before, img_after,
                                                          good_corners_before, good_corners_after,
                                                          nb_intervals, interval_rank,
                                                          15)  # <== nb of pixels moved around the point
        except Exception as e:
            print(exception_to_string(e))

        return img_with_blobs_updated


class DuplicateGunnerFarnerback(DuplicateMethod):

    def __init__(self,
                 ShiTomasi_maxCorners=200000,
                 ShiTomasi_qualityLevel=0.01,
                 ShiTomasi_minDistance=7,
                 ShiTomasi_blockSize=7,
                 LukasKanade_winSize=(15, 15),
                 LukasKanade_maxLevel=2,
                 LukasKanade_criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
                 ):
        super().__init__()
        self._method_name = "gunner_farnerback"

        self.FEATURES_PARAMS = dict(maxCorners=ShiTomasi_maxCorners,
                                    qualityLevel=ShiTomasi_qualityLevel,  # minimum quality level between 0 and 1
                                    minDistance=ShiTomasi_minDistance,  # minimum distance between corners
                                    blockSize=ShiTomasi_blockSize)

        self.LK_PARAMS = dict(winSize=LukasKanade_winSize,
                              maxLevel=LukasKanade_maxLevel,
                              criteria=LukasKanade_criteria)

    @property
    def computation_parameters(self):
        return (self.FEATURES_PARAMS, self.LK_PARAMS)

    def create_transition_image(self, img_before, img_after, nb_intervals, interval_rank):

        img_before_gray = cv2.cvtColor(img_before, cv2.COLOR_BGR2GRAY)
        img_after_gray = cv2.cvtColor(img_after, cv2.COLOR_BGR2GRAY)

        flow = cv2.calcOpticalFlowFarneback(img_before_gray, img_after_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)

        flow_weigthed = flow * (interval_rank / nb_intervals)
        move_instruction = flow_weigthed.astype(np.int8)  # flow is float convert to int8

        nb_rows, nb_cols, _ = img_before.shape
        all_inds = np.where(np.ones((nb_rows, nb_cols), dtype=np.int8))
        moves = move_instruction[all_inds]
        new_r = all_inds[0] + moves[..., 0]
        new_c = all_inds[1] + moves[..., 1]

        # Filter for invalid shifts
        filter = (new_r < 0) + (new_r >= nb_rows - 1) + (new_c < 0) + (new_c >= nb_cols - 1)
        new_r[filter] = all_inds[0][filter]  # This just recovers the original non-moved index
        new_c[filter] = all_inds[1][filter]

        new_inds = (new_r, new_c)  # or the other way round (new_c, new_r) ?
        img_computed = np.copy(img_before)
        img_computed[new_inds] = img_before[all_inds]

        # for y in range(nb_rows):
        #     for x in range(nb_cols):
        #         try:
        #             # img_computed[y + flow_int[y][x][1]][x + flow_int[y][x][0]] = img_before[y][x]
        #             img_computed[x + flow_int[x][y][0]][y + flow_int[x][y][1]] = img_before[x][y]
        #         except IndexError:
        #             logger.warning("out of scope copy row %s col %s", str(y), str(x))
        #             continue
        # for x in range(nb_cols):
        #     for y in range(nb_rows):
        #         try:
        #             x_computed = x + flow_int[y][x][1]
        #             y_computed = y + flow_int[y][x][0]
        #             if x_computed < 0:
        #                 logger.warning("x_computed value %s is negative for x value %s and y value %s\
        #                                for flow values %s", str(x_computed), str(x), str(y),str(flow_int[x][y]))
        #                 continue
        #             elif x_computed > nb_cols - 1:
        #                 logger.warning("x_computed value %s is out of range for x value %s and y value %s\
        #                                for flow values %s", str(x_computed), str(x), str(y), str(flow_int[x][y]))
        #                 continue
        #             elif y_computed < 0:
        #                 logger.warning("y_computed value %s is negative for x value %s and y value %s\
        #                                for flow values %s", str(y_computed), str(x), str(y), str(flow_int[x][y]))
        #                 continue
        #             elif y_computed > nb_rows - 1:
        #                 logger.warning("y_computed value %s is out of range for x value %s and y value %s\
        #                                for flow values %s", str(y_computed), str(x), str(y), str(flow_int[x][y]))
        #                 continue
        #             else:
        #                 img_computed[y_computed][x_computed] = img_before[y][x]
        #         except IndexError:
        #             logger.warning("out of scope copy row %s col %s ", str(y), str(x))
        #             continue
        # print(flow)
        # print(flow_weigthed)
        # print(flow_int)


        # try:
        #     x_src, y_src = coord_src.ravel()
        #     x_end, y_end = coord_dest.ravel()
        # except Exception as e:
        #     print(exception_to_string(e))
        #
        # distance = ((x_src - x_end) ** 2 + (y_src - y_end) ** 2) ** 0.5
        # logger.info("shape of distance is :%s", str(distance.shape))
        # print(distance)

        # select only corners which flow can be found in img_after_gray - st==1 in that case

        # try:
        #     img_with_blobs_updated = __class__._move_blob(img_before, img_after,
        #                                                   good_corners_before, good_corners_after,
        #                                                   nb_intervals, interval_rank,
        #                                                   15)  # <== nb of pixels moved around the point
        # except Exception as e:
        #     print(exception_to_string(e))

        return img_computed


class PhotoCloned(StoreQPixmap):
    """
    PhotoWithMetadata cloned from the PhotoWithMetadata class but not yet created on disk
    PhotoWithMetadata is created based on the CloneSet instance passed as first parameter

    :parameter

     cloned_set : reference to an instance of the CloneSet heritated class to be used for the cloned PhotoWithMetadata creation

     precedent : reference of preceding picture - needed to compute timestamp of the PhotoCloned instance to be created

     successor : reference of following picture - needed to compute timestamp of the PhotoCloned instance to be created

    Provide exact same method as the genuine photo class
    """

    # This is the reference list of supported picture clone method together with their clone_set Class name
    # This dict is used as reference for supported duplicate method and for their associated CloneSet class
    # clone_set_classes_dict_per_clone_method = {"lucas_kanade": CloneSetLucasKanade,
    #                                            "interpolate": CloneSetInterpolate,
    #                                            "simple_copy": CloneSetSimpleCopy}

    # @classmethod
    # def get_clone_method_supported_list(cls):
    #     return __class__.clone_set_classes_dict_per_clone_method.keys()

    # @classmethod
    # def get_duplicate_method_from_class(cls, clone_set_subclass):
    #     rc = None
    #     for duplicate_method, class_reference in cls.clone_set_classes_dict_per_clone_method.items():
    #         if class_reference == clone_set_subclass:
    #             rc = duplicate_method
    #     return rc

    def __init__(self, clone_set, precedent, successor):
        super().__init__()

        self.clone_set = clone_set
        # filename derived from original file by adding _clone  before the dot preceeding the suffix
        # TODO ensure new clone name is unique for all clones of  given Physical picture the random below
        # TODO is quick and dirty fixing - uniqueId should be provide from outside the class
        if len(self.clone_set.previous_picture.file_name.split(".")) == 1:   # video frame fake file name have  no suffix
            self.file_name \
                = self.clone_set.previous_picture.file_name.split(".")[-1] \
                  + "_" \
                  + str(len(clone_set.list_of_clones) + 1)
        else:   # whereas photo file do have a suffix (.jPG, .NEF,...)
            self.file_name \
                = self.clone_set.previous_picture.file_name.split(".")[-2] \
                  + "_" \
                  + str(len(clone_set.list_of_clones) + 1) \
                  + "." \
                  + self.clone_set.previous_picture.file_name.split(".")[-1]
        # set timestamp to the middle of interval between preceding and following pictures
        self._shot_time = precedent.shot_timestamp \
                          + (successor.shot_timestamp - precedent.shot_timestamp) / 2

    @property
    def shot_timestamp(self):
        return self._shot_time

    def set_shot_timestamp(self, shot_time):
        self._shot_time = shot_time

    @property
    def duplicate_method(self):
        return self.clone_set.duplicate_method

    def update_duplicate_method(self, new_duplicate_method):
        """
        if the new duplicate method is different from the current one then a new clone_set is created
        with the default duplicate_method parameter and applied to all picture that are part of the same
        clone_set
        :param new_duplicate_method: new duplicate method as described in exclusive_button_group.checkedButton()
        :return: True if OK False otherwise
        """
        if new_duplicate_method not in DuplicateMethod.get_duplicate_method_supported_list():
            logger.error("PhotoCloned.update_duplicate_method() called with non supported duplicate method: %s"
                         , str(new_duplicate_method))
            raise ValueError

        if self.duplicate_method != new_duplicate_method:
            # create a new clone_set of new duplicate method
            self.clone_set.duplicate_method_set \
                = DuplicateMethod.create_duplicate_method_subclass(new_duplicate_method)
        else:
            pass

        return True

    def get_binary_image_preview(self):
        return self.clone_set.previous_picture.get_binary_image_preview()

    def get_matplotlib_image_preview(self):
        return self.clone_set.previous_picture.get_matplotlib_image_preview()

    def get_tag_value(self, list_of_tag_synonyms=[]):
        '''
        return value for the first tag that matches in the list by calling the same method on the PhotoWithMetadata instance
        from which this virtual photo is cloned from.
        Except for filename and create date that are not the same and return from this method
        '''
        if list_of_tag_synonyms[0] == "SourceFile":
            return self.file_name
        elif list_of_tag_synonyms[0] == "EXIF:CreateDate":
            return datetime.datetime.fromtimestamp(self._shot_time).strftime('%Y:%m:%d %H:%M:%S')
        else:
            return self.clone_set.previous_picture.get_tag_value(list_of_tag_synonyms)

    def __getstate__(self):

        state = {}

        state["clone_set"] = self.clone_set
        state["file_name"] = self.file_name
        state["_shot_time"] = self._shot_time

        # then call pickle for StoreQpixmap moke class
        store_qpixmap_state = self._pickle_stored_qpixmap()
        for key, value in store_qpixmap_state.items():
            state[key] = value

        return state

    def __setstate__(self, state):

        self.clone_set = state["clone_set"]
        del state["clone_set"]
        self.file_name = state["file_name"]
        del state["file_name"]
        self._shot_time = state["_shot_time"]
        del state["_shot_time"]

        # then call unpickle for StoreQpixmap moke class - what remains in state is for StoreQpixmap
        self._unpickle_stored_qpixmap(state)

        return

    # TODO implement __repr__


class PhotoCollection:
    '''
    Container for instance of the PhotoWithMetadata class

    This class is not to be used it is there to implement common methods for classes inheritating from it:
     - PhotoNonOrderedCollection
     - PhotoOrderedCollectionByCapturetime

     It uses a list containing PhotoWithMetadata instances as a data model

     It implements support for iterator and "in" type of use
    '''
    _nb_of_background_picture_loading_threads = 5

    def __init__(self):

        self._photo_collection = []
        self._stop_background_preview_load_event = threading.Event()
        self._update_background_preview_load_event = threading.Event()
        self._update_background_barrier = threading.Barrier(__class__._nb_of_background_picture_loading_threads)
        return

    @classmethod
    def set_nb_of_background_picture_loading_threads(cls, nb_thread):
        __class__._nb_of_background_picture_loading_threads = nb_thread

    def __iter__(self):
        self.n = 0
        return self

    def __next__(self):
        try:
            self._photo_collection[0]
        except LookupError:
            raise StopIteration
        else:
            if self.n <= len(self._photo_collection) - 1:
                result = self._photo_collection[self.n]
                self.n += 1
                return result
            else:
                raise StopIteration

    def __getitem__(self, index):
        return self._photo_collection[index]

    def __delitem__(self, index):
        if not self._update_background_preview_load_event.is_set():
            self._update_background_preview_load_event.set()
        del self._photo_collection[index]
        return

    def position(self, photo):
        return self._photo_collection.index(photo)

    def remove(self, photo):
        if not self._update_background_preview_load_event.is_set():
            self._update_background_preview_load_event.set()
        self._photo_collection.remove(photo)
        return

    def __len__(self):
        return len(self._photo_collection)

    def __contains__(self, item):
        return True if item in self._photo_collection else False

    def __repr__(self):
        try:
            self._photo_collection[0]
        except LookupError:
            _string = "no Photos in collection"
        else:
            _string = ""
            for n, photo in enumerate(self._photo_collection):
                _string += "\nPHOTO NUMBER " + str(n).upper() + "\n\n"
                _string += str(photo)
        return _string

    def pop(self):
        return self._photo_collection.pop()

    def reset(self):
        if not self._update_background_preview_load_event.is_set():
            self._update_background_preview_load_event.set()
        self._photo_collection = []
        return True

    # def add(self, photo_to_be_inserted, index=None):
    #     logger.error("NotImplementedError")
    #     raise NotImplementedError
    def add(self, photo_to_be_inserted, index=None):
        if len(self._photo_collection) == 0:  # first element
            self._photo_collection.append(photo_to_be_inserted)
        # "younger" picture inserted @ first position
        elif photo_to_be_inserted.shot_timestamp <= self._photo_collection[0].shot_timestamp:
            self._photo_collection.insert(0, photo_to_be_inserted)
        # "older" picture appended @ last position
        elif photo_to_be_inserted.shot_timestamp >= self._photo_collection[-1].shot_timestamp:
            self._photo_collection.append(photo_to_be_inserted)
        else:
            for i in range(1, len(self._photo_collection)):
                if self._photo_collection[i - 1].shot_timestamp \
                        < photo_to_be_inserted.shot_timestamp \
                        <= self._photo_collection[i].shot_timestamp:
                    self._photo_collection.insert(i, photo_to_be_inserted)
                    break

        if not self._update_background_preview_load_event.is_set():
            self._update_background_preview_load_event.set()

        return self._photo_collection.index(photo_to_be_inserted)

    def interval_with_previous(self, photo):
        if len(self._photo_collection) == 0:
            raise LookupError  # TODO check if i can invent an exception or it needs to be declared in some ways
        try:
            index_ = self._photo_collection.index(photo)
            if index_ == 0:
                return 0
            else:
                return self._photo_collection[index_].shot_timestamp - self._photo_collection[index_ - 1].shot_timestamp
        except LookupError as Err:
            logger.error("%s \n %s not in PhotoCollection", Err, photo)
            raise ValueError

    def interval_with_next(self, photo):
        if len(self._photo_collection) == 0:
            raise LookupError  # TODO check if i can invent an exception or it needs to be declared in some ways
        try:
            index_ = self._photo_collection.index(photo)
            if index_ == len(self._photo_collection) - 1:
                return 0
            else:
                return self._photo_collection[index_ + 1].shot_timestamp - self._photo_collection[index_].shot_timestamp
        except LookupError as Err:
            logger.error("%s \n %s not in PhotoCollection", Err, photo)
            raise ValueError

    def compute_statistics_interval_with_previous(self):

        _StatDict = {}
        interval_list = [self.interval_with_previous(foto) for foto in self._photo_collection]
        _StatDict["NbShots"] = len(interval_list)
        if len(interval_list) <= 1:
            _StatDict["mean"] = 0
            _StatDict["median"] = 0
            _StatDict["mode"] = 0
            _StatDict["Standard Deviation"] = 0
            _StatDict["Duration 24fps"] = 0
            _StatDict["Duration 30fps"] = 0
            _StatDict["Duration 60fps"] = 0
        else:
            interval_list = interval_list[
                            1:]  # remove 1st element with interval=0 because having no predecessor
            _StatDict["mean"] = statistics.mean(interval_list)
            _StatDict["median"] = statistics.median(interval_list)
            try:
                _StatDict["mode"] = statistics.mode(interval_list)
            # in case there are 2 values in interval_list mode can't be found and returns an error that we catch
            except StatisticsError:
                _StatDict["mode"] = " NA "

            _StatDict["Standard Deviation"] = statistics.pstdev(interval_list)
            _StatDict["Duration 24fps"] = _StatDict["NbShots"] / 24
            _StatDict["Duration 30fps"] = _StatDict["NbShots"] / 30
            _StatDict["Duration 60fps"] = _StatDict["NbShots"] / 60

        return _StatDict

    def set_stop_background_preview_load_event(self):
        self._stop_background_preview_load_event.set()

    def is_set_stop_background_preview_load_event(self):
        return self._stop_background_preview_load_event.is_set()

    def clear_stop_background_preview_load_event(self):
        self._stop_background_preview_load_event.clear()

    def load_metadata_from_files(self, file_path, file_suffixes_in_scope=None,
                                 file_treated_tick_function_reference=False):
        logger.error("NotImplementedError")
        raise NotImplementedError



    def load_image_previews_in_memory(self, index_):
        """
        this functions checks if preview image is already stored in the PhotoWithMetadata class instance of the picture.
        If not it loads the preview

        function is called from the gui when a picture is clicked on the TableView and in a background thread that is
        loading all ui.active_photos images in memory.

        :param index_: index of the photo to be treated in the  PhotoOrderedCollection list
        :return: True if the preview can be loaded False in case it can not (because FileType not implemented or not
                 containing a preview
        """
        # global ui
        loaded = True
        if not isinstance(self._photo_collection[index_].get_matplotlib_image_preview(), np.ndarray):
            logger.warning(" %s photo preview not loaded as MIME type %s is not yet implemented",
                           self._photo_collection[index_].file_name,
                           self._photo_collection[index_].exif_metadata["File:MIMEType"])
            loaded = False

        return loaded

    def launch_background_picture_loader_threads(self,
                                                 progress_bar_ticker_function_reference=False,
                                                 load_completed_signal_emit_function_reference=False):
        logger.info(" IMAGE PREVIEW BACKGROUND LOADING STARTED")
        thread_list = []
        for index_ in range(__class__._nb_of_background_picture_loading_threads):
            t = threading.Thread(target=self._load_active_photos_preview_pictures_in_memory,
                                 args=( self._stop_background_preview_load_event,
                                       self._update_background_preview_load_event,
                                       self._update_background_barrier,
                                       index_,
                                       __class__._nb_of_background_picture_loading_threads,
                                       progress_bar_ticker_function_reference,
                                       load_completed_signal_emit_function_reference)
                                 )
            t.daemon = True
            t.start()
            thread_list.append(t)
        return thread_list

    def _load_active_photos_preview_pictures_in_memory(self,
                                                       stop_event,
                                                       update_event,
                                                       barrier,
                                                       index_,
                                                       step,
                                                       progress_bar_ticker_function_reference,
                                                       load_completed_signal_emit_function_reference):
        """
        This function is to be used in // threads. It loads preview images of the ui.active_photos list.
        Each tasks loads the images of "index_" rank  within a slot of p image
        For example index_ = 2 with step = 4 will load picture at rank 2,6,10,14,18,..

        :param stop_event: event signaling that loading has to be stopped because loading of another bunch of
                           picture has been requested via the interface
        :param update_event:  event signaling that image have been added or removed in the ui.active_photos list. Upon
                              this event tasks stop processing and reloads the ui.active_photos list so that they do not
                              fail by requesting no longer valid indexes in the list
        :param barrier:  threading  barrier used to wait that all thread reaches a consistent step
        :param index_: index of the picture to be treated within the modulo p
        :param step: p is the modulo that distribute the load between threads. For every slot of P image, thread one
                     deals with images 1, p+1, 2p+1, 3p+1,...
        :return: None or 255 if interrupted
        """
        # global ui  # required as called in a separate thread
        keep_going = True
        while keep_going:
            keep_going = False
            update_event.clear()
            for i in range(0, len(self._photo_collection), step):
                if stop_event.is_set():
                    rc = barrier.wait()
                    if rc == 0:
                        stop_event.clear()
                        logger.info(" IMAGE PREVIEW BACKGROUND LOADING CANCELLED")
                    return 255
                if update_event.is_set():
                    rc = barrier.wait()  # wait all thread received the event before clearing it
                    if rc == 0:
                        logger.info(" IMAGE PREVIEW BACKGROUND LOADING RESET")
                    update_event.clear()
                    keep_going = True
                    break
                pos = i + index_
                if pos <= len(self._photo_collection) - 1:
                    self.load_image_previews_in_memory(pos)
                    if progress_bar_ticker_function_reference:
                        progress_bar_ticker_function_reference()
        rc = barrier.wait()  # wait all tasks to complete
        if rc == 0:
            if load_completed_signal_emit_function_reference:
                load_completed_signal_emit_function_reference()
            logger.info(" IMAGE PREVIEW BACKGROUND LOADING COMPLETED")

        return None


    def duplicate_photo(self, row, duplicate_method):
        picture = self._photo_collection[row]
        status = True
        message = "OK"

        # input parameters checks
        if len(self._photo_collection) < 2:
            status = False
            message = "collection must be made of at least 2 picture so that duplicate can be created"
            return (status, message, None)
        elif row == len(self._photo_collection) - 1:
            status = False
            message = "last picture can't be duplicated"
            return (status, message, None)
        elif duplicate_method not in DuplicateMethod.get_duplicate_method_supported_list():
            status = False
            message = "duplicate method: " + str(duplicate_method) + " is not supported"
            return (status, message, None)

        # we are almost safe now :-) adequation of duplicate_method with previous clone still to be checked

        if isinstance(picture, Photo):  # Picture is a real one
            if picture.clone_set_with_next is None:  # picture has no clone yet
                # create a CloneSet by instanciating the  loneSet subclass that is registered as a PhotoCloned
                # class dictionnary - This spares the effort of implementing multiple if statements
                clone_set = CloneSet(
                    picture,
                    self._photo_collection[row + 1],
                    duplicate_method,
                )
                # create Cloned picture
                duplicated_picture = PhotoCloned(clone_set,
                                                 self._photo_collection[row],
                                                 self._photo_collection[row + 1]
                                                 )
                # set-up link between pictures and CloseSet both ways
                clone_set.list_of_clones.append(duplicated_picture)  # add picture to cloneset
                picture.clone_set_with_next = clone_set  # link clone_set with previous picture
                self._photo_collection[row + 1].clone_set_with_previous = clone_set  # and with following picture
                # print(picture.clone_set_with_next.list_of_clones)
                # add picture to collection
                self.add(duplicated_picture)
            else:  # physical picture with existing clones
                # check if existing clone_set has same duplicate method - if not it is not possible to add a new
                # clone to the clone set as all clones between two pictures must be generated with same duplicate
                # method
                if picture.clone_set_with_next.duplicate_method != duplicate_method:
                    # if PhotoCloned.clone_set_classes_dict_per_clone_method[duplicate_method] \
                    #         != type(picture.clone_set_with_next):
                    status = False
                    message = "Not possible to duplicate with : " + str(duplicate_method) + \
                              " method since cloned photos created with another method (" + \
                              str(picture.clone_set_with_next.duplicate_method) + ") already exist"
                    return (status, message, None)

                # create Cloned picture
                duplicated_picture = PhotoCloned(picture.clone_set_with_next,
                                                 self._photo_collection[row],
                                                 self._photo_collection[row + 1]
                                                 )
                # set-up link between pictures and CloseSet both ways
                picture.clone_set_with_next.list_of_clones.append(duplicated_picture)  # add picture to clone set
                # add picture to collection
                self.add(duplicated_picture)
                # distribute equal intervals between picture in clone set
                self.evenly_distribute_interval_over_clone_pictures_in_cloneset(duplicated_picture.clone_set)

        elif isinstance(picture, PhotoCloned):  # is a clone
            if picture.clone_set.duplicate_method != duplicate_method:
                # if PhotoCloned.clone_set_classes_dict_per_clone_method[duplicate_method] \
                #         != type(picture.clone_set):
                status = False
                message = "Not possible to duplicate with : " + str(duplicate_method) + \
                          " method since cloned photos created with another method (" + \
                          str(picture.clone_set.duplicate_method) + ") already exist"
                return (status, message, None)
            # create Cloned picture
            duplicated_picture = PhotoCloned(picture.clone_set,
                                             self._photo_collection[row],
                                             self._photo_collection[row + 1]
                                             )
            # set-up link between pictures and CloseSet both ways
            picture.clone_set.list_of_clones.append(duplicated_picture)  # add picture to cloneset
            # print(picture.clone_set.list_of_clones)
            # add picture to collection
            self.add(duplicated_picture)
            # distribute equal intervals between picture in clone set
            self.evenly_distribute_interval_over_clone_pictures_in_cloneset(duplicated_picture.clone_set)
        else:
            logger.error("NotImplementedError")
            raise NotImplementedError

        return (status, message, duplicated_picture)

    def duplicate_list_of_photos(self, sorted_list_of_photo_index_2_be_duplicated, duplicate_method):
        status = True
        duplicated_pictures = []  # contains list of picture that could be duplicated
        could_not_duplicate_rows = {}  # dict 'row number' : 'message' of picture that can't be duplicated
        message = "OK"

        # check if all rows/picture are eligible to duplication
        i = 0
        for row in sorted_list_of_photo_index_2_be_duplicated:
            rc, msg, duplicated_picture = self.duplicate_photo(row + i, duplicate_method)
            if rc:
                i += 1
                duplicated_pictures.append(duplicated_picture)
            else:
                status = False
                message = "NOK"
                could_not_duplicate_rows[self._photo_collection[row + i]] = msg

        return (status, message, duplicated_pictures, could_not_duplicate_rows)

    def evenly_distribute_interval_over_clone_pictures_in_cloneset(self, clone_set):
        """
        when picture are added to collection via "add" method their timestamp is assigned as the mid-point between
        previous and next picture - over multiple, not evenly dsitributed, duplication this causes interval of different
        values within same clone set (means set of clone picture between two real picture) - this is fake and error
        prone as clones have no real shot_timestamp and should therefore appear as evenly distributed.
        This method is assigning timestamp to all picture of a clone set so that they are distributed with the very
        same interval between themselves

        :param clone_set: clone_set for which picture are to be treated
        :return:
        """

        if len(clone_set.list_of_clones) == 0:
            return False

        # compute interval between clones
        interval = (clone_set.next_picture.shot_timestamp - clone_set.previous_picture.shot_timestamp) / \
                   (len(clone_set.list_of_clones) + 1)
        # sort the clone pictures by order of position in the collection as the order in the list_of_clones is the
        # one of insertion and depending it can be different subject to clone creation order. for instance
        # creating clone from previous image yield them to be in reverse order in list_of_clones as compared to the
        # order in the collection
        clone_set.list_of_clones.sort(key=lambda picture: self.position(picture))
        for i, picture in enumerate(clone_set.list_of_clones):
            picture.set_shot_timestamp(
                clone_set.previous_picture.shot_timestamp +
                ((i + 1) * interval)
            )

        return True

    def remove_photo(self, row):  # TODO should be an overwrite of parent method remove
        picture = self._photo_collection[row]
        status = True
        message = "OK"

        if not (0 <= row < len(self._photo_collection)):
            status = False
            message = "row: " + str(row) + "out of _photo_collection range"
            return (status, message)

        # Check if row is physical picture or clone
        if isinstance(picture, Photo):  # this is a physical picture
            # if the picture is used to build existing clone picture then it can't be deleted until all clones
            # are actually removed
            if (picture.clone_set_with_next is not None) or (picture.clone_set_with_previous is not None):
                status = False
                message = "Can't remove picture as it references an existing clone picture - " \
                          "clone picture must be removed first"
                return (status, message)
            self.remove(picture)  # use remove rather than del as it emits signal for preview background load reset
        elif isinstance(picture, PhotoCloned):  # this is a clone picture
            # if this is the last clone in the CloneSet then remove CloneSet and links to it before actually removing
            # picture
            picture.clone_set.list_of_clones.remove(picture)  # remove picture from clone_set list
            if len(picture.clone_set.list_of_clones) == 0:
                self._photo_collection[row - 1].clone_set_with_next = None
                self._photo_collection[row + 1].clone_set_with_previous = None
                del picture.clone_set
            else:
                self.evenly_distribute_interval_over_clone_pictures_in_cloneset(picture.clone_set)
            self.remove(picture)  # use remove rather than del as it emits signal for preview background load reset

            # TODO NOT IN ALL CASES duplicated_picture.clone_set.evenly_distribute_interval_over_clone_pictures_in_cloneset()
        else:
            logger.error("NotImplementedError")
            raise NotImplementedError

        return (status, message)

    def remove_list_of_photos(self, sorted_list_of_photo_index_2_be_duplicated):
        status = True
        removed_pictures = []  # contains list of picture that could be removed
        could_not_remove_rows = {}  # dict 'index number' : 'message' of picture that can't be removed
        message = "OK"

        # split clones from real pictures
        clone_index_list = [index for index in sorted_list_of_photo_index_2_be_duplicated if
                            isinstance(self._photo_collection[index], PhotoCloned)]
        photo_index_list = [index for index in sorted_list_of_photo_index_2_be_duplicated if
                            isinstance(self._photo_collection[index], Photo)]

        # first remove all clones as it is always possible to remove and it will "free-up" pictures with links
        # on clones that are to be deleted in the list
        for i, index in enumerate(clone_index_list):
            removed_pictures.append(self._photo_collection[index - i])
            rc, msg = self.remove_photo(index - i)
            if rc:  # adjust row number in photo_list_index in line with removal done in clone_list_index
                for j, photo_index in enumerate(photo_index_list):
                    if photo_index >= index - i:  # if a clone with lower index has been removed
                        photo_index_list[j] -= 1  # ...then shift row up by  one position
            else:
                raise ValueError  # means index out of _photo_collection range

        # then remove picture  that are not linked to any clone
        i = 0
        for index in photo_index_list:
            removed_pictures.append(self._photo_collection[index - i])
            rc, msg = self.remove_photo(index - i)
            if rc:
                i += 1
            else:
                removed_pictures.pop()
                status = False
                message = "NOK"
                could_not_remove_rows[self._photo_collection[index - i]] = msg

        return (status, message, removed_pictures, could_not_remove_rows)

    def generate_computed_pictures(self,
                                   output="file",  # can be in {"file", "opencv3", "qpixmap"}
                                   file_treated_tick_function_reference=False,
                                   row_start=None,
                                   row_stop=None,
                                   size=None  # if not None must be (width, height) format
                                   ):
        """
        Generate final pictures after computing clones with their respective duplicate method
        and resize operation done if requested
        Either to disk - in that case resize is ignored
        or by returning an in memory opencv3 image
        of by returning an in memory qpixmap image

        if qpixmap option is requested stores the qpixmap image within the PhotoWithMetadata or PhotoCloned instance of
        the picture - as provided by those classes that inherits StoreQPixmap class - This is to cache results
        and allows computing only pictures with parameters changed on succesive iterations



        :param output: specify if output is disk or cv2 valid values are {"file", "opencv3"}
        :param file_treated_tick_function_reference: fucntion to increment progress bar
        :param row_start: index position to start generation in collection
        :param row_stop: index position to start generation in collection
        :parame size: (width, height) couplestating dimension for resizing - Does not apply for "file" option
        :return: a triplet (status, message, img_computed_list)
                 status is either True or False
                 message contains a detailed error message
                 img_computed_list is
                 None if "file" output option is set
                 or a list of opencv3 images if output is set to "opencv3"

        """

        def opencv_2_resized_qpixmap(img_cv2_bgr, size=None):
            """
            Transform an opencv image in Qpixmap and resize it if a size coupe is provided
            :param img_cv2_bgr:
            :param size: None of (width, height)
            :return: qpixmap resized
            """
            if size is not None:
                width, height = size
                img_resized = cv2.resize(img_cv2_bgr, (width, height), interpolation=cv2.INTER_AREA)
            else:
                img_resized = img_cv2_bgr
            # BGR to RGB conversion
            img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
            # create QT image
            image = QImage(img_rgb, img_rgb.shape[1],
                           img_rgb.shape[0], img_rgb.shape[1] * 3, QImage.Format_RGB888)
            qpixmap_ = QPixmap(image)

            return qpixmap_

        def draw_text(frame, text, x, y, color=(255, 255, 255), thickness=3, size=3):
            if x is not None and y is not None:
                return cv2.putText(
                    frame, text, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, size, color, thickness)

        index_start = row_start
        index_stop = row_stop
        status = True
        message = "OK"
        img_computed_list = []

        if index_start is None and index_stop is None:
            index_start = 0
            index_stop = len(self._photo_collection) - 1

        # check parameter consistency
        # index values
        if len(self._photo_collection) == 0:
            message = " generate_computed_pictures not possible - collection is empty"
            status = False
            return (status, message, img_computed_list)
        elif not (type(index_start) == int and type(index_stop) == int):
            logger.error("generate_computed_pictures possible. index_start and index_stop must be int")
            raise ValueError
        elif index_stop < index_start \
                or not (0 <= index_start < len(self._photo_collection)) \
                or not (0 <= index_stop < len(self._photo_collection)):
            logger.error(
                "generate_computed_pictures not possible : index_start and index_stop values not properly ordered")
            raise ValueError
        # output values
        if output not in {"file", "opencv3", "qpixmap"}:
            logger.error(
                "invalid output value while generating picture : must be 'file' or 'opencv3'")
            raise ValueError

        # set-up directory stuff
        current_directory = os.getcwd()
        target_directory = current_directory + "\\_extract-" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        try:
            os.mkdir(target_directory)
        except FileExistsError:
            # if directory already exists we do nothing so as not to mess up current contents
            message = "can't generate_computed_pictures files folder % s already exists" \
                      + str(target_directory)
            logger.error(message)
            status = False
            return (status, message, img_computed_list)

        logger.info("START EXTRACT OF %s PICTURE INTO %s", str(index_stop - index_start + 1), target_directory)

        for index, picture in enumerate(self._photo_collection):

            # skip pictures out of range
            if index < index_start or index > index_stop:
                continue

            sequence_file_name = target_directory + "\\LRT_{0:05d}".format(
                index + 1) + ".jpg "  # TODO implement case where images are not jpg !

            if isinstance(picture, Photo):  # this is a real picture - simply copy it to new directory

                # only jpeg output format is implemented. It is defined when building the name of the output file

                if output == "file":

                    # try:   # TODO this copy file doesn't work for video as no file exists on disk
                    #     shutil.copyfile(current_directory + "\\" + picture.file_name,
                    #                     sequence_file_name)
                    #     logger.info("%s file extracted as %s", picture.file_name, sequence_file_name)
                    # except Exception as Err:
                    #     logger.error("copy failed for file %s with error code : %s", picture.file_name, str(Err))

                    try:
                        cv2.imwrite(sequence_file_name, picture.get_opencv_image_fullsize())
                        logger.info("%s file extracted as %s", picture.file_name, sequence_file_name)
                    except Exception as e:
                        print(exception_to_string(e))

                elif output == "opencv3":

                    img_cv2 = picture.get_opencv_image_fullsize()
                    if size is not None:
                        width, height = size
                        img_cv2 = cv2.resize(img_cv2, (width, height), interpolation=cv2.INTER_AREA)
                    img_computed_list.append(img_cv2)

                elif output == "qpixmap":

                    # compute a hash that uniquely identify generation parameters so that when a new generation is
                    # launched only the picture with new parameters are recomputed and the process is fast
                    key_string = str(size)

                    img_qpixmap = picture.get_qpixmap(key_string)
                    if img_qpixmap is None:
                        img_qpixmap = opencv_2_resized_qpixmap(picture.get_opencv_image_fullsize(), size=size)
                        picture.set_qpixmap(key_string, img_qpixmap)
                    img_computed_list.append(img_qpixmap)

                if file_treated_tick_function_reference:
                    file_treated_tick_function_reference()  # one tick per file treated if function provided

                method = ""

            elif isinstance(picture, PhotoCloned):

                # Determine  what is the rank of the current picture in the list of
                # virtual pictures using the clone_set
                nb_intervals = len(picture.clone_set.list_of_clones) + 1
                picture.clone_set.list_of_clones.sort(key=lambda picture: self.position(picture))
                interval_rank = picture.clone_set.list_of_clones.index(picture) + 1  # starts at 1, not at 0

                # TODO DEBUG REMOVE add method on frame
                # draw_text(img_transition,str(picture.duplicate_method), 150, 150)

                if output == "file":

                    # create the new picture in memory from previous ones -
                    img_transition \
                        = picture.clone_set.duplicate_method_set.create_transition_image(
                        picture.clone_set.previous_picture.get_opencv_image_fullsize(),
                        picture.clone_set.next_picture.get_opencv_image_fullsize(),
                        nb_intervals,
                        interval_rank,
                    )
                    # write it to disk
                    try:
                        cv2.imwrite(sequence_file_name, img_transition)
                        logger.info("%s file extracted as %s", picture.file_name, sequence_file_name)
                    except Exception as e:
                        print(exception_to_string(e))
                    # add metadata that will be copied from original using py3exiv2 and modify time based tags
                    # TODO write to disk to be implemented..as opencv has lost all metadata
                    # TODO NEED TO FIX INSTALL OF PY3EXIV2 package that fails with python 3.6 on windows due to utf-8 decode error

                elif output == "opencv3":
                    # create the new picture in memory from previous ones -
                    img_transition \
                        = picture.clone_set.duplicate_method_set.create_transition_image(
                        picture.clone_set.previous_picture.get_opencv_image_fullsize(),
                        picture.clone_set.next_picture.get_opencv_image_fullsize(),
                        nb_intervals,
                        interval_rank,
                    )
                    img_cv2 = img_transition
                    if size is not None:
                        width, height = size
                        img_cv2 = cv2.resize(img_transition, (width, height), interpolation=cv2.INTER_AREA)
                    img_computed_list.append(img_cv2)

                elif output == "qpixmap":

                    # compute a hash that uniquely identify generation parameters so that when a new generation is
                    # launched only the picture with new parameters are recomputed and the process is fast
                    key_string = str(picture.clone_set)
                    key_string += str(interval_rank)
                    key_string += str(picture.duplicate_method)
                    key_string += str(picture.clone_set.computation_parameters)
                    key_string += str(size)
                    # print(picture.file_name," key_string= ",key_string)

                    img_qpixmap = picture.get_qpixmap(key_string)
                    if img_qpixmap is None:
                        # create the new picture in memory from previous ones -
                        img_transition \
                            = picture.clone_set.duplicate_method_set.create_transition_image(
                            picture.clone_set.previous_picture.get_opencv_image_fullsize(),
                            picture.clone_set.next_picture.get_opencv_image_fullsize(),
                            nb_intervals,
                            interval_rank,
                        )
                        img_qpixmap = opencv_2_resized_qpixmap(img_transition, size=size)
                        picture.set_qpixmap(key_string, img_qpixmap)
                    img_computed_list.append(img_qpixmap)

                if file_treated_tick_function_reference:
                    file_treated_tick_function_reference()  # one tick per file treated if function provided

                method = "with duplicate method: " + picture.duplicate_method
            else:
                raise ValueError

            logger.info("%s generated as %s %s", picture.file_name, output, str(method))

        logger.info("EXTRACTION COMPLETED")

        return (status, message, img_computed_list)

    # TODO implement __repr__



class PhotoNonOrderedCollection(PhotoCollection):  # TODO MAY BE THIS COLLECTION IS NOT NEEDED - PARENT ONE CAN DO
    '''
    Collection of non ordered PhotoWithMetadata
    '''

    def __init__(self):
        super().__init__()
        return

    def append(self, photo):
        self._photo_collection.append(photo)
        return

    # TODO implement __repr__


class PhotoOrderedCollectionFromVideoRead(PhotoCollection):

    _video_file_path = None   # store video file path for further reference
    _video_properties = {}

    # properties of the video made avlailable by opencv3
    VIDEO_PROPERTIES = {cv2.CAP_PROP_FRAME_HEIGHT,
                        cv2.CAP_PROP_FRAME_WIDTH,
                        cv2.CAP_PROP_FPS,
                        cv2.CAP_PROP_FRAME_COUNT,
                        cv2.CAP_PROP_FORMAT,
                        cv2.CAP_PROP_MODE,
                        cv2.CAP_PROP_BRIGHTNESS,
                        cv2.CAP_PROP_CONTRAST,
                        cv2.CAP_PROP_SATURATION,
                        cv2.CAP_PROP_HUE,
                        cv2.CAP_PROP_GAIN,
                        cv2.CAP_PROP_EXPOSURE,
                        cv2.CAP_PROP_CONVERT_RGB,
                        cv2.CAP_PROP_TEMPERATURE}

    @classmethod
    def set_video_file_path(cls, file_path):
        __class__._video_file_path = file_path

    @classmethod
    def get_video_file_path(cls):
        return __class__._video_file_path

    def __init__(self):
        super().__init__()
        # logger.error("NotImplementedError")
        # raise NotImplementedError
        return

    def load_metadata_from_files(self, video_file, file_suffixes_in_scope=None,
                                 file_treated_tick_function_reference=False):
        """
        load the description of frames stored in vidoe_file passed as first parameter ans stores as class attribute
        the property of the video for further reference.
        TODO second parameter is o far ignored - could be implemente later.
        Optionnally a function passed as third parameter can be called each time a file from the list is treated so
        that ui can display progress

        :param video_file: a list containing a unique item which is the video file to extract frames from
        :param file_suffixes_in_scope: not implemented for video - kept for compatibilty with other collections
        :param file_treated_tick_function_reference: a function that will be called every time a file is treated
        :return:

        How to read a single frame with its index is documented here:
        https://stackoverflow.com/questions/46100858/how-to-get-frame-from-video-by-its-index-via-opencv-and-python?rq=1&utm_medium=organic&utm_source=google_rich_qa&utm_campaign=google_rich_qa
        """
        cap = cv2.VideoCapture(video_file)

        # get video property
        for property_id in __class__.VIDEO_PROPERTIES:
            __class__._video_properties[property_id] = cap.get(property_id)
        if __debug__:
            logger.info("VIDEO PROPERTIES for %s = %s", str(video_file), str(__class__._video_properties))

        # time_ = os.stat(video_file).st_ctime  # creation time of the file as timestamp of video
        frame_fake_time = os.stat(video_file).st_ctime  # creation time of the file as timestamp of video
        head, tail = os.path.split(video_file)  # remove path - keep file name only
        # load picture
        for i in range(int(__class__._video_properties[cv2.CAP_PROP_FRAME_COUNT])):
        # for i in range(1, 100):  # Frame count starts at 1

            # ret, frame = cap.read()  # return a numpy array BGR in frame  - actuelly no need to read

            # build fake file name made from videofile + file index on 6 digits (can cope with 9 hours 30fps)
            file_name = "".join(tail.split(".")[:-1]) + "_{0:06d}".format(i)  # TODO no suffix added..tb clarified
            # build fake metadata containing SourceFile and create date in EXIF format "YYYY:mm:dd HH:MM:SS"
            metadata = {}
            metadata["SourceFile"] = file_name
            # frame_fake_time = time_ + cap.get(cv2.CAP_PROP_POS_MSEC) / 10  # so that frames  have different seconds
            frame_fake_time += 1  # so that frames  have different seconds
            metadata["EXIF:CreateDate"] = \
                str(datetime.datetime.fromtimestamp(frame_fake_time).strftime('%Y:%m:%d %H:%M:%S'))

            # add photo to collection
            self.add(PhotoFromVideo(file_name, metadata, video_file, i))

            # # compute and store matplotlib
            # width, heigth = Photo._matplotlib_image_preview_size
            # image_resized = cv2.resize(frame, (width, heigth),
            #                            interpolation=cv2.INTER_AREA)  # /!\ THIS IS CPU INTENSIVE IF image_cv IS BIG
            # # TODO THIS IS NOT PROTECTING THE ORIGINAL IMAGE RATIO
            # # transform opencv BGR in RGB as supported by Qt images
            # matplotlib_image = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)


            # compute and add qpixmap
            """
            Better to do it at metadata loading time rather than later because full image is already
            loaded in memory 
            Investigate if it can be threaded ? probably not as this is based on a sequential reading
            of a unique file - or it requires to create files on disk at metadta loading time nd then
            to read them in // later with same approach than genuine pictures
            """

            # store full size image ? or not as it can be accessed directly

            if file_treated_tick_function_reference:
                file_treated_tick_function_reference()  # one tick per file treated if function provided

        # release opencv videoreader
        cap.release()

        logger.info(" META DATA LOADED from video_file ")
        #
        # # and then creates PhotoWithMetadata class instances and populate container
        # for file_, metadata_ in resultats.items():
        #     try:
        #         self.add(PhotoWithMetadata(file_, metadata_))
        #     except Exception as e:
        #         print(exception_to_string(e))
        #     if file_treated_tick_function_reference:
        #         file_treated_tick_function_reference()  # second tick per file treated if function provided

        return len(self)


class PhotoOrderedCollectionByCapturetime(PhotoCollection):
    '''
    Collection of Photos ordered by capture time

    items are always kept by order of growing age of capture (younger first)

    add:  add a photo in Collection at the position corresponding to its capture time

    interval_with_previous: return interval with previous shot in collection (secs)

    interval_with_next: return interval with next shot in collection (secs)

    '''

    _size_of_file_chunks = 300  # size of file chunks to be loaded per Process

    @classmethod
    def set_size_of_file_chunks(cls, size):
        __class__._size_of_file_chunks = size

    @staticmethod
    def _move_blob(image_src, image_dest, coord_src, coord_end, nb_intervals, interval_rank,
                   length=150, radius_max=20):
        """
        receives two images taken in sequence and generate an intermediate image based on Shi-Tomasi Corner Detector and
        Lukas-Kanade Optical Flow method implemented in opencv
        (see https://docs.opencv.org/3.4.0/d7/d8b/tutorial_py_lucas_kanade.html)


        :param image_src: an numpy array image as generated by opencv2
        :param image_dest:an numpy array image as generated by opencv2
        :param coord_src: a numpy array as generated by the opencv function cv2.goodFeaturesToTrack which value have
                          been converted to integers via np.int0
        :param coord_end: a numpy array as generated by the opencv function cv2.calcOpticalFlowPyrLK which value have
                          been converted to integers via np.int0
        :nb_intervals: number of interval between the two real pictures
        :interval_rank: position of the current interval in the sequence of intervals between the two real pictures
                        Starts at 1 not at 0 as usual in Python position
        :param length: nb of pixel to move around the good features coordinates
        :param radius_max : maximum distance between the coord_src and coord_end within which we copy the blob.
                            if distance is > radius_max no copy is made
        :return: a new image for which the feature point that moved fof more than 1 pixels are moved. New image is based
                 on src image. the place of initial feature is set to the value in dest image and vice versa
        """

        # TODO ignore data input validation for time being
        if image_src.shape != image_dest.shape:
            raise ValueError

        # print("FROM MOVE BLOB shape src - dest",image_src.shape, image_dest.shape)

        # record image size
        x_max, y_max, c = image_src.shape

        # initialize new image with image source
        image_new = np.copy(image_src)

        # compute coordinates of intervals points. Thank you numpy !
        corners_midpoint = np.int0((coord_src * (nb_intervals - interval_rank) + coord_end * interval_rank)
                                   / nb_intervals)

        nb_of_points = len(coord_src)
        nb_of_moves = 0
        distance_list = []
        for (src, end, target) in zip(coord_src, coord_end, corners_midpoint):

            y_src, x_src = src.ravel()
            y_end, x_end = end.ravel()
            y_target, x_target = target.ravel()

            # points with a move of less than 1 and located in border (of length width) are not treated
            distance = ((x_src - x_end) ** 2 + (y_src - y_end) ** 2) ** 0.5
            distance_list.append(distance)
            if distance <= 1 \
                    or distance > radius_max \
                    or (x_src < length) \
                    or (y_src < length) \
                    or (x_max - x_src < length) \
                    or (y_max - y_src < length) \
                    or (x_end < length) \
                    or (y_end < length) \
                    or (x_max - x_end < length) \
                    or (y_max - y_end < length) \
                    :
                continue
            nb_of_moves += 1

            # generate_computed_pictures blob to be moved on source image at location of source corner
            blob = np.zeros((2 * length, 2 * length, 3), np.uint8)
            # blob[:, :] = image_src[x_src - length:x_src + length, y_src - length:y_src + length]
            blob[:, :] = image_dest[x_end - length:x_end + length, y_end - length:y_end + length]

            # replace blob at source location by it's footprint on destination image
            image_new[x_src - length:x_src + length, y_src - length:y_src + length] = \
                image_dest[x_src - length:x_src + length, y_src - length:y_src + length]

            # copy blob at target interpolated point
            image_new[x_target - length:x_target + length, y_target - length:y_target + length] = blob[:, :]

        logger.info(" %s blobs copied - nb intervals = %s - interval rank = %s", str(nb_of_moves),
                    str(nb_intervals), str(interval_rank))
        # distance_list = sorted(distance_list, reverse=True)
        # print(str(nb_of_moves) + " point moved out of " + str(nb_of_points) + " detected")
        # for i in range(50):
        #     print(distance_list[i])

        # print("size of img_new", image_new.shape)
        return image_new

    @staticmethod
    def _create_lucas_kanade_image(img_before, img_after, nb_intervals, interval_rank
                                   , shi_tomasi_dict_param=None, lucas_Kanade_dict_param=None):

        # Default value for params for ShiTomasi corner detection
        FEATURES_PARAMS = dict(maxCorners=200000,
                               qualityLevel=0.01,  # minimum quality level between 0 and 1
                               minDistance=7,  # minimum distance between corners
                               blockSize=7)

        # default values parameters for lucas kanade optical flow
        LK_PARAMS = dict(winSize=(15, 15),
                         maxLevel=2,
                         criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

        if shi_tomasi_dict_param is not None:
            FEATURES_PARAMS = shi_tomasi_dict_param

        if lucas_Kanade_dict_param is not None:
            LK_PARAMS = lucas_Kanade_dict_param

        # Detect Shi-Tomasi corners as per current parameters
        img_before_gray = cv2.cvtColor(img_before, cv2.COLOR_BGR2GRAY)
        corners = cv2.goodFeaturesToTrack(img_before_gray, mask=None, **FEATURES_PARAMS)
        corners_ = np.int0(corners)

        img_after_gray = cv2.cvtColor(img_after, cv2.COLOR_BGR2GRAY)
        corners_after, st, err = cv2.calcOpticalFlowPyrLK(img_before_gray, img_after_gray, corners, None,
                                                          **LK_PARAMS)
        corners_after_ = np.int0(corners_after)

        # select only corners which flow can be found in img_after_gray - st==1 in that case
        good_corners_before = corners_[st == 1]
        good_corners_after = corners_after_[st == 1]

        try:
            img_with_blobs_updated = __class__._move_blob(img_before, img_after,
                                                          good_corners_before, good_corners_after,
                                                          nb_intervals, interval_rank,
                                                          15)  # <== nb of pixels moved around the point
        except Exception as e:
            print(exception_to_string(e))

        return img_with_blobs_updated

    def __init__(self):
        super().__init__()
        return

    def _load_metadata_chunk_of_photos(fileList, q_results, q_signal, file_suffixes_in_scope):
        '''

        Call ExifTool and generate_computed_pictures metadata from image file
        This fonction is meant to be spawn in parallel sub processes in order to speed up the
        loading of image metadata for big sequences

        :param fileList: a list containing file names
        :param q_results: a queue used to return Photos results to the main task
        :param q_signal: a queue used to send a tick after each file is treated - tick is used to update progress bar
                         in the user interface
        :file_suffixes_in_scope:  list of file suffix to be loaded
        :return:  a dictionary with key=filename and value = a dictionary of all metatag for the picture
        '''

        with exiftool.ExifTool() as et:
            photo_metatag_list = {}
            for file in fileList:
                for ext in file_suffixes_in_scope:
                    q_signal.put(1)  # tick for progress Bar
                    if file.endswith(ext):
                        photo_metatag_list[exiftool.fsencode(file)] = et.get_metadata(file)

        q_results.put(photo_metatag_list)
        # tell mother process that the job is done by putting False in the queue
        q_results.put(False)  # TODO investigate what can be done with end task that i saw somewher in the doc
        return

    def load_metadata_from_files(self, folder, file_suffixes_in_scope=None,
                                 file_treated_tick_function_reference=False):
        """
        load the list of files passed as first parameter for the file suffix passed in second parameter.
        Optionnally a function passed as third parameter can be called each time a file from the list is treated so
        that ui can display progress

        :param files: a list containinf files to be loaded
        :param file_suffixes_in_scope: a list containing the suffix to be considered  i.e. [".NEF", ".JPG", ".jpg"]
        :param file_treated_tick_function_reference: a function that will be called every time a file is treated
        :return:
        """

        def list_2_listoflist_by_chunk(file_list, size_of_chunk):
            '''
            create a list of list out of a flat list
            needed to use pool.map function submitting suprocesses with lists rather than items as input

            :param file_list: list of filename
            :param size_of_chunk: size of chunks to be created out of the input list
            :return: A list of list for which inner list are of size size_of_chunk
            '''
            size = size_of_chunk
            length = len(file_list)
            list_of_list = []
            for i in range(0, length, size):
                sublist = []
                for j in range(size):
                    if (i + j) < length:
                        sublist.append(file_list[i + j])
                list_of_list.append(sublist)

            return list_of_list

        # get list of files in folder
        file_list = os.listdir(folder)

        # split the file list in several chunks that will be treated in parrallel by subprocesses
        chunks = list_2_listoflist_by_chunk(file_list,
                                            __class__._size_of_file_chunks)  # divide the work by chunk of 300 files

        # prepare environment for communicating with sub processes
        q_results = Queue()
        q_signal = Queue()
        nb_processes = len(chunks)

        process_list = [Process(target=__class__._load_metadata_chunk_of_photos,
                                args=(chunks[i], q_results, q_signal, file_suffixes_in_scope)) for i in
                        range(nb_processes)]

        for p in process_list:
            p.start()

        # Poll both queues until all processes are completed
        results = []
        nb_terminated_process = 0
        while True:
            try:  # collect tick signal so as to update the progress bar
                q_signal.get_nowait()
                if file_treated_tick_function_reference:
                    file_treated_tick_function_reference()  # one first tick per file treated if function provided
            except queue.Empty:
                pass
            try:  # collect results from sub processes
                item = q_results.get_nowait()
                if not item:  # when a sub process finishes its task it put False into the result queue
                    nb_terminated_process += 1
                else:
                    results.append(item)
            except queue.Empty:
                pass
            if nb_terminated_process >= nb_processes:
                break

        logger.info(" META DATA available in queues ")

        # result is a list of Dictionaries
        resultats = ChainMap(*results)  # ...that we transform into a flat dictionary

        logger.info(" META DATA LOADED from files ")

        # and then creates PhotoWithMetadata class instances and populate container
        for file_, metadata_ in resultats.items():
            try:
                self.add(PhotoWithMetadata(str(file_, 'utf-8'), metadata_))
            except Exception as e:
                print(exception_to_string(e))
            if file_treated_tick_function_reference:
                file_treated_tick_function_reference()  # second tick per file treated if function provided

        return len(self)





def controler_temps():
    """
    Contrôle le temps mis par une fonction pour s'exécuter.
    :rtype: function
    """

    def decorateur(fonction_a_executer):
        """Notre décorateur. C'est lui qui est appelé directement LORS
        DE LA DEFINITION de notre fonction (fonction_a_executer)"""

        def fonction_modifiee(*args):
            """Fonction renvoyée par notre décorateur. Elle se charge
            de calculer le temps mis par la fonction à s'exécuter"""

            tps_avant = time.time()  # Avant d'exécuter la fonction
            valeur_renvoyee = fonction_a_executer(*args)  # On exécute la fonction
            tps_apres = time.time()
            tps_execution = tps_apres - tps_avant
            # if tps_execution >= nb_secs:
            logger.info("La fonction %s a mis %s pour s'exécuter", str(fonction_a_executer), str(tps_execution))
            return valeur_renvoyee

        return fonction_modifiee

    return decorateur


def print_exec_time(tps_avant, label_to_be_printed):
    tps_apres = time.time()
    tps_execution = tps_apres - tps_avant
    logger.info("La fonction %s a mis %s pour s'exécuter", str(label_to_be_printed), str(tps_execution))
    return


def exception_to_string(excp):
    stack = traceback.extract_stack()[:-3] + traceback.extract_tb(excp.__traceback__)  # add limit=??
    pretty = traceback.format_list(stack)
    return ''.join(pretty) + '\n  {} {}'.format(excp.__class__, excp)


if __name__ == "__main__":
    sys.exit()
