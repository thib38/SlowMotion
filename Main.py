'''	==================================================================================================

    ==================================================================================================  '''
import logging
import os
import statistics
import copy
import sys
import traceback
import pickle
import datetime
import shutil
from io import BytesIO
import threading
import numpy as np
import lzma
import calendar
import time
import random
import string
from collections import ChainMap, OrderedDict
from multiprocessing import Queue, Process
from statistics import StatisticsError
import queue


import matplotlib as mpl
import matplotlib.pyplot as plt
# from PyQt5.QtCore import QAbstractTableModel, QModelIndex, QVariant
from PyQt5.QtCore import *  # crashes if not generic
from PyQt5.QtGui import *
from PyQt5.QtWidgets import QMainWindow, QSizePolicy, QHeaderView, QFileSystemModel, QMessageBox, QMenu, QAction, \
    QLabel, QProgressBar, QDialog, QTableView, QButtonGroup, QFileDialog

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

import cv2
# QtDesigner generated code imports
# from MyFithWindow import *

from photo import Photo, PhotoCloned, PhotoWithMetadata, PhotoCollection, PhotoNonOrderedCollection,\
    PhotoOrderedCollectionByCapturetime, PhotoOrderedCollectionFromVideoRead,\
    CloneSet, DuplicateMethod, DuplicateInterpolate, DuplicateLucasKanade, DuplicateSimpleCopy
from My6thWindow import *
from removeNoP import *
from Nplicate2 import *
from lucasKanadeParam import *
from duplicateMethod import *
from MyVideoWindowQPixmap import *


class ShowSummary:
    '''
    Display Summary key values from a PhotoWithMetadata container provided as inputs

    It uses methods of PhotoCollection and PhotoOrderedColletionByCapturetime

    all method prefixed by "stat" are gathered in the summary

    '''

    def __init__(self, gui_object, photo_container):
        self._list_model = QStandardItemModel(gui_object)
        self.photo_container = photo_container
        statistics_dict = self.photo_container.compute_statistics_interval_with_previous()
        self._populate_list_model(statistics_dict)
        gui_object.setModel(self._list_model)

    def _populate_list_model(self, statistics_dict):
        list_of_lines = []
        for key, value in statistics_dict.items():
            list_of_lines.append(str(key) + " = " + (str(value) if not isinstance(value, float) else str(
                "{:.1f}".format(value))))
        for line in list_of_lines:
            item = QStandardItem(line)
            self._list_model.appendRow(item)

    def update(self):
        self._list_model.clear()
        statistics_dict = self.photo_container.compute_statistics_interval_with_previous()
        self._populate_list_model(statistics_dict)

    # TODO implement __repr__


class ImageAndPlotDisplayWithMatplotlib(FigureCanvas):

    def __init__(self, parent=None):
        self.parent_ = parent

        # get parent widget size
        parent_width = self.parent_.frameGeometry().width()
        parent_height = self.parent_.frameGeometry().height()

        # Initialize figure
        dpi = 100
        self.fig = plt.figure(
            figsize=(parent_width / dpi, parent_height / dpi),
            dpi=dpi  # figsize is in inches because dpi is !
        )
        FigureCanvas.__init__(self, self.fig)
        self.setParent(self.parent_)
        self.fig.patch.set_facecolor("None")
        # FigureCanvas.setSizePolicy(self, QSizePolicy.Expanding, QSizePolicy.Expanding)
        FigureCanvas.setSizePolicy(self, QSizePolicy.Fixed, QSizePolicy.Fixed)
        FigureCanvas.updateGeometry(self)
        self.fig.set_tight_layout(True)  # force layout optimization each time fig is drawn
        self.fig.canvas.setStyleSheet("background-color:transparent;")

        # bbox = self.fig.get_window_extent().transformed(self.fig.dpi_scale_trans.inverted())
        # width, height = bbox.width * self.fig.dpi, bbox.height * self.fig.dpi

        # create Axes host for the sake of axis management later - it will not be used for display
        self.ax_host = self.fig.add_subplot(111)
        self.ax_host.get_xaxis().set_visible(False)
        self.ax_host.get_yaxis().set_visible(False)

        # Axes for image display need to be created first (if not first then plot is not visible as behind image
        self.ax_image = self.ax_host.twinx()
        self.ax_image.set_anchor('SE')
        self.ax_image.get_xaxis().set_visible(False)
        self.ax_image.get_yaxis().set_visible(False)


        # then create Axe for plotting purpose with same y axis scale as the image so that it fits together
        # in same rectangle
        self.ax_plot = self.ax_host.twinx()
        self.ax_plot.set_anchor('SE')  # fraction of figure size
        self.ax_plot.set_autoscale_on(True)
        self.ax_plot.autoscale(True, axis='both', tight=True)
        self.ax_plot.get_xaxis().set_visible(True)
        self.ax_plot.yaxis.set_label_position('left')  # move axis on left as it is on right by default
        self.ax_plot.yaxis.set_ticks_position('left')

        self.ax_plot.get_yaxis().set_visible(True)

        # store current x axis length
        self._x_axis_length = None
        # store image currently displayed
        self._img_currently_displayed = None
        # Display option default value
        self._image_is_to_be_shown = None
        self._graph_is_to_be_shown = True
        # set font size for matplotlib envt
        mpl.rcParams.update({'font.size': 8})

    def set_image_to_be_shown(self):
        self._image_is_to_be_shown = True
        self.show_img(self._img_currently_displayed)

    def clear_image_to_be_shown(self):
        self._image_is_to_be_shown = False
        self.ax_image.cla()
        self.fig.canvas.draw_idle()

    def compute_and_display_figure(self, photo_container):
        stat_dict = photo_container.compute_statistics_interval_with_previous()
        # x = [index_ for index_ in range(len(ui.active_photos[1:]))]  # skip first element with interval = 0
        x = [index_ for index_ in range(len(ui.active_photos))]  # skip first element with interval = 0
        y_interval = [photo_container.interval_with_previous(picture) for picture in photo_container[1:]]
        y_interval.insert(0, 0)
        y_mean = [stat_dict["mean"] for _ in photo_container]
        y_stddevplus = [stat_dict["mean"] + stat_dict["Standard Deviation"] for _ in photo_container]
        y_stddevminus = [stat_dict["mean"] - stat_dict["Standard Deviation"] for _ in photo_container]

        self.show_graph(x, y_interval, y_mean, y_stddevplus, y_stddevminus)

    def show_img(self, img):
        """
        Display the image preserving its original form factor
        :param img: Numpy RGB image matplotlib comptaible
        :return: None
        """
        self._img_currently_displayed = img
        if self._image_is_to_be_shown:
            image = img
        else:
            image = np.zeros([1920, 1080, 3], dtype=np.uint8)
            image.fill(255)  # or img[:] = 255

        self.ax_image.cla()

        row, col, nb = image.shape
        img_ratio = col / row

        self.im = self.ax_image.imshow(image,
                                       extent=[0, self._x_axis_length - 1,
                                               0, (self._x_axis_length - 1) / img_ratio
                                               ]
                                       )
        self.fig.canvas.draw_idle()  # required to draw again

        return None

    def show_graph(self, x, y_interval, y_mean, y_stddevplus, y_stddevminus):

        self._x_axis_length = len(x)

        if self._graph_is_to_be_shown and len(x) != 0:
            self.ax_plot.cla()

            self.ax_plot.set_xlabel(" Number of shots ")
            self.ax_plot.set_ylabel(" seconds ")

            # then plot
            self.line_interval = self.ax_plot.plot(x, y_interval, 'b')
            self.line_mean = self.ax_plot.plot(x, y_mean, 'r')
            self.line_stddevplus = self.ax_plot.plot(x, y_stddevplus, 'm', dashes=[10, 5, 3, 3])
            self.line_stddevminus = self.ax_plot.plot(x, y_stddevminus, 'm', dashes=[10, 5, 3, 3])
            # and add legend
            self.fig.legend(
                (self.line_interval[0], self.line_mean[0], self.line_stddevplus[0], self.line_stddevminus[0])
                , ('interval', 'mean', 'stddevplus', 'stddevminus')
                , 'upper right')
            self.ax_plot.get_legend()
            # set cursor at O - do not remove else one of above line are removed in draw_vertical_cursor when the
            # ax_plot.lines.pop() is called in order to remove previous cursor vertical line
            self.ax_plot.axvline(x=0)

            self.ax_plot.set_autoscale_on(True)
            self.ax_plot.autoscale(True, axis='both', tight=True)

            self.fig.canvas.draw_idle()  # required to draw again else display_upon_sliderReleased_signal is not updated

            # Adjust picture to new axis dimension else we have desynchro between image size and graph size
            if self._img_currently_displayed is not None:
                self.show_img(self._img_currently_displayed)
            else:
                self.ax_image.cla()
                self.fig.canvas.draw_idle()

    def draw_vertical_cursor(self, row):
        """
        draw a vertical line at the position of the last selected row
        :return:
        """
        self.ax_plot.lines.pop()  # remove previous cursor
        self.ax_plot.axvline(x=row)
        self.fig.canvas.draw_idle()

    # TODO implement __repr__


class TreeViewPopulate:

    def __init__(self, gui_object):
        self._gui = gui_object  # store QWidget from ModelToViewController class
        # TODO make this global and no longer hidden in the code - it must be consistent with opencv
        # TODO capabilities and codecs available
        filters = ["*" + suffix for suffix in VALID_VIDEO_FILE_SUFFIXES]
        self.model = QFileSystemModel()
        self.model.setRootPath('C:\\')
        self.model.setFilter(QtCore.QDir.AllDirs | QtCore.QDir.NoDotAndDotDot | QtCore.QDir.AllEntries)
        self.model.setNameFilters(filters)
        self.model.setNameFilterDisables(False)
        self._gui.treeView.setModel(self.model)
        # remove unnecessary columns size, type and Date modified
        self._gui.treeView.setColumnHidden(1, True)
        self._gui.treeView.setColumnHidden(2, True)
        self._gui.treeView.setColumnHidden(3, True)

    # TODO implement __repr__


class ShowTableView:
    '''
    displays every list from left to right with a same vertical scrollbar on the rigth
    '''

    def __init__(self, guiObject, PhotoList):
        self._gui = guiObject  # store QWidget from ModelToViewController class
        # __class__.list_of_lists = build_list_of_list(PhotoList)  # Build lists required to display_upon_sliderReleased_signal
        # for title in TAG_DICTIONNARY.keys():
        #     __class__.title_of_lists.append(title)
        # __class__.title_of_lists.append(" Interval (secs) ")
        self.tm = MyTableModel(ui.active_photos, TAG_DICTIONNARY)
        self._gui.tableView.setModel(self.tm)                          # TODO CA PLANTE ICI /!\
        header = self._gui.tableView.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeToContents)  # have columns width adjusted to content
        self._gui.tableView.selectionModel().selectionChanged.connect(self.handle_selection_changed)

    def handle_selection_changed(self, selected, deselected):
        """
        Upon new row selected or deselected load corresponding picture, update cursor position on the statistic graph
        and move rows in the table display so that selected row is visible

        :param selected: list of row selected
        :param deselected: list of row deselected
        :return:
        """
        row_selected_by_selection_order \
            = [index.row() for index in self._gui.tableView.selectionModel().selectedRows()]

        # Display image of the last selected picture
        if len(row_selected_by_selection_order) != 0:
            row = row_selected_by_selection_order[-1]

            self._gui.my_slider.display_by_row(row)
            self._gui.my_slider.set_cursor(row)
            # Show vertical cursor on matplotlib graph
            self._gui.my_graph.draw_vertical_cursor(row)

            # scroll tableView so that selected row is visible
            if not self._gui.tableView.verticalScrollBar().sliderPosition() \
                   < row <= \
                   self._gui.tableView.verticalScrollBar().sliderPosition() \
                   + self._gui.tableView.verticalScrollBar().pageStep():
                self._gui.tableView.verticalScrollBar().setSliderPosition(
                    min(row, self._gui.tableView.verticalScrollBar().maximum())
                )

    # TODO implement __repr__


class MyTableModel(QAbstractTableModel):
    '''

        photo_ordered_collection:
                    is an instance of the class PhotoOrderedCollectionByCapturetime containing instances of PhotoWithMetadata class
                    to be treated by the model

        title_dict : Dictionary of list
                        keys stating title of columns
                        values being a list containing tags in decreasing priority of search

    '''

    # TODO to be complemented
    def __init__(self, photo_ordered_collection, title_dict, parent=None, *args):
        QAbstractTableModel.__init__(self, parent)
        self.photo_container = photo_ordered_collection
        self.title_and_tags = title_dict  # this is an Ordered Dictionary
        # build list of tiltes
        self.title_list = list(title_dict.keys())

    def rowCount(self, parent=None):
        return len(self.photo_container)

    def columnCount(self, parent):
        return len(self.title_list) + 2
        # + 2 because in addition to metatags we show:
        #   +1 whether clone or real picture and if clone what is the duplication method
        #   +1 we show interval with previous picture

    def data(self, index, role):
        if not index.isValid():
            return QVariant()
        elif role == Qt.TextAlignmentRole:
            return Qt.AlignCenter
        elif role == Qt.TextColorRole:
            if isinstance(self.photo_container[index.row()], PhotoCloned):
                return QVariant(QColor(Qt.blue))
            else:
                return QVariant(QColor(Qt.black))
        elif role == Qt.FontRole:
            font = QFont()
            if isinstance(self.photo_container[index.row()], Photo):
                font.setWeight(QFont.DemiBold)
                return QVariant(font)
            else:  # PhotoClone in Italic
                font.setItalic(True)
                return QVariant(font)
        elif role != Qt.DisplayRole:
            return QVariant()
        elif index.column() < len(self.title_and_tags):  # requesting tags but not interval that is last column
            return QVariant(self.photo_container[index.row()].get_tag_value(
                self.title_and_tags[
                    self.title_list[index.column()]
                ]
            )
            )
            # return QVariant(self.photo_container[position.row()].get_tag_value(self.title_and_tags[position.column()]))
            # TODO PhotoWithMetadata.get_tag_value  to be implemented
        elif index.column() == len(self.title_and_tags):
            text = "N/A"
            if isinstance(self.photo_container[index.row()], Photo): text = "original"
            if isinstance(self.photo_container[index.row()], PhotoCloned):
                text = self.photo_container[index.row()].duplicate_method
            return QVariant(text)
        else:  # computed interval to be returned in last position
            return QVariant(self.photo_container.interval_with_previous(self.photo_container[index.row()]))

    def headerData(self, col, orientation, role):
        if orientation == Qt.Horizontal and role == Qt.DisplayRole:
            if col < len(self.title_and_tags):  # tags
                return QVariant(self.title_list[col])
            elif col == len(self.title_and_tags):
                return QVariant("type - clone \n duplicate method")
            else:  # interval in last column
                return QVariant("Interval (secs)")
        elif orientation == Qt.Horizontal and role == Qt.FontRole:
            font = QFont()
            font.setBold(True)
            return QVariant(font)
        elif orientation == Qt.Vertical and role == Qt.DisplayRole:
            return QVariant(col + 1)

    def insertRows(self, foto, index=QModelIndex()):
        #  determine what is the position for insertion (row)
        if len(self.photo_container) == 0:
            row = 0
        else:
            row = len(self.photo_container)  # by default we add at the end
            for i, picture in enumerate(self.photo_container):  # else we identify picture taken right before
                if self.photo_container[i].shot_timestamp > foto.shot_timestamp:
                    row = i
                    break
        # insert row
        self.beginInsertRows(QModelIndex(), row, row)  # row, row+1  = first row and last row of insert
        self.photo_container.add(foto)
        self.endInsertRows()
        # TODO assumption that interval is last column - remove this in the model if possible
        return True

    def removeRows(self, row, index=QModelIndex()):
        self.beginRemoveRows(QModelIndex(), row, row)
        del self.photo_container[row]
        self.endRemoveRows()
        return True


class SliderManager:

    def __init__(self, gui_oject):
        self.gui = gui_oject
        # self.gui.picture_slider.valueChanged.connect(lambda: print("SliderManager __init__ __init__ __init__ "))
        # IT WORKS
        self.gui.picture_slider.setMinimum(0)
        self.gui.picture_slider.setMaximum(100)
        self.gui.picture_slider.setValue(0)
        # connect button slots
        self.gui.picture_slider.sliderReleased.connect(
            lambda: self.display_upon_sliderReleased_signal(self.gui.picture_slider.value())
        )

    def reset_slider_cursor_to_zero(self):
        self.gui.picture_slider.setValue(0)

    def set_cursor(self, row_value):
        # convert row_value in %
        value_ = 0
        if len(self.gui.active_photos) > 1:
            value_ = (row_value / (len(self.gui.active_photos) - 1)) * 100
        self.gui.picture_slider.setValue(value_)

    def display_by_row(self, index_):

        if index_ <= len(self.gui.active_photos) - 1:
            if self.gui.active_photos.load_image_previews_in_memory(index_):  # display picture
                self.gui.my_graph.show_img(self.gui.active_photos[index_].get_matplotlib_image_preview())
                # len(self.gui.active_photos))
            else:  # inform that file type is not supported
                self.gui.picture_label.setText(
                    str(self.gui.active_photos[index_].get_tag_value(["File:FileType"]))
                    + " extension is not yet supported for display")
        else:
            logger.warning("function called with index %s higher than nb of photos loaded %s",
                           str(index_), str(len(self.gui.active_photos) - 1))

    def display_upon_sliderReleased_signal(self, slider_value):
        """
        called upon sliderReleased signal from the slider Widget. In this case the corresponding row is selected.
        picture display and graph vertical cursor are triggered by the slot connected to the selectionChanged signal
        that is emitted when the selection is changed
        therefore they are not called in this method to avoid double work

        :param slider_value: slider value that is between 0 and 99
        :return:
        """
        # convert slider value in Photo_container index
        if len(self.gui.active_photos) > 1:
            index_ = int((slider_value / 100) * (len(self.gui.active_photos) - 1))
        else:
            index_ = 0

        # self.gui.tableView.clearSelection()
        # index_to_be_selected is a QModelIndex() object
        index_to_be_selected = self.gui.tableView.model().index(index_, 0)
        self.gui.tableView.selectionModel().select(
            index_to_be_selected,
            QItemSelectionModel.Select | QItemSelectionModel.Rows
        )
        self.gui.tableView.setFocus()

    def slider_connect_valueChanged_signal_slot(self):

        self.gui.picture_slider.valueChanged.connect(
            lambda: self.display_upon_slider_valueChanged_signal(self.gui.picture_slider.value())
        )

    def display_upon_slider_valueChanged_signal(self, slider_value):

        # convert slider value in Photo_container index
        if len(self.gui.active_photos) > 0:
            index_ = int((slider_value / 100) * (len(self.gui.active_photos) - 1))
        else:
            index_ = 0

        self.display_by_row(index_)
        # Show vertical cursor on matplotlib graph
        self.gui.my_graph.draw_vertical_cursor(index_)


class StatusProgressBar(QProgressBar):

    def __init__(self, gui_object):
        super().__init__()
        self.setValue(0)
        gui_object.statusBar().addWidget(self)

    def progress_reset_to_zero(self):
        self.setValue(0)
        QtWidgets.QApplication.processEvents()

    def progress_five_percent(self):
        self.setValue(self.value() + 5)
        QtWidgets.QApplication.processEvents()


class StatusProgressBarTicker:
    '''
    initialized with the number of "ticks" that will be sent to reach 100% completion
    and increase ProgressBar by 5% chunk upon tick received via the self.tick() method

    ProgressBar value is incremented indirectly via the emitting of a PyQT signal (IncreaseProgressBarBy5Pc) that
    is connected to a slot responsible for execution within the ModelToViewController Class - So progressBar implementation can varies
    independently from this class

    In current implementation assumption is that we use a unique and same ProgressBar displayed within the status bar
    for the whole program whatever the action progress to be shown but should it be required it could change in
    future by adding to this class a parameter stating which progressBar is to be addressed
    '''

    def __init__(self, total_nb_of_ticks):
        self.tick_counter = 0
        self.five_percent_chunk_occurence = 0
        self.five_percent_chunk_value = round(total_nb_of_ticks) * 0.05

    def tick(self):
        self.tick_counter += 1
        if self.tick_counter // self.five_percent_chunk_value > self.five_percent_chunk_occurence:
            self.five_percent_chunk_occurence += 1
            ui.IncreaseProgressBarBy5Pc.emit()
            QtWidgets.QApplication.processEvents()

    # TODO implement __repr__


class RollBackHeap:  # TODO NEED TO BE FIXED IN LINE WITH CLONESET MODEL RECENTLY IMPLEMENTED

    def __init__(self):
        self._heap = []  # contains couples of (verb , [list of operation to rollback] in FILO mode

    def __len__(self):
        return len(self._heap)

    def append(self, verb, rollback_list):
        self._heap.append((verb, rollback_list))
        return

    def pop(self):
        return self._heap.pop()

    def rollback(self):
        if len(self._heap) == 0:
            QMessageBox.about(ui, PROGRAM_NAME, "Roll Back history is exhausted - No Roll Back possible ")
            # TODO centralize all message sending stuff within ModelToViewController class - Dirty to call ui here !
            return
        # retrieve picture to be inserted
        verb, list_of_pictures = self._heap.pop()

        # roll back
        if verb == "remove":  # roll back the removed pictures b re insetring them
            for foto in list_of_pictures:
                ui.myTable.tm.insertRows(foto)
                ui.discarded_photos.remove(foto)
        elif verb == "duplicate":  # remove pictures created by duplication
            for foto in list_of_pictures:
                ui.myTable.tm.removeRows(ui.active_photos.position(foto))
        else:
            raise ValueError

        # update view : summary and graph
        ui.update_view_but_table()

    def reset(self):
        self._heap = []
        return

    # TODO implement __repr__


class ShowVideoController(QMainWindow, Ui_PictureWindow):
    """
    This class manages the window that allows to display selected pictures frame per frame
    or as video at various fps if pictures are contiguous
    It takes as an input a list of Qpixmap object which size must be equal to the labelQpixmap Qlabel.
    Getting the size of labelQpixmap must be done via the get_image_size() method prior to submiting the video
    for display with display method
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setupUi(self)

        INITIAL_SLIDER_VALUE = 48  # slider value is twice the fps
        # initialize attributes
        # will receive the list of right sized numpy BGR photo
        self.list_of_right_sized_Pixmap = None
        # will store index of current picture displayed on screen in the list
        self.index_ = None
        # reference to the starting row position so that it can be displayed on screen
        self.row_start = None
        # state if video is playing
        self.is_playing = False

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.handle_display_next_picture)

        self.fpsSlider.setMinimum(2)  # 1 fps
        self.fpsSlider.setMaximum(120)  # up to 60 fps
        self.fpsSlider.setValue(INITIAL_SLIDER_VALUE)
        self.timer_value_ms = INITIAL_SLIDER_VALUE / 2  # 24 fps
        self.fpsValueQlabel.setText(str(INITIAL_SLIDER_VALUE / 2))
        self.fpsSlider.sliderReleased.connect(lambda: self.set_fps(self.fpsSlider.value()))
        self.fpsSlider.valueChanged.connect(lambda: self.set_fps_display(self.fpsSlider.value()))

        # connect button slots
        self.previousButton.clicked.connect(self.handle_display_previous_picture)
        self.nextButton.clicked.connect(self.handle_display_next_picture)
        self.playButton.clicked.connect(self.handle_play_video)
        self.pauseButton.clicked.connect(self.handle_pause_video)
        self.resumeButton.clicked.connect(self.handle_resume_video)

        # initialize statusbar
        self.statusbar_message_qlabel = QLabel("Ready for video play")
        self.statusBar.addWidget(self.statusbar_message_qlabel)

    def get_image_size(self):
        width = self.labelQpixmap.frameGeometry().width()
        height = self.labelQpixmap.frameGeometry().height()
        return (width, height)

    def set_fps(self, slider_value):
        fps = int(slider_value / 2)
        self.timer_value_ms = int(1000 / fps) - 5  # 5 ms is time to execute display
        if self.is_playing:
            self.timer.stop()
            self.timer.start(self.timer_value_ms)

    def set_fps_display(self, slider_value):
        self.fpsValueQlabel.setText(str(int(slider_value / 2)))

    def update_image_number_on_status_bar(self):
        """
        Display row number and image number in sequence in status bar
        row number is the position of the picture in the collection so that one can refer to it in main Qtableview
                   in order to adapt the duplicate method settings
        image number is the position within the subset currently displayed
        :return:
        """
        self.statusbar_message_qlabel.setText("Row: " + str(self.row_start + self.index_ + 1)
                                              + " - Image: " + str(self.index_ + 1) + "/"
                                              + str(len(self.list_of_right_sized_Pixmap)))

    def display(self, list_of_right_sized_Pixmap, row_start=0):

        if len(list_of_right_sized_Pixmap) == 0:
            return None

        # store the list of photos as an object attribute
        self.list_of_right_sized_Pixmap = list_of_right_sized_Pixmap
        # Display first picture in list
        self.index_ = 0
        self.row_start = row_start
        self.labelQpixmap.setPixmap(self.list_of_right_sized_Pixmap[self.index_])
        self.update_image_number_on_status_bar()

    def handle_display_previous_picture(self):

        if self.index_ != 0:
            self.index_ -= 1
            self.labelQpixmap.setPixmap(self.list_of_right_sized_Pixmap[self.index_])
        self.update_image_number_on_status_bar()

    def handle_display_next_picture(self):

        if self.index_ < len(self.list_of_right_sized_Pixmap) - 1:
            self.index_ += 1
            self.labelQpixmap.setPixmap(self.list_of_right_sized_Pixmap[self.index_])
        else:
            self.timer.stop()
            self.index_ = 0  # reset index at first position when done so that it can be played again
        self.update_image_number_on_status_bar()

    def handle_play_video(self):
        self.index_ = 0
        self.is_playing = True
        self.timer.start(self.timer_value_ms)  # 24 fps

    def handle_pause_video(self):
        self.timer.stop()
        self.is_playing = False

    def handle_resume_video(self):
        self.timer.start(self.timer_value_ms)  # 24 fps

    def handle_not_implemented(self):
        QMessageBox.about(self, PROGRAM_NAME, "method not yet implemented...be patient !")


class ModelToViewController(QMainWindow, Ui_MainWindow):
    ToggleProgressBar = pyqtSignal()  # signal to show or close progress bar for loading file
    IncreaseProgressBarBy5Pc = pyqtSignal()  # signal to move progress Bar by 5 pc
    BackgroundPictureLoadCompleted = pyqtSignal()  # signal background load of picture is completed

    def __init__(self, parent=None):
        super(ModelToViewController, self).__init__(parent)
        self.setupUi(self)

        # create PhotoWithMetadata containers for active photos and discarded ones
        self.active_photos = PhotoOrderedCollectionByCapturetime()
        self.active_photos_backup = PhotoOrderedCollectionByCapturetime()
        self.discarded_photos = PhotoNonOrderedCollection()
        self.discarded_photos_backup = PhotoNonOrderedCollection()
        self.commit_history = RollBackHeap()
        self.commit_history_backup = RollBackHeap()

        # list that will store threads launched to perform picture loading in background
        self.thread_list = []
        # state whether all preview are loaded or not
        self.image_preview_load_completed = True  # initialized at True so that it passes completion test 1st timme

        # long live attribute to hold progressbar instance
        self.progress_bar = None

        # status bar permanent header
        self.statusbar.addPermanentWidget(QLabel("  |  " + PROGRAM_NAME + " " + VERSION))
        # ..and initial message
        self.statusbar_message_qlabel = QLabel("Ready for file loading - Double click directory in left pane")
        self.statusbar.addWidget(self.statusbar_message_qlabel)

        # display_upon_sliderReleased_signal Directory tree view so that folder to be loaded can be chosen
        self.my_directory = TreeViewPopulate(self)
        self.treeView.doubleClicked.connect(self.handle_load_files_from_directory)

        # initialize matplotlib canvas and size of preview image in PhotoWithMetadata class based on size of matplotlibwidget
        self.my_graph = ImageAndPlotDisplayWithMatplotlib(self.matplotlibwidget)
        PhotoWithMetadata.set_matplotlib_image_preview_size(self.matplotlibwidget.frameGeometry().width() * 0.95
                                                            , self.matplotlibwidget.frameGeometry().height() * 0.95
                                                            )

        # initialize picture display_upon_sliderReleased_signal
        self.my_slider = SliderManager(self)
        self.my_slider.slider_connect_valueChanged_signal_slot()

        # connect button slots
        self.RemovePushButton.clicked.connect(self.handle_remove_button)
        self.DuplicatePushButton.clicked.connect(self.handle_duplicate_button)
        self.RollbackPushButton.clicked.connect(self.commit_history.rollback)  # rollback method of RollBackHeap object
        self.ExtractPushButton.clicked.connect(self.handle_extract_button)
        self.ClearSelectionPushButton.clicked.connect(self.handle_clear_selection_button)
        self.showvideoButton.clicked.connect(self.handle_show_video_button)
        self.PicklePushButton.clicked.connect(self.handle_pickle_button)
        self.UnpicklePushButton.clicked.connect(self.handle_unpickle_button)
        self.ShowImageRadioButton.setChecked(False)

        # self.ShowImageRadioButton.toggled.connect(self.my_graph.toggle_image_to_be_shown)
        self.ShowImageRadioButton.toggled.connect(self.toggle_image_to_be_shown)

        # connect Menu action to their respective handle
        self.actionExit.triggered.connect(self.handle_exit_menu)
        self.actionsimple_copy.triggered.connect(self.handle_duplicate_method_set_to_simple_copy_menu)
        self.actioninterpolate.triggered.connect(self.handle_duplicate_method_set_to_interpolate_menu)
        self.actionlucas_kanade.triggered.connect(self.handle_duplicate_method_set_to_lucas_kanade_menu)
        self.actiongunner_farnerback.triggered.connect(self.handle_duplicate_method_set_to_gunner_farnerback_menu)

        # connect progress bar pyQtsignals to their slots
        self.ToggleProgressBar.connect(self.toggle_progress_bar)
        self.IncreaseProgressBarBy5Pc.connect(self.increment_progress_bar_by_5_percent)
        self.BackgroundPictureLoadCompleted.connect(self.upon_background_picture_load_completed)

        # initialize duplicate method
        self.duplicate_method = "lucas_kanade"
        logger.info("Duplicate method set to lucas kanade mode")

    def _save_context(self):
        self.commit_history_backup = copy.copy(self.commit_history)
        self.active_photos_backup = copy.copy(self.active_photos)
        self.discarded_photos_backup = copy.copy(self.discarded_photos)
        self.status_message_backup = self.statusbar_message_qlabel.text()
        return True

    def _restore_context(self):
        self.commit_history = copy.copy(self.commit_history_backup)
        self.active_photos = copy.copy(self.active_photos_backup)
        self.discarded_photos = copy.copy(self.discarded_photos_backup)
        self.statusbar_message_qlabel.setText(self.status_message_backup)
        # TODO centralise statusbar management in a class that hides the "Qt tringlerie" from application logic
        return True

    def toggle_progress_bar(self):
        if not self.progress_bar:
            self.progress_bar = StatusProgressBar(self)
            self.progress_bar.hide()
        if self.progress_bar.isVisible():
            self.progress_bar.hide()
            self.progress_bar.progress_reset_to_zero()
        else:
            self.progress_bar.show()

    def increment_progress_bar_by_5_percent(self):
        self.progress_bar.progress_five_percent()

    def update_view_but_table(self):
        '''
        update List and graph and in future other display_upon_sliderReleased_signal element after tableview is updated
        :return:
        '''
        #  update summary
        self.my_list.update()
        #  update graph
        self.my_graph.compute_and_display_figure(self.active_photos)

    def handle_load_files_from_directory(self, signal):
        """
        This method is calles when TreeView is double clicked
        File_path provided can be either a directory or a video file
        In case it is a directory it is interpreted as a collection of picture to be loaded
        else treatement for a video is triggered
        It is based on the same methods being implemented in both
            - PhotoOrderedCollectionFromVideoRead
            - PhotoOrderedCollectionByCapturetime
        :param signal:
        :return: nothing
        """
        file_path = self.treeView.model().filePath(signal)

        # detect if this video or directory of image
        if "." + file_path.split(".")[-1] in VALID_VIDEO_FILE_SUFFIXES:
            # this is a video
            logger.info("VIDEO DETECTED %s", file_path)
            input_media_is_file = False
            # store video file path in PhotoOrderedCollectionFromVideoRead class attribute
            PhotoOrderedCollectionFromVideoRead.set_video_file_path(file_path)
        else:
            input_media_is_file = True  #  default is set of photo files in a directory
            logger.info("FOLDER OF FILE DETECTED %s", file_path)

        # create Photo containers for active photos and discarded ones depending on loading case
        # set directory and compute number of ticks for progress bar increments
        if input_media_is_file:
            self.active_photos = PhotoOrderedCollectionByCapturetime()
            self.active_photos_backup = PhotoOrderedCollectionByCapturetime()
            self.discarded_photos = PhotoNonOrderedCollection()
            self.discarded_photos_backup = PhotoNonOrderedCollection()
            os.chdir(file_path)
            nb_ticks = 2 * len(os.listdir(file_path))  # 2 tick per file
        else:  # this is a video
            self.active_photos = PhotoOrderedCollectionFromVideoRead()
            self.active_photos_backup = PhotoOrderedCollectionFromVideoRead()
            self.discarded_photos = PhotoNonOrderedCollection()
            self.discarded_photos_backup = PhotoNonOrderedCollection()
            head, tail = os.path.split(file_path)  # remove filename
            os.chdir(head)
            # get number of frames in video
            cap = cv2.VideoCapture(file_path)
            nb_ticks = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()

        self.commit_history = RollBackHeap()
        self.commit_history_backup = RollBackHeap()

        # stop background thread loading pictures in background if still running
        if not self.image_preview_load_completed:
            self.active_photos.set_stop_background_preview_load_event()

        # wipe PhotoWithMetadata containers after taking copy of them and their path in case no file is loaded
        keep_working_directory = os.getcwd()
        self._save_context()
        self.active_photos.reset()
        self.discarded_photos.reset()
        self.commit_history.reset()

        self.toggle_progress_bar()  # display_upon_sliderReleased_signal progress bar
        progress_bar_ticker = StatusProgressBarTicker(nb_ticks)  # two stages per file
        QtWidgets.QApplication.processEvents()

        logger.info(" START PHOTO META DATA EXTRACTION OF TYPE %s FROM %s", VALID_PHOTO_FILE_SUFFIXES, str(file_path))
        self.ShowImageRadioButton.setChecked(False)

        if self.active_photos.load_metadata_from_files(file_path , VALID_PHOTO_FILE_SUFFIXES, progress_bar_ticker.tick) > 0:
            self.statusbar_message_qlabel.setText("pictures from :" + str(file_path))
            msgbox_txt = "Photos successfully loaded"
        else:  # no picture found restore previous folder
            os.chdir(keep_working_directory)
            self._restore_context()
            msgbox_txt = "No elligible file found in {}\nprevious folder remain loaded".format(str(file_path))

        if __debug__:
            logger.debug("PHOTO LOADED ABOUT TO START TableView Display")

        # load compute and display_upon_sliderReleased_signal PhotoWithMetadata details, Summary and Graph of Intervals
        self.myTable = ShowTableView(self, self.active_photos)
        self.my_list = ShowSummary(self.listView, self.active_photos)
        self.my_graph.compute_and_display_figure(self.active_photos)
        self.my_slider.display_by_row(0)
        self.my_slider.set_cursor(0)
        self.toggle_progress_bar()  # close progress bar
        self.status_message_preserved = self.statusbar_message_qlabel.text()
        self.statusbar_message_qlabel.setText("loading image previews...")
        progress_bar_ticker = StatusProgressBarTicker(len(self.active_photos))
        self.image_preview_load_completed = False

        # start or restart background threads that will load pictures in memory in background
        self.thread_list = self.active_photos.launch_background_picture_loader_threads(
            progress_bar_ticker.tick,
            self.BackgroundPictureLoadCompleted.emit
        )
        # inform about status of load
        QMessageBox.about(self, PROGRAM_NAME, msgbox_txt)
        logger.info(" %s PHOTOS LOADED in Photo Class ", str(len(self.active_photos)))
        self.toggle_progress_bar()  # close progress bar
        logger.info(" %s PHOTOS LOADED in Photoxxxx Class ", str(len(self.active_photos)))

        return

    def toggle_image_to_be_shown(self):

        if self.ShowImageRadioButton.isChecked():
            self.my_graph.set_image_to_be_shown()
        else:
            self.my_graph.clear_image_to_be_shown()

    def upon_background_picture_load_completed(self):
        self.image_preview_load_completed = True
        self.toggle_progress_bar()
        self.statusbar_message_qlabel.setText(self.status_message_preserved)
        self.my_slider.slider_connect_valueChanged_signal_slot()
        self.ShowImageRadioButton.setChecked(True)  # check radio button and activate display of image

    def handle_remove_button(self):

        # save rows selected
        selection = sorted([index.row() for index in self.tableView.selectionModel().selectedRows()])

        self.myTable.tm.layoutAboutToBeChanged.emit()

        status, \
        message, \
        removed_pictures, \
        could_not_remove_rows = \
            self.active_photos.remove_list_of_photos(selection)

        self.myTable.tm.layoutChanged.emit()

        if not status:
            text = "Remove failed for below images: \n\n"
            for image, msg in could_not_remove_rows.items():
                text += str(image.file_name) + " : " + msg + "\n"
            QMessageBox.about(self, PROGRAM_NAME, text)
        # store list of pictures for future roll back

        if len(removed_pictures) != 0:
            self.commit_history.append("remove", removed_pictures)
        for pict in removed_pictures:
            self.discarded_photos.append(pict)

        # update view : summary and graph
        self.update_view_but_table()
        # deselect all selected rows
        self.tableView.clearSelection()

        # self.gui.tableView.setFocus()

    def handle_duplicate_button(self):
        """
        Duplicate picture selected in tableView by creating a PhotoCloned instance

        PhotoCloned instances keeps info of which real picture they are cloned from. Therefore it is taken care of
        the case where a photo is duplicated from an already duplicated picture by walking up the chain link until the
        original real picture is identified.


        :return: True if successful - False if selection is empty or composed only of last row of tableView
        """
        # save rows selected
        selection = sorted([index.row() for index in self.tableView.selectionModel().selectedRows()])
        try:  # check selection is not empty
            selection_not_empty = selection[0]

            self.myTable.tm.layoutAboutToBeChanged.emit()

            status, \
            message, \
            duplicated_pictures, \
            could_not_duplicate_rows = \
                self.active_photos.duplicate_list_of_photos(selection, self.duplicate_method)

            self.myTable.tm.layoutChanged.emit()

            if not status:
                text = "Duplicate failed for below images: \n\n"
                for image, msg in could_not_duplicate_rows.items():
                    text += str(image.file_name) + " : " + msg + "\n"
                QMessageBox.about(self, PROGRAM_NAME, text)

            # keep history for potential roll back
            if len(duplicated_pictures) > 0:
                self.commit_history.append("duplicate", duplicated_pictures)
                # update view : summary and graph
                self.update_view_but_table()
            # deselect all selected rows
            self.tableView.clearSelection()
            # self.gui.tableView.setFocus()
        except IndexError:
            return False  # empty selection list
        return True

    def handle_extract_button(self):
        """

        :return:
        """
        self.ToggleProgressBar.emit()
        ProgressBarTicker = StatusProgressBarTicker(len(self.active_photos))

        status, \
        message, \
        img_cv2_list = \
            self.active_photos.generate_computed_pictures(
                output="file",
                file_treated_tick_function_reference=ProgressBarTicker.tick
            )


        self.ToggleProgressBar.emit()

        if status:
            message = "Extract completed"

        QMessageBox.about(ui, PROGRAM_NAME, message)

        return True

    def handle_set_extension_menu(self):
        print('Set Extension  clicked')

    def handle_show_video_button(self):
        """
        Display on a new modal window the set of picture selected
        It allows frame per frame visualization or playing the full set as a video which fps can be choosen

        :return:
        """
        self.sv = ShowVideoController(parent=self)  # pass self as parent else it won't be a modal window
        width, height = self.sv.get_image_size()

        selection = sorted([index.row() for index in self.tableView.selectionModel().selectedRows()])

        if len(selection) > 0:
            logger.info("START SHOW VIDEO PREPARATION FOR %s PHOTO", str(len(selection)))
            # check that picture selected are consecutives
            is_not_consecutive = False
            for index, row_ in enumerate(selection):
                if index == 0:
                    pass
                elif selection[index - 1] != selection[index] - 1:
                    is_not_consecutive = True
                    break
            if is_not_consecutive:
                QMessageBox.about(self, PROGRAM_NAME,
                                  " Pictures must be consecutives - select a consecutive set of pictures before")
                return

            status, \
            message, \
            img_qpixmap_list = \
                self.active_photos.generate_computed_pictures(
                    output="qpixmap",
                    row_start=selection[0],
                    row_stop=selection[-1],
                    size=(width, height)
                    # file_treated_tick_function_reference=ProgressBarTicker.tick
                )

            self.sv.display(img_qpixmap_list, selection[0])

            logger.info("SHOW VIDEO PREPARATION COMPLETED")

            self.sv.show()
        else:
            QMessageBox.about(self, PROGRAM_NAME, " No picture selected - select a contiguous set of pictures before"
                                                  " pushing Show Video button")

    def contextMenuEvent(self, QContextMenuEvent):
        pos = QContextMenuEvent.globalPos()  # get cursor global position

        row = self.tableView.rowAt(
            self.tableView.viewport().mapFromGlobal(pos).y())
        # test if cursor is actually within tableView boundary - if not no action triggered
        if row == -1 \
                or self.tableView.columnAt(
            self.tableView.viewport().mapFromGlobal(pos).x()) == -1: return

        # catches mouse rigth click events
        self.menu = QMenu(self)
        displayTagAction = QAction('Display Tags...', self)
        duplicateAction = QAction('Duplicate', self)
        # NplicateAction = QAction('Nplicate next P rows...', self)
        removeAction = QAction('Remove', self)
        removeNoPAction = QAction('Remove N picture out of P...', self)
        self.menu.addAction(displayTagAction)
        self.menu.addAction(duplicateAction)
        # self.menu.addAction(NplicateAction)
        self.menu.addAction(removeAction)
        self.menu.addAction(removeNoPAction)
        displayTagAction.triggered.connect(lambda: self.display_tag_from_rigth_click(pos))
        duplicateAction.triggered.connect(lambda: self.duplicate_row_from_rigth_click(pos))
        # NplicateAction.triggered.connect(lambda: self.Nplicate_next_P_rows_from_rigth_click(pos))
        removeAction.triggered.connect(lambda: self.remove_row_from_rigth_click(pos))
        removeNoPAction.triggered.connect(lambda: self.remove_N_rows_out_of_P_from_rigth_click(pos))

        if isinstance(self.active_photos[row], PhotoCloned):
            duplicateMethodAction = QAction("Set duplicate method...", self)
            self.menu.addAction(duplicateMethodAction)
            duplicateMethodAction.triggered.connect(
                lambda: self.set_duplicate_method_from_rigth_click(pos)
            )
            if self.active_photos[row].clone_set.duplicate_method == "lucas_kanade":
                lukasKanadeAction = QAction("Display & Set lucas kanade parameters...", self)
                self.menu.addAction(lukasKanadeAction)
                lukasKanadeAction.triggered.connect(
                    lambda: self.set_lucas_kanade_parameter_from_rigth_click(pos)
                )

        self.menu.popup(pos)

    def display_tag_from_rigth_click(self, pos):
        print("hello from rigth click display_upon_sliderReleased_signal tag")

    def duplicate_row_from_rigth_click(self, pos):
        row = self.tableView.rowAt(
            self.tableView.viewport().mapFromGlobal(pos).y()
        )

        if row == len(self.active_photos) - 1:
            self.statusbar.showMessage("Can't duplicate last row")
            return

        self.myTable.tm.layoutAboutToBeChanged.emit()
        status, message, duplicated_picture = self.active_photos.duplicate_photo(row, self.duplicate_method)
        self.myTable.tm.layoutChanged.emit()

        if not status:
            QMessageBox.about(self, PROGRAM_NAME, message)
            return

        # keep history for potential roll back
        self.commit_history.append("duplicate", [duplicated_picture])
        # update view : summary and graph
        self.update_view_but_table()

    def Nplicate_next_P_rows_from_rigth_click(self, pos):

        row_start = self.tableView.rowAt(
            self.tableView.viewport().mapFromGlobal(pos).y()
        )

        # get input parameter from Qdialog
        NplicateQDialog = QtWidgets.QDialog()
        ui_nop = Ui_NplicateQDialog()
        ui_nop.setupUi(NplicateQDialog)
        ui_nop.duplicateMethodComboBox.insertItems(0, DuplicateMethod.get_duplicate_method_supported_list())
        while True:
            if NplicateQDialog.exec_() == 0: return  # cancel button pushed do nothing and return
            if (ui_nop.spinBox_NbRow.value() < self.myTable.tm.rowCount() - row_start): break
        nb_of_times_to_duplicate = ui_nop.spinBox_N.value()  #
        nb_row_to_be_treated = ui_nop.spinBox_NbRow.value()  #
        duplicate_method_selected = ui_nop.duplicateMethodComboBox.currentText()

        # self.picture_slider.valueChanged.emit(1)
        # print(self.picture_slider.receivers(self.picture_slider.valueChanged))

        # duplicated_pictures = []
        # for i in range(nb_row_to_be_treated):
        #     picture_index = row_start + (i * (nb_of_times_to_duplicate + 1))
        #     for _ in range(nb_of_times_to_duplicate):
        #         inner_picture_index = picture_index
        #         # identify real PhotoWithMetadata to clone from in case preceding photo is already virtual
        #         cloned_from = self.active_photos[inner_picture_index]
        #         if isinstance(cloned_from, PhotoCloned):
        #             cloned_from = cloned_from.cloned_from
        #         cloned_from_index = self.photo_virtual_clone_index_manager.get_next_index(cloned_from.file_name)
        #         # create and stores virtual photos
        #         virtual_foto = PhotoCloned(cloned_from,
        #                                    cloned_from_index,
        #                                    self.active_photos[inner_picture_index],
        #                                    self.active_photos[inner_picture_index + 1]
        #                                    )
        #         inner_picture_index += 1
        #         # insert virtual cloned pictures in tableView and active_Photos collection
        #         self.myTable.tm.insertRows(virtual_foto)
        #         # keep history for potential roll back
        #         duplicated_pictures.append(virtual_foto)
        #
        # self.commit_history.append("duplicate", duplicated_pictures)
        # # update view : summary and graph
        # self.update_view_but_table()

    def remove_row_from_rigth_click(self, pos):

        row = self.tableView.rowAt(
            self.tableView.viewport().mapFromGlobal(pos).y()
        )

        # store picture for future roll back before it is deleted
        self.commit_history.append("remove", [self.active_photos[row]])
        self.discarded_photos.append(self.active_photos[row])

        self.myTable.tm.layoutAboutToBeChanged.emit()
        status, message = self.active_photos.remove_photo(row)
        self.myTable.tm.layoutChanged.emit()

        if not status:
            self.commit_history.pop()  # remove not performed - clean-up history log
            self.discarded_photos.pop()
            QMessageBox.about(self, PROGRAM_NAME, message)
            return

        # update view : summary and graph
        self.update_view_but_table()

    def remove_N_rows_out_of_P_from_rigth_click(self, pos):

        row_start = self.tableView.rowAt(
            self.tableView.viewport().mapFromGlobal(pos).y()
        )

        # get input parameters from QDialog
        RemoveNoPQDialog = QtWidgets.QDialog()
        ui_nop = Ui_RemoveNoPQDialog()
        ui_nop.setupUi(RemoveNoPQDialog)
        while True:
            if RemoveNoPQDialog.exec_() == 0: return  # cancel button pushed do nothing and return
            # check parameters consistency
            if ((ui_nop.spinBox_N.value() < ui_nop.spinBox_P.value()) and
                    (ui_nop.spinBox_NbRow.value() < self.myTable.tm.rowCount() - row_start) and
                    (ui_nop.spinBox_P.value() <= ui_nop.spinBox_NbRow.value())
            ): break
        p = ui_nop.spinBox_P.value()
        n = ui_nop.spinBox_N.value()  #
        nb_row_to_be_treated = ui_nop.spinBox_NbRow.value()  #

        ordered_list_of_photo_index_to_be_removed = []
        for row_current in range(nb_row_to_be_treated):
            if row_current % p == 0:
                pass
            elif 0 < row_current % p <= n:
                ordered_list_of_photo_index_to_be_removed.append(row_start + row_current)
        ordered_list_of_photo_index_to_be_removed.sort()

        status, \
        message, \
        removed_pictures, \
        could_not_remove_rows = \
            self.active_photos.remove_list_of_photos(ordered_list_of_photo_index_to_be_removed)

        self.myTable.tm.layoutChanged.emit()

        if not status:
            text = "Remove failed for below images: \n\n"
            for image, msg in could_not_remove_rows.items():
                text += str(image.file_name) + " : " + msg + "\n"
            QMessageBox.about(self, PROGRAM_NAME, text)
        # store list of pictures for future roll back

        if len(removed_pictures) != 0:
            self.commit_history.append("remove", removed_pictures)
        for pict in removed_pictures:
            self.discarded_photos.append(pict)

        # update view : summary and graph
        self.update_view_but_table()
        # deselect all selected rows
        self.tableView.clearSelection()

    def set_duplicate_method_from_rigth_click(self, pos):
        """
        Assign a duplicate method to row, selected of all pictures depending on selection
        This function is called only whenever self is a clone
        :param pos:
        :return:
        """
        SCOPE_LIST = {
            "this": "This picture",
            "selected": "selected pictures",
            "all": "All pictures"
        }

        row = self.tableView.rowAt(
            self.tableView.viewport().mapFromGlobal(pos).y()
        )

        set_duplicate_method_qdialog = QtWidgets.QDialog()
        ui_dm = Ui_duplicateMethodChoice()
        ui_dm.setupUi(set_duplicate_method_qdialog)
        # set value for various scope
        ui_dm.scopeComboBox.addItems([value for key, value in SCOPE_LIST.items()])
        # group check bow in an exclusive button group
        exclusive_button_group = QButtonGroup()
        exclusive_button_group.setExclusive(True)
        exclusive_button_group.addButton(ui_dm.gunner_farnerback_checkbox, 1)
        exclusive_button_group.addButton(ui_dm.interpolate_checkbox, 2)
        exclusive_button_group.addButton(ui_dm.lucas_kanade_checkbox, 3)
        exclusive_button_group.addButton(ui_dm.simple_copy_checkbox, 4)

        if set_duplicate_method_qdialog.exec_() == 0: return  # cancel button pushed do nothing and return
        while exclusive_button_group.checkedButton() == -1:
            QMessageBox.about(ui, PROGRAM_NAME, "No duplicate method selected - please select one")

        # retrieve selected duplicate method
        available_method = sorted(DuplicateMethod.get_duplicate_method_supported_list())
        id_ = exclusive_button_group.checkedId()
        selected_duplicate_method = available_method[id_ - 1]

        if str(ui_dm.scopeComboBox.currentText()) == SCOPE_LIST["this"]:
            self.active_photos[row].update_duplicate_method(selected_duplicate_method)

        elif str(ui_dm.scopeComboBox.currentText()) == SCOPE_LIST["selected"]:
            selected_rows = sorted([index.row() for index in self.tableView.selectionModel().selectedRows()])
            selected_pictures = [self.active_photos[row] for row in selected_rows]
            for picture in selected_pictures:
                if isinstance(picture, PhotoCloned):
                    picture.update_duplicate_method(selected_duplicate_method)

        elif str(ui_dm.scopeComboBox.currentText()) == SCOPE_LIST["all"]:
            for picture in self.active_photos:
                if isinstance(picture, PhotoCloned):
                    picture.update_duplicate_method(selected_duplicate_method)

    def set_lucas_kanade_parameter_from_rigth_click(self, pos):

        def store_in_duplicate_method_set(duplicate_method_set):
            duplicate_method_set.FEATURES_PARAMS["maxCorners"] = int(ui_lk.ShiTomasiMaxCorner.text())
            duplicate_method_set.FEATURES_PARAMS["qualityLevel"] = float(ui_lk.ShiTomasiQualityLevel.text()) / 100
            duplicate_method_set.FEATURES_PARAMS["minDistance"] = int(ui_lk.ShiTomasiMinDistance.text())
            duplicate_method_set.FEATURES_PARAMS["blockSize"] = int(ui_lk.ShiTomasiBlockSize.text())
            win_size = int(ui_lk.LucasKanadeWindowSize.text())
            duplicate_method_set.LK_PARAMS["winSize"] = (win_size, win_size)
            duplicate_method_set.LK_PARAMS["maxLevel"] = int(ui_lk.LucasKanadeMaxLevel.text())
            # duplicate_method_set.LK_PARAMS["criteria"] = TODO deals with the triplet

            return

        def get_duplicate_method_set_list(list_of_picture):
            duplicate_method_set_list = []
            for picture in list_of_picture:
                if isinstance(picture, PhotoCloned) and picture.duplicate_method == "lucas_kanade":
                    if picture.clone_set.duplicate_method_set not in duplicate_method_set_list:
                        duplicate_method_set_list.append(picture.clone_set.duplicate_method_set)

            return duplicate_method_set_list

        SCOPE_LIST = {
            "this": "This picture",
            "selected": "selected pictures",
            "all": "All pictures"
        }

        row = self.tableView.rowAt(
            self.tableView.viewport().mapFromGlobal(pos).y()
        )

        duplicate_method_set = self.active_photos[row].clone_set.duplicate_method_set

        set_lucas_kanade_qdialog = QtWidgets.QDialog()
        ui_lk = Ui_lucasKanadeParam()
        ui_lk.setupUi(set_lucas_kanade_qdialog)
        ui_lk.scopeComboBox.addItems([value for key, value in SCOPE_LIST.items()])

        # retrieve current values in CloneSet
        ui_lk.ShiTomasiMaxCorner.setText(str(duplicate_method_set.FEATURES_PARAMS["maxCorners"]))
        ui_lk.ShiTomasiQualityLevel.setText(
            str(int(duplicate_method_set.FEATURES_PARAMS["qualityLevel"] * 100)))  # actual is float 0 to 1
        ui_lk.ShiTomasiMinDistance.setText(str(duplicate_method_set.FEATURES_PARAMS["minDistance"]))
        ui_lk.ShiTomasiBlockSize.setText(str(duplicate_method_set.FEATURES_PARAMS["blockSize"]))
        ui_lk.LucasKanadeWindowSize.setText(str(duplicate_method_set.LK_PARAMS["winSize"][0]))  # (15,15) format takes first only
        ui_lk.LucasKanadeMaxLevel.setText(str(duplicate_method_set.LK_PARAMS["maxLevel"]))
        ui_lk.LucasKanadeCriteria.setText(str(duplicate_method_set.LK_PARAMS["criteria"]))

        # define input control Validator
        self.stmc = QIntValidator(0, 200000)
        ui_lk.ShiTomasiMaxCorner.setValidator(self.stmc)
        self.stql = QIntValidator(0, 100)
        ui_lk.ShiTomasiQualityLevel.setValidator(self.stql)
        self.stmd = QIntValidator(0, 100)
        ui_lk.ShiTomasiMinDistance.setValidator(self.stmd)
        self.stbs = QIntValidator(0, 100)
        ui_lk.ShiTomasiBlockSize.setValidator(self.stbs)
        self.lkws = QIntValidator(0, 100)
        ui_lk.LucasKanadeWindowSize.setValidator(self.lkws)
        self.lkml = QIntValidator(0, 10)
        ui_lk.LucasKanadeMaxLevel.setValidator(self.lkml)
        # self.lkc = QIntValidator(0,100)
        # ui_lk.LucasKanadeCriteria.setValidator(self.only_int)

        # while True:
        if set_lucas_kanade_qdialog.exec_() == 0: return  # cancel button pushed do nothing and return

        if str(ui_lk.scopeComboBox.currentText()) == SCOPE_LIST["this"]:

            store_in_duplicate_method_set(duplicate_method_set)

        elif str(ui_lk.scopeComboBox.currentText()) == SCOPE_LIST["selected"]:

            selected_rows = sorted([index.row() for index in self.tableView.selectionModel().selectedRows()])

            selected_pictures = [self.active_photos[row] for row in selected_rows]

            for duplicate_method_set in get_duplicate_method_set_list(selected_pictures):
                store_in_duplicate_method_set(duplicate_method_set)

        elif str(ui_lk.scopeComboBox.currentText()) == SCOPE_LIST["all"]:

            try:
                for duplicate_method_set in get_duplicate_method_set_list(self.active_photos):
                    store_in_duplicate_method_set(duplicate_method_set)
            except Exception as e:
                print(exception_to_string(e))

    def handle_duplicate_method_set_to_simple_copy_menu(self):
        self.uncheck_duplicate_method_actions_menu_items()
        self.actionsimple_copy.setChecked(True)
        self.duplicate_method = "simple_copy"
        logger.info("Duplicate method set to simple_copy mode")

    def handle_duplicate_method_set_to_interpolate_menu(self):
        self.uncheck_duplicate_method_actions_menu_items()
        self.actioninterpolate.setChecked(True)
        self.duplicate_method = "interpolate"
        logger.info("Duplicate method set to interpolate mode")

    def handle_duplicate_method_set_to_lucas_kanade_menu(self):
        self.uncheck_duplicate_method_actions_menu_items()
        self.actionlucas_kanade.setChecked(True)
        self.duplicate_method = "lucas_kanade"
        logger.info("Duplicate method set to lucas-kanade mode")

    def handle_duplicate_method_set_to_gunner_farnerback_menu(self):
        self.uncheck_duplicate_method_actions_menu_items()
        self.actiongunner_farnerback.setChecked(True)
        self.duplicate_method = "gunner_farnerback"
        logger.info("Duplicate method set to Gunner Farnerback mode")

    def uncheck_duplicate_method_actions_menu_items(self):
        self.actionsimple_copy.setChecked(False)
        self.actioninterpolate.setChecked(False)
        self.actionlucas_kanade.setChecked(False)
        self.actiongunner_farnerback.setChecked(False)

    def handle_clear_selection_button(self):
        self.tableView.clearSelection()

    def handle_pickle_button(self):

        # get filename

        label = "scenario Files (*" + SCENARIO_FILE_SUFFIX + ")"
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        options |= QFileDialog.Detail
        file_name, _ = QFileDialog.getSaveFileName(self, "Scenario file name", os.getcwd(),
                                                   label, options=options)
        if file_name:
            file_name_raw = str(file_name)
            print(file_name_raw)
            # add suffix if none provided
            if len(file_name.split(".")) == 1:  # no suffix
                file_name += SCENARIO_FILE_SUFFIX
            # check suffix and force it to SCENARIO_FILE_SUFFIX
            elif file_name.split(".")[-1] != SCENARIO_FILE_SUFFIX.split(".")[-1]:
                file_name = "".join(file_name.split(".")[:-1]) + SCENARIO_FILE_SUFFIX
                logger.info("%s with wrong suffix - rewritten as %s", file_name_raw, file_name )
        else:
            return  # no file name or cancel - just quit

        if len(self.active_photos) != 0:
            to_be_pickled = [self.active_photos[i] for i in range(len(self.active_photos))]
        else:
            QMessageBox.about(ui, PROGRAM_NAME, "Empty scenario - no photo loaded - saving is not possible :-)")
            return  # nothing to pickle - just quit

        logger.info(" START PICKLING %s PHOTOS IN FILE: %s", str(len(to_be_pickled)), str(file_name))
        # set recursion depth else we get   <class 'RecursionError'> maximum recursion depth exceeded
        sys.setrecursionlimit(
            max(
                len(self.active_photos) * 100,
                sys.getrecursionlimit()
            )
        )
        with open(file_name, 'wb') as f:
            try:
                pickle.dump(to_be_pickled, f)
            except RecursionError as e:
                print(exception_to_string(e))
                logger.error("recursion limit (%s) reached while pickling", str(sys.getrecursionlimit()))
            except Exception as e:
                print(exception_to_string(e))

        logger.info("PICKLING COMPLETED")

    def handle_unpickle_button(self):

        # get filename
        label = "scenario Files (*" + SCENARIO_FILE_SUFFIX + ")"
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog  # TODO stupid copy paste should i keep options ?
        options |= QFileDialog.Detail

        file_name, _ = QFileDialog.getOpenFileName(self, "Scenario file name", os.getcwd(),
                                                   label, options=options)

        if not file_name:
            return  # no file name or cancel - just quit

        # TODO add a request for confirmation that current data will be lost
        self.active_photos.reset()

        logger.info(" START UNPICKLING PHOTOS FROM %s", str(file_name))
        with open(file_name, 'rb') as f:
            try:
                photo_list = pickle.load(f)
            except Exception as e:
                print(exception_to_string(e))

        self.myTable.tm.layoutAboutToBeChanged.emit()
        for picture in photo_list:
            self.active_photos.add(picture)
        self.myTable.tm.layoutChanged.emit()

        self.my_graph.compute_and_display_figure(self.active_photos)
        self.my_slider.display_by_row(0)
        self.my_slider.set_cursor(0)
        self.statusbar_message_qlabel.setText("files from scenario:" + str(file_name))

        logger.info("UNPICKLING OF %s PHOTO COMPLETED", str(len(photo_list)))

    def handle_exit_menu(self):
        sys.exit(app.exec_())  # TODO wort out why it exits with return code -1 whereas clik on window bar is 0


def handle_uncaugth_exception(*exc_info):
    """
    This function will be subsituted to sys.except_hook standard function that is raised when ecxeptions are raised and
    not caugth by some try: except: block
    :param exc_info: (exc_type, exc_value, exc_traceback)
    :return: stop program with return code 1
    """
    stack = traceback.extract_stack()[:-3] + traceback.extract_tb(exc_info[1].__traceback__)  # add limit=??
    pretty = traceback.format_list(stack)
    text = ''.join(pretty) + '\n  {} {}'.format(exc_info[1].__class__, exc_info[1])
    # text = "".join(traceback.format_exception(*exc_info))
    logger.error("Unhandled exception: %s", text)
    sys.exit(1)


def exception_to_string(excp):
    stack = traceback.extract_stack()[:-3] + traceback.extract_tb(excp.__traceback__)  # add limit=??
    pretty = traceback.format_list(stack)
    return ''.join(pretty) + '\n  {} {}'.format(excp.__class__, excp)


def print_exec_time(tps_avant, label_to_be_printed, logger=logging.getLogger("my_logger")):
    tps_apres = time.time()
    tps_execution = tps_apres - tps_avant
    logger.info("La fonction %s a mis %s pour s'exécuter", str(label_to_be_printed), str(tps_execution))
    return


if __name__ == "__main__":
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

    sys.excepthook = handle_uncaugth_exception  # reassign so that log is fed with problem

    VERSION = "V0.9.1"
    PROGRAM_NAME = "PHOTO INTERVAL MANAGER"

    TAG_DICTIONNARY = OrderedDict({
        "File Name": ["SourceFile"],  # /!\ TAG Hardcoded in PhotoCloned.get_tag_value method
        "F number": ["EXIF:FNumber"],
        "Exposure Time": ["EXIF:ExposureTime"],
        "ISO": ["EXIF:ISO", "MakerNotes:ISO"],  # when using ISO below 100 Nikon stores ISO value in MakerNotes
        "Focal Length": ["EXIF:FocalLength"],
        # "Thumbnail" : ["EXIF:ThumbnailTIFF"],
        # "OtherImage" : ["EXIF:OtherImage"],
        # "JPEG Image" : ["EXIF:JpgFromRaw"],
        "Create Date": ["EXIF:CreateDate"],  # /!\ TAG Hardcoded in PhotoCloned.get_tag_value method
    })

    SIZE_OF_FILE_CHUNKS = 300  # size of file chunks to be loaded for sub processing
    PhotoOrderedCollectionByCapturetime.set_size_of_file_chunks(SIZE_OF_FILE_CHUNKS)

    NB_BACKGROUND_PICTURE_LOADING_THREADS = 3

    # global VALID_PHOTO_FILE_SUFFIXES
    # VALID_PHOTO_FILE_SUFFIXES = [".jpg",".NEF",".JPG",".CR2","DNG","RAF","SR2"]
    # VALID_PHOTO_FILE_SUFFIXES = [".NEF"]
    VALID_PHOTO_FILE_SUFFIXES = [".NEF", ".JPG", ".jpg"]

    VALID_VIDEO_FILE_SUFFIXES = [".mp4", ".avi"]   # has to be handle properly by opencv3

    SCENARIO_FILE_SUFFIX = ".pkl"   # stands for pickle

    logger.info('=========  ' + PROGRAM_NAME + ' ' + VERSION + ' STARTED ===========')
    logger.info("TAGS TREATED =\n" + str(TAG_DICTIONNARY))



    # TODO <<< DO NOT CHANGE ORDER OF TAG AS MyTableModel is relying on their position >>> TO BE FIXED LATER
    # TODO    IN PARTICULAR CreateDate TO BE IN LAST POSITION
    # TODO find a way to not depend on CreateDate position in the list

    # TODO implement a stop event in Thread https://stackoverflow.com/questions/323972/is-there-any-way-to-kill-a-thread-in-python

    # TODO put a lock on ui.active_photos that prevent adding photos until backgroung task is completed OR that
    # TODO impleemnts an approach whereby the background thread is stopped and restarted with fresh data


    try:
        app = QtWidgets.QApplication(sys.argv)
        ui = ModelToViewController()
        ui.show()
        sys.exit(app.exec_())
    except Exception:
        logger.exception("Exception caugth:")
