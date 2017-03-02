#!/usr/bin/python
"""
Functions for detection of repeating and near-repeating events in seismic
data with little a-priori constraints. Based on the methods of Brown et al.
(2008): An autocorrelation method to detect low frequency earthquakes within
tremor.

:copyright:
    EQcorrscan developers.

:license:
    GNU Lesser General Public License, Version 3
    (https://www.gnu.org/copyleft/lesser.html)
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import itertools

from obspy import Stream
from multiprocessing import Pool

from eqcorrscan.core import match_filter


def _segment_data(stream, length, lag, max_lag=200, cores=1):
    """
    Function to cut the continuous data into templates of set length and lag.

    :type stream: :class:`obspy.core.stream.Stream`
    :param stream: Stream of continuous data to be cut from.
    :type length: float
    :param length: Length for cut channels in seconds.
    :type lag: float
    :param lag: Lag step to apply across channels.
    :type max_lag: float
    :param max_lag: Largest lag time-step to use in seconds.
    :type cores: int
    :param cores: number of processed to parallel over.

    :return: List of :class:`obspy.core.stream.Stream`
    """
    zero_time = stream.sort(['starttime'])[0].stats.starttime
    n_segments = int(((stream.sort(['starttime'])[-1].stats.endtime -
                       zero_time) - length) / lag)
    # pool = Pool(processes=cores)
    # results = [pool.apply_async(_pool_segment, (stream,),
    #                             {'lags': t, 'stations': stations,
    #                              'zero_time': zero_time, 'length': length,
    #                              'lag': lag, 'max_lag': max_lag})
    #            for t in itertools.product(*iterables)]
    # pool.close()
    templates = []
    for t in range(n_segments):
        if abs((t * lag)) > max_lag:
            continue
        # print('Start cut %s' % str(zero_time + (lag * t)))
        # print('End cut %s' % str(zero_time + length + (lag * t)))
        templates.append(stream.slice(
            starttime=zero_time + (lag * t),
            endtime=zero_time + length + (lag * t)).copy())
    # templates = [p.get() for p in results]
    # pool.join()
    return templates


def _pool_segment(stream, lags, stations, zero_time, length, lag, max_lag):
    """
    Internal function for parallel processing

    :param stream:
    :param lags:
    :param stations:
    :param zero_time:
    :param length:
    :param lag:
    :return:
    """
    for lag in lags:
        if abs(lag - min(lags)) > max_lag:
            return
    template = Stream()
    for lagmult, station in zip(lags, stations):
        template += stream.select(station=station).slice(
            starttime=zero_time + (lag * lagmult),
            endtime=zero_time + length + (lag * lagmult)).copy()
    return template


def autocorr(stream, length, lag, threshold, threshold_type, plotvar,
             plotdir='.', cores=1, debug=0, plot_format='png',
             output_cat=False, output_event=False, extract_detections=False,
             arg_check=True):
    """
    Main autocorrelation method function.

    :type stream: :class:`obspy.core.stream.Stream`
    :param stream: Stream of continuous data to search within.
    :type length: float
    :param length: Length for cut channels in seconds.
    :type lag: float
    :param lag: Lag step to apply across channels.
    :type threshold: float
    :param threshold: A threshold value set based on the threshold_type
    :type threshold_type: str
    :param threshold_type: The type of threshold to be used, can be MAD, \
        absolute or av_chan_corr.  See Note on thresholding below.
    :type plotvar: bool
    :param plotvar: Turn plotting on or off
    :type plotdir: str
    :param plotdir: Path to plotting folder, plots will be output here, \
        defaults to run location.
    :type cores: int
    :param cores: Number of cores to use
    :type debug: int
    :param debug: Debug output level, the bigger the number, the more the \
        output.
    :type plot_format: str
    :param plot_format: Specify format of output plots if saved
    :type output_cat: bool
    :param output_cat: Specifies if matched_filter will output an \
        obspy.Catalog class containing events for each detection. Default \
        is False, in which case matched_filter will output a list of \
        detection classes, as normal.
    :type extract_detections: bool
    :param extract_detections: Specifies whether or not to return a list of \
        streams, one stream per detection.
    :type arg_check: bool
    :param arg_check: Check arguments, defaults to True, but if running in \
        bulk, and you are certain of your arguments, then set to False.\n

    .. rubric::
        If neither `output_cat` or `extract_detections` are set to `True`,
        then only the list of :class:`eqcorrscan.core.match_filter.DETECTION`'s
        will be output:
    :return: :class:`eqcorrscan.core.match_filter.DETECTION`'s detections for
        each detection made.
    :rtype: list
    .. rubric::
        If `output_cat` is set to `True`, then the
        :class:`obspy.core.event.Catalog` will also be output:
    :return: Catalog containing events for each detection, see above.
    :rtype: :class:`obspy.core.event.Catalog`
    .. rubric::
        If `extract_detections` is set to `True` then the list of
        :class:`obspy.core.stream.Stream`'s will also be output.
    :return:
        list of :class:`obspy.core.stream.Stream`'s for each detection, see
        above.
    :rtype: list

    .. warning::
        Plotting within the match-filter routine uses the Agg backend
        with interactive plotting turned off.  This is because the function
        is designed to work in bulk.  If you wish to turn interactive
        plotting on you must import matplotlib in your script first, when you
        them import match_filter you will get the warning that this call to
        matplotlib has no effect, which will mean that match_filter has not
        changed the plotting behaviour.

    .. note::
        **Data overlap:**

        Internally this routine shifts and trims the data according to the
        offsets in the template (e.g. if trace 2 starts 2 seconds after trace 1
        in the template then the continuous data will be shifted by 2 seconds
        to align peak correlations prior to summing).  Because of this,
        detections at the start and end of continuous data streams
        **may be missed**.  The maximum time-period that might be missing
        detections is the maximum offset in the template.

        To work around this, if you are conducting matched-filter detections
        through long-duration continuous data, we suggest using some overlap
        (a few seconds, on the order of the maximum offset in the templates)
        in the continous data.  You will then need to post-process the
        detections (which should be done anyway to remove duplicates).

    .. note::
        **Thresholding:**

        **MAD** threshold is calculated as the:

        .. math::

            threshold {\\times} (median(abs(cccsum)))

        where :math:`cccsum` is the cross-correlation sum for a given template.

        **absolute** threshold is a true absolute threshold based on the
        cccsum value.

        **av_chan_corr** is based on the mean values of single-channel
        cross-correlations assuming all data are present as required for the
        template, e.g:

        .. math::

            av\_chan\_corr\_thresh=threshold \\times (cccsum / len(template))

        where :math:`template` is a single template from the input and the
        length is the number of channels within this template.
    """
    templates = _segment_data(stream=stream, length=length, lag=lag,
                              cores=cores)
    template_names = ['autocorr_' + str(i) for i in range(len(templates))]
    candidates = match_filter.match_filter(
        template_names=template_names, template_list=templates, st=stream,
        threshold=threshold, threshold_type=threshold_type, trig_int=lag,
        plotvar=plotvar,  plotdir='.', cores=1, debug=0, plot_format='png',
        output_cat=False, output_event=False,
        extract_detections=False, arg_check=True)
    return candidates
