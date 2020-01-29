'''
How to create a Taylor diagram with modified axes and data point colors
that show co-located points.
A ninth example of how to create a Taylor diagram given one set of
reference observations and multiple model predictions for the quantity.
This example is a variation on the fifth example (taylor5) where now a
legend is added, axes titles are suppressed, and four points are co-located
(i.e. overly each other). Symbols with transparent faces are used so the
co-located points can be seen. The list of points are checked for those
that agree within 1% of each other and reported to the screen.
All functions in the Skill Metrics library are designed to only work with
one-dimensional arrays, e.g. time series of observations at a selected
location. The one-dimensional data are read in as dictionaries via a
pickle file: ref['data'], pred1['data'], pred2['data'],
and pred3['data']. The plot is written to a file in Portable Network
Graphics (PNG) format.
The reference data used in this example are cell concentrations of a
phytoplankton collected from cruise surveys at selected locations and
time. The model predictions are from three different simulations that
have been space-time interpolated to the location and time of the sample
collection. Details on the contents of the dictionary (once loaded) can
be obtained by simply executing the following two statements
>> key_to_value_lengths = {k:len(v) for k, v in ref.items()}
>> print key_to_value_lengths
{'units': 6, 'longitude': 57, 'jday': 57, 'date': 57, 'depth': 57,
'station': 57, 'time': 57, 'latitude': 57, 'data': 57}
Author: Peter A. Rochford
        Symplectic, LLC
        www.thesymplectic.com
Created on Apr 22, 2017
@author: prochford@thesymplectic.com
'''

import matplotlib.pyplot as plt
import numpy as np
import pickle
import skill_metrics as sm
from sys import version_info


def load_obj(name):
    # Load object from file in pickle format
    if version_info[0] == 2:
        suffix = 'pkl'
    else:
        suffix = 'pkl'

    with open(name + '.' + suffix, 'rb') as f:
        return pickle.load(f)  # Python2 succeeds


class Container(object):

    def __init__(self, pred1, pred2, pred3, ref):
        self.pred1 = pred1
        self.pred2 = pred2
        self.pred3 = pred3
        self.ref = ref


if __name__ == '__main__':
    # Close any previously open graphics windows
    # ToDo: fails to work within Eclipse
    plt.close('all')

    # Read data from pickle file
    data = load_obj('/home/irene/PycharmProjects/04_Risk_Model/data/taylor_data')

    # Calculate statistics for Taylor diagram
    # The first array element corresponds to the reference series
    # for the while the second is that for the predicted series.
    taylor_stats1 = sm.taylor_statistics(data.pred1, data.ref, 'data')
    taylor_stats2 = sm.taylor_statistics(data.pred2, data.ref, 'data')
    taylor_stats3 = sm.taylor_statistics(data.pred3, data.ref, 'data')

    # Store statistics in arrays, making the fourth element a repeat of
    # the first.
    sdev = np.array([taylor_stats1['sdev'][0], taylor_stats1['sdev'][1],
                     taylor_stats2['sdev'][1], taylor_stats3['sdev'][1],
                     taylor_stats1['sdev'][1], 0.991 * taylor_stats3['sdev'][1]])
    crmsd = np.array([taylor_stats1['crmsd'][0], taylor_stats1['crmsd'][1],
                      taylor_stats2['crmsd'][1], taylor_stats3['crmsd'][1],
                      taylor_stats1['crmsd'][1], taylor_stats3['crmsd'][1]])
    ccoef = np.array([taylor_stats1['ccoef'][0], taylor_stats1['ccoef'][1],
                      taylor_stats2['ccoef'][1], taylor_stats3['ccoef'][1],
                      taylor_stats1['ccoef'][1], taylor_stats3['ccoef'][1]])

    # Specify labels for points in a cell array (M1 for model prediction 1,
    # etc.). Note that a label needs to be specified for the reference even
    # though it is not used.
    label = ['Non-Dimensional Observation', 'M1', 'M2', 'M3', 'M4', 'M5']

    # Check for duplicate statistics
    duplicateStats = sm.check_duplicate_stats(sdev[1:], crmsd[1:])

    # Report duplicate statistics, if any
    sm.report_duplicate_stats(duplicateStats)

    '''
    Produce the Taylor diagram
    Label the points and change the axis options for SDEV, CRMSD, and CCOEF.
    Increase the upper limit for the SDEV axis and rotate the CRMSD contour
    labels (counter-clockwise from x-axis). Exchange color and line style
    choices for SDEV, CRMSD, and CCOEFF variables to show effect. Increase
    the line width of all lines.
    For an exhaustive list of options to customize your diagram, 
    please call the function at a Python command line:
    >> taylor_diagram
    '''
    sm.taylor_diagram(sdev, crmsd, ccoef, markerLabel=label,
                      markerLabelColor='r',
                      markerColor='r', markerLegend='on',
                      tickRMS=range(0, 60, 10),
                      colRMS='m', styleRMS=':', widthRMS=2.0,
                      titleRMS='on', tickSTD=range(0, 80, 20),
                      axismax=60.0, colSTD='b', styleSTD='-.',
                      widthSTD=1.0, titleSTD='on',
                      colCOR='k', styleCOR='--', widthCOR=1.0,
                      titleCOR='on', markerSize=10, alpha=0.0)

    # Write plot to file
    plt.savefig('taylor9.png')

    # Show plot
plt.show()