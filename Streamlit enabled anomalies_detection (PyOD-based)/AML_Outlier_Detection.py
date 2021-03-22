# -*- coding: utf-8 -*-
"""AML Test - Anomalies Detection.ipynb

# A generic facilitator for exploration of potential sample outliers in *.csv and EXCEL (*.xlx, *.xlsx, *.xlsm) data files.
### Álvaro Montesino, March 17th, 2021

This code is an adaptation of the PyOD example source code named "[compare_all_models.py](https://github.com/yzhao062/pyod/blob/master/examples/compare_all_models.py)", available in the [PyOD project repository](https://pyod.readthedocs.io/en/latest/). 

There is Input file 

**[PyOD](https://github.com/yzhao062/pyod)** is a comprehensive **Python toolkit** to **identify outlying objects** in 
multivariate data with both unsupervised and supervised approaches.
The model covered in this example includes:

  1. Linear Models for Outlier Detection:
     1. **PCA: Principal Component Analysis** use the sum of
       weighted projected distances to the eigenvector hyperplane 
       as the outlier outlier scores)
     2. **MCD: Minimum Covariance Determinant** (use the mahalanobis distances 
       as the outlier scores)
     3. **OCSVM: One-Class Support Vector Machines**
     
  2. Proximity-Based Outlier Detection Models:
     1. **LOF: Local Outlier Factor**
     2. **CBLOF: Clustering-Based Local Outlier Factor**
     3. **kNN: k Nearest Neighbors** (use the distance to the kth nearest 
     neighbor as the outlier score)
     4. **Median kNN** Outlier Detection (use the median distance to k nearest 
     neighbors as the outlier score)
     5. **HBOS: Histogram-based Outlier Score**
     
  3. Probabilistic Models for Outlier Detection:
     1. **ABOD: Angle-Based Outlier Detection**
  
  4. Outlier Ensembles and Combination Frameworks
     1. **Isolation Forest**
     2. **Feature Bagging**
     3. **LSCP**
"""


###############################################################################
#
#   Modules section
#
###############################################################################

#from __future__ import division
#from __future__ import print_function

import os
import sys
from sys import float_info
from time import time
import time
import math

# New to both ways of manipulating file paths, os.path and libpath libs, and will develop this, its libpath equivalent
from pathlib import Path
import pathlib
import zipfile

# temporary solution for relative imports in case pyod is not installed
# if pyod is installed, no need to use the following line
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname("__file__"), '..')))

import numpy as np
from numpy import percentile

import pandas as pd
from pandas.api.types import is_datetime64_any_dtype as is_datetime

import matplotlib.pyplot as plt
import matplotlib.font_manager

# Import all models
import pyod as p

import pyod
from pyod.models.abod import ABOD
from pyod.models.cblof import CBLOF
from pyod.models.feature_bagging import FeatureBagging
from pyod.models.hbos import HBOS
from pyod.models.iforest import IForest
from pyod.models.knn import KNN
from pyod.models.lof import LOF
from pyod.models.mcd import MCD
from pyod.models.ocsvm import OCSVM
from pyod.models.pca import PCA
from pyod.models.lscp import LSCP

# We need to provide the means for the user to be able to specify interactively...
#   - which file should be processed.
#   - which features, out of just those that may be fit for identification of 
#     outliers, should be inspected...
#   - the threshold to be used for considering the predicted characteristic of 
#     being an outlier like a truly outlier
# All interactive exchange will be carried out through a Streamlit web page.
import streamlit as st


###############################################################################
#
#   Functions
#
###############################################################################

# Reusing code for reading multiple files and loading contents onto 
# pandas dataframe
def process_data_files( my_path, my_filename, my_func, my_exclusion_list, *args, **kwargs ):
    # The function walks recursively through all relevant files somewhere under
    # path. The function passed as parameter will then be applied
    # upon each of such dataframes and the pandas dataframes returned as output.
    # The functions assumes that the specified input file is one that matches 
    # the general outlay of a datafile that is liable to be uploaded onto a
    # pandas dataframe.
    #
    # INPUTS:
    #   - my_path (Path object): name of root directory where to start 
    #       looking for the target file.
    #   - my_filename (string): regular expression that, when matched, 
    #       identifies the file as a target.
    #   - my_func (function): function to be applied to each of the dataframes
    #   - my_exclusion_list (list): a list of pathnames that ought to be 
    #       explicitly excluded from processing. Certain files with potentially
    #       suitable extensions, turn out to include a mix of types for some of
    #       the columns. Other files contain header rows which interfere with
    #       a simplified approach to recursive processing.
    # OPTIONAL PARAMETERS
    #   - confirm_exists= True (boolean): Processing of most files require that
    #       certain checks be performed. Zipfiles, however, presents challenges
    #       that require that same checks not be performed.
    # RETURNS:
    #   A liat of pandas dataframes, one for each of the target datafiles that
    #   were readable by pandas.
    #

    # Prior to checking whether the file exists, we have to confirm the approach to be used for confirmation of existence of path provided as parameter. 
    confirm_exists = True
    print( "Parameters - path({}: '{}' filename({}):'{}'".format( type( my_path ), my_path, type( my_filename ), my_filename ) )
    print( "Variable parameters ('%d'):'%s'" % ( len( kwargs ), kwargs ) )
    confirm_exists = kwargs.get( 'confirm_exists', None )

    # Invoking the exists method upon a path of contents of a zip file is not contemplated
    if ( confirm_exists and not my_path.exists() ):
        print( "Path does not exist:'%s'" % my_path )
        raise TypeError( "Path does not point to anything: '%s'" % my_path )
    
    list_processed_files = []
    list_dfs = []

    for a_path in my_path.rglob( str( my_filename ) ):
        print( "Path:", os.fspath( a_path ) )
        if ( a_path.name in my_exclusion_list ):
              print( "File is to be explictly excluded from processing" )
              continue
        elif ( a_path.name != my_filename ):
              print( "Current file '{}' is not the target file".format( a_path ) ) 
        
        if a_path.suffix in [ '.csv' ]:
            print( "CSV file" )  
            p = pd.read_csv( os.fspath( a_path ) )
            list_dfs.append( p )
            list_processed_files.append( a_path )
        elif a_path.suffix in [ '.xlsx', '.xlsm', '.xls' ]:
            print( "EXCEL file" )     
            p = pd.read_excel( os.fspath( a_path ) )
            list_dfs.append( p )
            list_processed_files.append( a_path )
        # elif a_path.suffix in [ '.zip' ]:
        #      print( "ZIP file:", a_path )
        #      if zipfile.is_zipfile( a_path ):
        #          zf = zipfile.ZipFile( a_path, 'r' )
        #          for file in zf.namelist(): 
        #              print( "\tZip contents:'%s'" % file )
        #              if file.suffix in [ '.csv' ]:
        #                  print( "Zipped CSV file")
        #                  p = pd.read_csv( os.fspath( file.Path ) )
        #                  list_dfs.append( p )   
        #                  list_processed_files.append( file.Path )           
        #              elif file.suffix in [ '.xlsx', 'xlsm', 'xls' ]:
        #                  p = pd.read_excel( os.fspath( file.Path ) )
        #                  list_dfs.append( p )   
        #                  list_processed_files.append( file.Path )                                          
        #              # Handling contents of zip files recursively requires rework which I will not prioritize at this time.
        #              #list_processed_files.append( process_data_files_recursively( file, my_func, my_exclusion_list ), confirm_exists= False )
        #              else:
        #                  # print( "Zip file not among formats of interest" )
        #                  continue   
        else:
            # print( "File not among formats of interest" )
            continue   
        
    print( "Processed files {}: {}".format( len( list_processed_files ), list_processed_files ) )
    #print( "Dataframes {} {}:".format( len( list_dfs ), map( list_dfs, print ) ) )
    print( "Dataframes {} {}:".format( len( list_dfs ), list_dfs ) )
    output_list = list( map( my_func, list_dfs, list_processed_files ) )
    # output_list    
    print() 
 
    print( "Output list", output_list )
        
    return list_dfs


###############################################################################
#
#   Main section
#
###############################################################################

# By default, process the Lockdown_Tracking data.
DATA_ROOT_DIR = Path( "C:/Users/alvar/OneDrive/Data" )
PATH_TO_DATA_FILE = DATA_ROOT_DIR / "Short Term Datasets" / "LockDown_Tracking.csv"
uploaded_file = PATH_TO_DATA_FILE

# Start drawing the header for the central Stramlit frame
st.sidebar.title( "Anomaly detection for file" )
st.sidebar.write( "'", uploaded_file, "'" )

with st.sidebar.beta_expander( 'Data file uploader' ):
    # Some of the latest versions of the file_uploader widget point to the fact that latest versions don't seem to provide
    # the path to the target file. It only provides the actual filename. https://share.streamlit.io/streamlit/release-demos/master/0.68%2Fstreamlit_app.py 
    read_filename = st.file_uploader( '', key= [ '.csv', '.xls', '.xlsx', '.xlsm' ] )
    if read_filename:
        uploaded_file = Path( read_filename.name )
    else:
        pass

# Certain files to be excluded from search because of the impact that their layout has on pandas'
# ability to read table contents
my_exclusion_list = [ 
      'EDA_reports'
]
#df = pd.read_csv( os.fspath( PATH_TO_DATA_FILE ) )
# The file_uploader widget 
dataframe_list = process_data_files( DATA_ROOT_DIR, uploaded_file.name , print, my_exclusion_list )

# For the time being we will assume the simplest case that one a single file matches and that, hence, 
# a single dataframe is returned. 
# st.write( "Length of dataframe_list(", len( dataframe_list ), ")" )
df = dataframe_list[ 0 ]

with st.beta_expander( "Structure of data"):
    st.write( "Feature types: ", df.dtypes )
    # st.write( "Feature types: ", df.info() )
    st.write( "Data prior to any modification:" )
    st.dataframe ( df )
    st.write( "Exploring which _object_ columns can actually be converted into columns of type _datetime_..." )
    for column in df.columns:
        if df[ column ].dtype == 'object':
            try:
                df[ column ] = pd.to_datetime( df[ column ] )
            except ValueError:
                pass
    st.write( "Dataframe after adjustments:" )
    st.dataframe ( df )
    st.write( "Number of nulls", df.isnull().sum() )
    st.write( "Valid cell count: ", df.count() )
    st.write( "Dataframe statistics:", df.describe() )
    
clusters_separation = [0]

# features_of_interest = [ 'Original Res Forecast before Lockdown', 'Actual Residual Demand' ]
features_of_interest = [ '', '' ]

df = df.select_dtypes( include= np.number )
column_headings = df.columns

features_of_interest[ 0 ] = st.sidebar.selectbox(
    'Choose among numeric feature for X axis:',
    column_headings
)
st.sidebar.write( features_of_interest[ 0 ], "': ", df[ features_of_interest[ 0 ] ].dtype, " Row count: ", df[ features_of_interest[ 0 ] ].count() )

features_of_interest[ 1 ] = st.sidebar.selectbox(
    'Choose among numeric feature for Y axis:',
    np.setdiff1d( column_headings, features_of_interest[ 0 ] )
)
st.sidebar.write( features_of_interest[ 1 ], "': ", df[ features_of_interest[ 1 ] ].dtype, " Row count: ", df[ features_of_interest[ 1 ] ].count() )

# Once features of interest are identified, we need to remove rows with NaNs. 
relevantFeaturesWithNaNs = [ feature for feature in df.columns[ df.isna().any() ].tolist() if feature in features_of_interest ]
print( "Columns with NaNs among the features of interest:'%s'" % relevantFeaturesWithNaNs  )
df = df.dropna( subset= relevantFeaturesWithNaNs )
st.sidebar.write( "Correlation between chose: ", np.corrcoef( df[ features_of_interest[ 0 ] ], df[ features_of_interest[ 1 ] ], rowvar= False ) )

# Define the number of inliers and outliers
n_samples = 200
n_samples = min( df[ features_of_interest[ 0 ] ].count(), df[ features_of_interest[ 1 ] ].count() )

outliers_fraction = st.sidebar.slider( 
    'Fraction of outliers (%)',
    min_value = 1,
    max_value = 50
) / 100
st.write( "Orange area marks the boundary for ", outliers_fraction*100, "% of outliers" )
# Most likely will have to undo this change and instead, revert back to pandas Series but, for the time being 
# I'll just proceed sorting on the first feature, in ascending order.
df.sort_values( by=[ features_of_interest[ 0 ] ], inplace=True )
print( df.count( ) )
# df

# Compare given detectors under given settings
# We initialize two matrices for the range of potential values the data
MAX_X = df[ features_of_interest[ 0 ] ].max()*1.05
MIN_X = df[ features_of_interest[ 0 ] ].min()*0.95

MAX_Y = df[ features_of_interest[ 1 ] ].max()*1.05
MIN_Y = df[ features_of_interest[ 1 ] ].min()*0.95

major_units = 100


n_inliers = math.ceil( (1. - outliers_fraction) * n_samples )
n_outliers = int( outliers_fraction * n_samples )

# It seems as if we are using a vector to define which ranges are considered inliers vs. which other range is considered an outlier
# By default, no outliers.
ground_truth = np.zeros(n_samples, dtype=int)
# ...but anything found in the final (topmost) range will from now on be considered an outlier.
ground_truth[-n_outliers:] = 1

# initialize a set of detectors for LSCP
detector_list = [LOF(n_neighbors=5), LOF(n_neighbors=10), LOF(n_neighbors=15),
                 LOF(n_neighbors=20), LOF(n_neighbors=25), LOF(n_neighbors=30),
                 LOF(n_neighbors=35), LOF(n_neighbors=40), LOF(n_neighbors=45),
                 LOF(n_neighbors=50)]

# We create a mesh, which we will then feed as a pair of coordinates to the already trained model, in order to 
# to collect the resulting prediction from the model, which we will store in matrix Z 
xx, yy = np.meshgrid( np.linspace( MIN_X, MAX_X, major_units ), np.linspace( MIN_Y, MAX_Y, major_units ) )
# with st.beta_expander( "Axis data" ):
#    st.write( "X axis - '", features_of_interest[ 0 ], "' \[", MIN_X, ",", MAX_X, "\]" )
#    # st.dataframe( pd.DataFrame( xx ) )
#    st.write( "Y axis - '", features_of_interest[ 1 ], "' \[", MIN_Y, ",", MAX_Y, "\]" )
#    # st.dataframe( pd.DataFrame( yy ) )
#    st.write( "\# samples = ", n_samples, "\# Inlier samples ( ", n_inliers, " ) + \#Outlier samples ( ", n_outliers, " ) = ", n_inliers+n_outliers )
#
#    # Ground truth - Outliers are 1's. Inliers are 0's
#    st.write( "Ground truth shape is ", ground_truth.shape )
#    st.write( "Showing is transposed: ", np.transpose( ground_truth ) )

random_state = np.random.RandomState(42)
# Define nine outlier detection tools to be compared
classifiers = {
    'Angle-based Outlier Detector (ABOD)':
        ABOD( contamination=outliers_fraction ),
    'Cluster-based Local Outlier Factor (CBLOF)':
        CBLOF(contamination=outliers_fraction,
              check_estimator=False, random_state=random_state),
    'Feature Bagging':
        FeatureBagging(LOF(n_neighbors=35),
                       contamination=outliers_fraction,
                       random_state=random_state),
    'Histogram-base Outlier Detection (HBOS)': HBOS(
        contamination=outliers_fraction),
    'Isolation Forest': IForest(contamination=outliers_fraction,
                                random_state=random_state),
    'K Nearest Neighbors (KNN)': KNN(
        contamination=outliers_fraction),
    'Average KNN': KNN(method='mean',
                       contamination=outliers_fraction),
    'Local Outlier Factor (LOF)':
        LOF(n_neighbors=35, contamination=outliers_fraction),
    'Minimum Covariance Determinant (MCD)': MCD(
        contamination=outliers_fraction, random_state=random_state),
    'One-class SVM (OCSVM)': OCSVM(contamination=outliers_fraction),
    'Principal Component Analysis (PCA)': PCA(
        contamination=outliers_fraction, random_state=random_state),
    'Locally Selective Combination (LSCP)': LSCP(
        detector_list, contamination=outliers_fraction,
        random_state=random_state)
}

# Show all detectors
for i, clf in enumerate( classifiers.keys() ):
    print('Model', i + 1, clf)

# Fit the models with the generated data and compare model performances

for i, offset in enumerate( clusters_separation ):
    print( "Features of interest: '%s', '%s'" % ( features_of_interest[ 0 ], features_of_interest[ 1 ] ) )
    X = np.c_[ df[ features_of_interest[ 0 ] ], df[ features_of_interest[ 1 ] ] ]
    print( "Size of the matrix of samples: %d, %d" % ( X.shape[ 0 ], X.shape[ 1 ] ) )
    print( X )


    plt.figure( figsize=(15, 12) )
    
    # Fit the model
    for i, (clf_name, clf) in enumerate( classifiers.items() ):
        # I am currently only familiar with the most frequent unsupervised and supervised classical ML algorithms. 
        # In particular, I am not familiar with deep learning algorithms, an issue that will interfere with my ability to debug
        # why some of algorithms may be crashing. I will therefore might need to temporarily skip using some of the classifiers, 
        # temporarely delaying debugging of their invocatiom, for the sake of letting the classifiers that follow be run, also.
        if ( clf_name in [ 'One-class SVM (OCSVM)' ] ):
             continue
        print(i + 1, 'fitting', clf_name)
 
        # fit the data and tag outliers. Since only one parameter is passed, the approach is one of unsupervised classification.
        clf.fit( X )
        # 0 - Inliers / 1 - Outliers
        scores_pred = clf.decision_function(X) * -1
                
        # Requesting a prediction using exactly the same dataset used for original training of the model (??) 
        y_pred = clf.predict(X)
        # Return the percentile of predicted outlier scores that is *above* the outlier threshold
        threshold = percentile( scores_pred, 100 * outliers_fraction )
        # We now count how many of the predictions with our predefined 'mask' for % of outliers in the sample.
        n_errors = ( y_pred != ground_truth ).sum()
        
        # We now predict what ought to be the value resulting for the input tuple of (xx, yy), on the basis if the decision function 
        # resulting from having trained the model with the applicable algorithm/classifier. 
        # The result is stored in a multidimensional matrix
        # Why the need for ravelling? Both xx and yy are square matrices. Why? Why not simple 1-D arrays.
        Z = clf.decision_function( np.c_[ xx.ravel(), yy.ravel() ] ) * -1
        # We resize it according to the x axis.
        Z = Z.reshape( xx.shape )
        # We will be drawing diagram #i+1, within a layout of 3 rows by 4 columns that starts on 
        # the upper left corner and ends on the right lower corner. 
        # subplot = plt.subplot( 3, 4, i + 1 )
        
        # We start by drawing the outer contours, covering the whole domain, from the lowest known value up to the threshold, filling 
        # the area delimited by each contour line with a different range of a palette of blues.
        #subplot.contourf( xx, yy, Z, levels=np.linspace( Z.min(), threshold, 7 ), cmap=plt.cm.Blues_r )
        plt.contourf( xx, yy, Z, levels=np.linspace( Z.min(), threshold, 7 ), cmap=plt.cm.Blues_r )
        
        # And then we draw the contour line for the threshold itself. 
        # a = subplot.contour( xx, yy, Z, levels=[threshold], linewidths=2, colors='red' ) 
        a = plt.contour( xx, yy, Z, levels=[threshold], linewidths=2, colors='red' ) 
        
        # We then proceed to draw inliners and outliers.
        # First, we fill the areas within the contour lines in orange, from the threshold, up to the maximum value.
        # subplot.contourf( xx, yy, Z, levels=[threshold, Z.max()], colors='orange' )
        plt.contourf( xx, yy, Z, levels=[threshold, Z.max()], colors='orange' )
        # Remember that matrix X contains feature 0 in the first column and feature 1 in the second column.
        # We start by drawing the true inliers
        # Plot all tuples of ( feature[ 0 ], feature[ 1 ] ), starting with the last rows in the column, and working your way backwards to its beguinning.
        # Use white markers of size 20.
        # b = subplot.scatter( X[:-n_outliers, 0], X[:-n_outliers, 1], c='white', s=20, edgecolor='k' )
        b = plt.scatter( X[:-n_outliers, 0], X[:-n_outliers, 1], c='white', s=20, edgecolor='k' )
        # Now, proceed to draw the true outliers
        # Now, plot starting from the first value in the column, but counting forwards towards the last row in the column.
        # Use black markers, of size 20, and edge  
        # c = subplot.scatter( X[-n_outliers:, 0], X[-n_outliers:, 1], c='black', s=20, edgecolor='k' )
        c = plt.scatter( X[-n_outliers:, 0], X[-n_outliers:, 1], c='black', s=20, edgecolor='k' )
 
        #with st.spinner( text='In progress' ):
        #    time.sleep( 5 )
        #    st.success( 'Done' )

        # plt.axis('tight')
        plt.legend(
            [ a.collections[0], b, c],
            [ 'learned decision function', 'true inliers', 'true outliers' ],
            prop=matplotlib.font_manager.FontProperties( size=10 ),
            loc='lower right' 
        )
        #plt.set_xlabel( "%s (errors: %d)" % ( clf_name, str( n_errors ) ) )
        #plt.set_xlim( ( MIN_X, MAX_X ) )
        #plt.set_ylim( ( MIN_Y, MAX_Y ) )
        
        with st.beta_expander( clf_name+' decision model: ('+str( n_errors )+' errors)' ):
            #st.write( "Scores prediction - ", scores_pred.shape, " rows: ", scores_pred )
            st.write( n_errors, "samples predicted by OD model to be outliers, in excess of the set threshold of", outliers_fraction*100, "% of total number of available samples" )
            #st.write( "Shapes - x: ", xx.shape, " , y: ", yy.shape, " , Z: ", Z.shape )
            #st.write( "Shapes - x ravel: ", xx.ravel().shape, " , y: ", yy.ravel().shape, " , Z: ", Z.shape )
            #st.write( "y_pred", y_pred )
            #st.write( "Contour lines (Z): " )
            # st.dataframe( pd.DataFrame( Z ) )
            #subplot.legend(
            #    [a.collections[0], b, c],
            #    ['learned decision function', 'true inliers', 'true outliers'],
            #    prop=matplotlib.font_manager.FontProperties(size=10),
            #    loc='lower right')
            # Left, Bottom, Right, Top        
            plt.subplots_adjust( 0.04, 0.1, 0.96, 0.94, 0.1, 0.26 )
            plt.suptitle( "Outlier detection: '"+ features_of_interest[ 0 ]+"' vs. '"+features_of_interest[ 1 ]+"'" )
            st.pyplot( plt )


#plt.show()

# I'm not totally sure I used the algorithm correctly, as I'm still trying to make sense of how could it be that 
# samples marked as "true outliers" or "true inliers" seems to be in reversed locations of where I would have expected 
# them to be (outliers and learned decision function seem to be clustering in the center of the scatterplot, wheras 
# I would have expected them to be in the outer ring of the oval).
# I still need to give it a thought on whether systematically correct forecasts, as tracked on the input datafile, 
# should have resulted in a hollow ovaled ring, and that the fact that outliers may be in the central part of the
# ring actually needs to be interpreted that the deviations of the source predictive model are precisely the ones 
# bunching up in the central part of the ring. (?)