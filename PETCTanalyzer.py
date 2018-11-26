#!/usr/bin/env python
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
# PyWAD is open-source software and consists of a set of modules written in python for the WAD-Software medical physics quality control software. 
# The WAD Software can be found on https://github.com/wadqc
# 
# The pywad package includes modules for the automated analysis of QC images for various imaging modalities. 
# PyWAD has been originaly initiated by Dennis Dickerscheid (AZN), Arnold Schilham (UMCU), Rob van Rooij (UMCU) and Tim de Wit (AMC) 
#
#
# Changelog:
#   20181126: first complete version


from __future__ import print_function

__version__ = '20181123'
__author__ = 'jmgroen'

import os
# this will fail unless wad_qc is already installed
from wad_qc.module import pyWADinput
from wad_qc.modulelibs import wadwrapper_lib

import numpy as np
import scipy
if not 'MPLCONFIGDIR' in os.environ:
    # using a fixed folder is preferable to a tempdir, because tempdirs are not automatically removed
    os.environ['MPLCONFIGDIR'] = "/tmp/.matplotlib" # if this folder already exists it must be accessible by the owner of WAD_Processor 
import matplotlib
matplotlib.use('Agg') # Force matplotlib to not use any Xwindows backend.

# we need pydicom to read out dicom tags
try:
    import pydicom as dicom
except ImportError:
    import dicom

def logTag():
    return "[MRI ACR] "

import math    
    
def fitellipse( x, **kwargs ):  
    x = np.asarray( x )  
    ## Parse inputs  
    ## Default parameters  
    kwargs.setdefault( 'constraint', 'bookstein' )  
    kwargs.setdefault( 'maxits', 200 )  
    kwargs.setdefault( 'tol', 1e-5 )  
    if x.shape[1] == 2:  
        x = x.T  
    centroid = np.mean(x, 1)  
    x     = x - centroid.reshape((2,1))  
    ## Obtain a linear estimate  
    if kwargs['constraint'] == 'bookstein':  
        ## Bookstein constraint : lambda_1^2 + lambda_2^2 = 1  
        z, a, b, alpha = fitbookstein(x)
    
    ## Add the centroid back on  
    z = z + centroid  
    return z, a, b, alpha
    
def fitbookstein(x):  
    '''  
    function [z, a, b, alpha] = fitbookstein(x)  
    FITBOOKSTEIN  Linear ellipse fit using bookstein constraint  
      lambda_1^2 + lambda_2^2 = 1, where lambda_i are the eigenvalues of A  
    '''  
    ## Convenience variables  
    m = x.shape[1]  
    x1 = x[0, :].reshape((1,m)).T  
    x2 = x[1, :].reshape((1,m)).T  
    ## Define the coefficient matrix B, such that we solve the system  
    ## B *[v; w] = 0, with the constraint norm(w) == 1  
    B = np.hstack([ x1, x2, np.ones((m, 1)), np.power( x1, 2 ), np.multiply( np.sqrt(2) * x1, x2 ), np.power( x2, 2 ) ])  
    ## To enforce the constraint, we need to take the QR decomposition  
    Q, R = np.linalg.qr(B)  
    ## Decompose R into blocks  
    R11 = R[0:3, 0:3]  
    R12 = R[0:3, 3:6]  
    R22 = R[3:6, 3:6]  
    ## Solve R22 * w = 0 subject to norm(w) == 1  
    U, S, V = np.linalg.svd(R22)  
    V = V.T  
    w = V[:, 2]  
    ## Solve for the remaining variables  
    v = np.dot( np.linalg.solve( -R11, R12 ), w )  
    ## Fill in the quadratic form  
    A     = np.zeros((2,2))  
    A.ravel()[0]    = w.ravel()[0]  
    A.ravel()[1:3] = 1 / np.sqrt(2) * w.ravel()[1]  
    A.ravel()[3]    = w.ravel()[2]  
    bv     = v[0:2]  
    c     = v[2]  
    ## Find the parameters  
    z, a, b, alpha = conic2parametric(A, bv, c)  
    return z, a, b, alpha
    
def conic2parametric(A, bv, c):  
    '''  
    function [z, a, b, alpha] = conic2parametric(A, bv, c)  
    '''  
    ## Diagonalise A - find Q, D such at A = Q' * D * Q  
    D, Q = np.linalg.eig(A)  
    Q = Q.T  
    ## If the determinant &lt; 0, it's not an ellipse  
    # if prod(D) &lt;= 0:  
      # raise RuntimeError, 'fitellipse:NotEllipse Linear fit did not produce an ellipse'  
    ## We have b_h' = 2 * t' * A + b'  
    t = -0.5 * np.linalg.solve(A, bv)  
    c_h = np.dot( np.dot( t.T, A ), t ) + np.dot( bv.T, t ) + c  
    z = t  
    a = np.sqrt(-c_h / D[0])  
    b = np.sqrt(-c_h / D[1])  
    alpha = math.atan2(Q[0,1], Q[0,0])  
    return z, a, b, alpha    
    
def acqdatetime_series(data, results, action):
    try:
        params = action['params']
    except KeyError:
        params = {}
    ## 1. read only headers
    dcmInfile = dicom.read_file(data.series_filelist[0][0], stop_before_pixels=True)
    dt = wadwrapper_lib.acqdatetime_series(dcmInfile)
    results.addDateTime('AcquisitionDateTime', dt)

def header_series(data, results, action):  
    # get the first file
    instances = data.getAllInstances()
#    if len(instances) != 1:
#        print('%s Error! Number of instances not equal to 1 (%d). Exit.'%(logTag(),len(instances)))
#    instance=instances[0]
        
    # look in the config file for tags and write them as results, nested tags are supported 2 levels
    for key in action['tags']:
        varname=key
        tag=action['tags'][key]
        if tag.count('/')==0:
            value=instance[dicom.tag.Tag(tag.split(',')[0],tag.split(',')[1])].value
        elif tag.count('/')==1:
            tag1=tag.split('/')[0]
            tag2=tag.split('/')[1]
            value=instance[dicom.tag.Tag(tag1.split(',')[0],tag1.split(',')[1])][0]\
            [dicom.tag.Tag(tag2.split(',')[0],tag2.split(',')[1])].value
        elif tag.count('/')==2:
            tag1=tag.split('/')[0]
            tag2=tag.split('/')[1]
            tag3=tag.split('/')[2]
            value=instance[dicom.tag.Tag(tag1.split(',')[0],tag1.split(',')[1])][0]\
            [dicom.tag.Tag(tag2.split(',')[0],tag2.split(',')[1])][0]\
            [dicom.tag.Tag(tag3.split(',')[0],tag3.split(',')[1])].value
        else:
            # not more then 2 levels...
            value='too many levels'

        # write results
        results.addString(varname, str(value)[:min(len(str(value)),100)])    

def frange(start,stop,step):
    i=start
    while i<stop:
        yield i
        i+=step
        
def find_circ_object(img,savefig,fn):
    from skimage import feature
    edges = feature.canny(img, sigma=1, low_threshold=np.max(img)*0.8, high_threshold=np.max(img)*0.9)
    
    estimate_x=round(np.shape(img)[0]/2)
    estimate_y=round(np.shape(img)[1]/2)
    t=img[:,estimate_x]  # horizontal profile
    count_above=sum(i > np.mean(t) for i in t)
    count_below=sum(i < np.mean(t) for i in t)
    if count_above > count_below: #the biggest is probably the object.
        estimate_r=count_above/2
    else:
        estimate_r=count_below/2

#    max_estimate=estimate_r+round(estimate_r*0.1)
    max_estimate=estimate_r+10
    if max_estimate>round(np.shape(img)[0]/2):  #if object is close to the edge.
        max_estimate=round(np.shape(img)[0]/2)
        
    min_estimate=estimate_r-10
    if min_estimate<0:
        min_estimate=0
    
    edge_x=[]
    edge_y=[]
    edge_coor=[]
        
    for angle in range(0, 360, 1):
        v=[]
        r=[]
        for rsub in frange(min_estimate,max_estimate,1):
            x=int(round(rsub * math.sin(math.radians(angle)) + estimate_x))
            y=int(round(rsub * math.cos(math.radians(angle)) + estimate_y))
            v.append(edges[x,y])
            r.append(rsub)
        pixsum=0
        rowsum=0    

        for i in range(0,len(v),1):
            pixsum=pixsum+v[i]
            rowsum=rowsum+v[i]*r[i]
        if pixsum>0:
            r_edge=rowsum/pixsum
        else:
            r_edge=0

        xt=int(round(r_edge * math.sin(math.radians(angle)) + estimate_x))
        yt=int(round(r_edge * math.cos(math.radians(angle)) + estimate_y))
        edge_x.append(xt)
        edge_y.append(yt)
        edge_coor.append((xt,yt))

    [z, a, b, alpha]=fitellipse(edge_coor)
    
    if savefig:
        from matplotlib import pyplot as plt
        fig = plt.figure() 
        ax = plt.gca()
        plt.axis('off')
        plt.imshow(img,cmap=plt.cm.gray,aspect='equal')
    
        ellipse_x=[]
        ellipse_y=[]
        for angle in range(0,360,1):
            xe=(a*math.cos(math.radians(angle))+z[0])
            ye=(b*math.sin(math.radians(angle))+z[1])
            ellipse_x.append(xe)
            ellipse_y.append(ye)
    
        plt.plot(ellipse_y,ellipse_x,'r')
        plt.plot(z[1],z[0],'b+')
    
        varname = fn 
        filename = varname+'.png'
        plt.margins(0,0)
        plt.savefig(filename, bbox_inches="tight",pad_inches = 0, dpi=100) 
        results.addObject(varname,filename)
        plt.close()
    
    return z, a, b, alpha

def load_image_HU(dicomfile):
    image=dicomfile.pixel_array
    image = image.astype(np.int16)

    intercept = dicomfile.RescaleIntercept
    slope = dicomfile.RescaleSlope

    if slope != 1:
        image = slope * image.astype(np.float64)
        image = image.astype(np.int16)
        
    image += np.int16(intercept)
    
    return np.array(image, dtype=np.int16)

def circ_roi(xc,yc,r,img):
    H, W = img.shape
    x, y = np.meshgrid(np.arange(W), np.arange(H))
    d2 = (x - xc)**2 + (y - yc)**2
    mask = d2 < r**2
    
    ROImean=np.mean(img[mask])
    ROIstd=np.std(img[mask])
    ROIsize=np.sum(mask)
    
    return ROImean,ROIstd,ROIsize,mask
        
def Head_analysis(serie, results):
    img=load_image_HU(serie[0])

    z, a, b, alpha=find_circ_object(img,True,'HeadSegmentation')
    
    ROI=circ_roi(z[1],z[0],(np.mean([a,b]))*0.8,img)
    results.addFloat('Head_HU_water', ROI[0])
    results.addFloat('Head_Noise', ROI[1])
    
    radius_ROI=(np.mean([a,b]))*0.3
    
    ROI_center=circ_roi(z[1],z[0],radius_ROI,img)
    ROI_top=circ_roi(z[1],z[0]-2*radius_ROI,radius_ROI,img)
    ROI_bottom=circ_roi(z[1],z[0]+2*radius_ROI,radius_ROI,img)
    ROI_right=circ_roi(z[1]+2*radius_ROI,z[0],radius_ROI,img)
    ROI_left=circ_roi(z[1]-2*radius_ROI,z[0],radius_ROI,img)
    
    diffs=[abs(ROI_top[0]-ROI_center[0]),
       abs(ROI_bottom[0]-ROI_center[0]),
       abs(ROI_left[0]-ROI_center[0]),
       abs(ROI_right[0]-ROI_center[0])]

    CT_uniformity=np.max(diffs)
    results.addFloat('Uniformity', CT_uniformity)
    
def Body_analysis(serie, results):
    img=load_image_HU(serie[0])
    
    main_z, main_a, main_b, main_alpha=find_circ_object(img,True,'BodySegmentation')
    main_ROI=circ_roi(main_z[1],main_z[0],(np.mean([main_a,main_b]))*0.1,img)
    results.addFloat('Body_HU_main', main_ROI[0])
    
    water_img=img[int(main_z[0]-50):int(main_z[0]+50),int(main_z[1]-170):int(main_z[1]-65)]
    teflon_img=img[int(main_z[0]-40):int(main_z[0]+40),int(main_z[1]+85):int(main_z[1]+160)]

    water_z, water_a, water_b, water_alpha=find_circ_object(water_img,True,'Water')
    teflon_z, teflon_a, teflon_b, teflon_alpha=find_circ_object(teflon_img,True,'Teflon')
    
    water_ROI=circ_roi(water_z[1],water_z[0],(np.mean([water_a,water_b]))*0.8,water_img)
    results.addFloat('Body_HU_water', water_ROI[0])

    teflon_ROI=circ_roi(teflon_z[1],teflon_z[0],(np.mean([teflon_a,teflon_b]))*0.8,teflon_img)
    results.addFloat('Body_HU_teflon', teflon_ROI[0])
        
def CT_analysis(data, results, action):
   
    
    try:
        params = action['params']
    except KeyError:
        params = {}
    
    #print(params)

    # the module is running on a study level, we need to get the different series
    series=data.getAllSeries()
    
    for serie in series:
        if 'SeriesDescription' in serie[0]:
            if serie[0].SeriesDescription==params['ct_tests']['Head']['SeriesDescription']:
                print('Running head analysis')
                Head_analysis(serie, results)
            if serie[0].SeriesDescription==params['ct_tests']['Body']['SeriesDescription']:
                Body_analysis(serie, results)
                print('Running body analysis')
          
if __name__ == "__main__":
    #import the pyWAD framework and get some objects
    data, results, config = pyWADinput()

    # look in the config for actions and run them
    for name,action in config['actions'].items():
        if name=='ignore':
            s='s'
        
        # save acquisition time and date as result        
        elif name == 'acqdatetime':
           acqdatetime_series(data, results, action)

        # save whatever tag is requested as result
        elif name == 'header_series':
           header_series(data, results, action)

        # run the CT analysis
        elif name == 'ct_series':
            CT_analysis(data, results, action)

    results.write()

    # all done
