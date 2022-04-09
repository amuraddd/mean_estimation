import glob
import pandas as pd
import numpy as np
import nibabel as nib
from nibabel.testing import data_path
import matplotlib.pyplot as plt

def get_mean_pixel_values(data_location='images/ADNI_data/*.nii', img_slice=20, num_images_to_use='all', plot_image_slice=True, plot_mean_image_slice=True):
    """
    Take in images files. Combine the same slices and return mean pixel values.
    """
    pixel_values_by_slice = list()
    files = glob.glob(data_location)
    
    if num_images_to_use=='all':
        n = len(files)
    else:
        n = num_images_to_use
        
    for file in glob.glob(data_location)[:n]:
        images = nib.load(file) #load the data
        data = images.get_fdata().T #transpose the original data - it should fit the format 95*79
        if plot_image_slice:
            plt.imshow((data[img_slice]), cmap='gray')
            plt.title(file)
            plt.show()
        pixel_values_by_slice.append(data[img_slice])
    
    mean_pixel_values = np.array(pixel_values_by_slice).mean(axis=0)
    if plot_mean_image_slice:
        plt.imshow(mean_pixel_values, cmap='gray')
        plt.show()
    mean_pixel_values_flattened = mean_pixel_values.ravel()
    return mean_pixel_values, mean_pixel_values_flattened

def preprocess_image(data_location='images/ADNI_data/*.nii', img_slice=20, num_images_to_use='all', plot_image_slice=True, plot_mean_image_slice=True):
    """
    Input: 
        img: Image as nii file - data cube with slices for an image
        img_slice: integer to specify which slice to select
    Return 
        X: dataframe with two columns containing X and Y coordinates.
        Y: flattened pixel values
    """
    pixel_values, pixel_values_flattened = get_mean_pixel_values(data_location=data_location,
                                                                 img_slice=img_slice,
                                                                 num_images_to_use=num_images_to_use,
                                                                 plot_image_slice=plot_image_slice,
                                                                 plot_mean_image_slice=plot_mean_image_slice)
    
    # Y = data[img_slice].ravel() #flatten the matrix of pixels into a single array
    Y = pd.DataFrame(pixel_values_flattened, columns=['pixel_value']) #for the first image
    
    #get the number of rows and columns for the matrix of pixels per image
    rows = pixel_values.shape[0] #number of rows
    cols = pixel_values.shape[1] #number of columns

    #generate coordinates
    row_indices = list()
    column_indices = list()

    row_coordinates =  (np.indices(dimensions=(rows, cols))[0]+1)*(1/rows) #add 1 to avoid start from 0 - multiply by 1/cols to notmalize
    column_coordinates =  (np.indices(dimensions=(rows, cols))[1]+1)*(1/cols) #add 1 to avoid start from 0 - multiply by 1/cols to notmalize

    for row in row_coordinates:
        for row_index in row:
            row_indices.append(row_index)

    for row in column_coordinates:
        for column_index in row:
            column_indices.append(column_index)
            
    X = pd.DataFrame(columns=['X_coordinate', 'Y_coordinate'])
    X['X_coordinate'] = np.array(row_indices)
    X['Y_coordinate'] = np.array(column_indices)
    
    return X, Y

def high_resolution_coordinates(num_rows=100, num_cols=100):
    """
    Input: 
        num_rows: pixel rows
        num_cols: pixel cols
    Return 
        X: dataframe with two columns containing X and Y coordinates.
        Y: flattened pixel values
    """
    
    #get the number of rows and columns for the matrix of pixels per image
    rows = num_rows #number of rows
    cols = num_cols #number of columns

    #generate coordinates
    row_indices = list()
    column_indices = list()

    row_coordinates =  (np.indices(dimensions=(rows, cols))[0]+1)*(1/rows) #add 1 to avoid start from 0 - multiply by 1/cols to notmalize
    column_coordinates =  (np.indices(dimensions=(rows, cols))[1]+1)*(1/cols) #add 1 to avoid start from 0 - multiply by 1/cols to notmalize

    for row in row_coordinates:
        for row_index in row:
            row_indices.append(row_index)

    for row in column_coordinates:
        for column_index in row:
            column_indices.append(column_index)
            
    X = pd.DataFrame(columns=['X_coordinate', 'Y_coordinate'])
    X['X_coordinate'] = np.array(row_indices)
    X['Y_coordinate'] = np.array(column_indices)
    
    return X
