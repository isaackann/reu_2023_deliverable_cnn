import torch
import json
import nibabel as nib
import numpy as np
import pandas as pd, plotly.express as px
import matplotlib.pyplot as plt, matplotlib.animation as anim
from pathlib import Path
from torch.utils.data import Dataset
from torchvision.transforms import Compose, ToTensor, ConvertImageDtype
from functools import partial
from random import randint


""" Isaac Kan 7.27.23
    ----------------------------------------------------------------------------------------------
    This file contains functions and classes that handle data processing for the
    convolutional neural network created in model_modules.py.
"""


#-------------------------------------------------------------------------------------------------
# Functions to process input tensors


def find_bounds(tensor):
    """ 
    Find bounds of a 3D-tensor that excludes blackspace from an image represented as numpy array 
    """

    array = tensor.numpy()
    nonzero_idxs = np.nonzero(array > 0.45)
    
    min_x, max_x = np.min(nonzero_idxs[2]), np.max(nonzero_idxs[2])
    min_y, max_y = np.min(nonzero_idxs[1]), np.max(nonzero_idxs[1])
    min_z, max_z = np.min(nonzero_idxs[0]), np.max(nonzero_idxs[0])
    
    return (min_x, max_x, min_y, max_y, min_z, max_z)


def crop_tensors(array_list, bounds):
    """ 
    Given x, y, and z bounds, slices all arrays in an iterable accordingly
    """
    
    min_x, max_x, min_y, max_y, min_z, max_z = bounds
    
    return (arr[min_x : max_x + 1, min_y : max_y + 1, min_z : max_z + 1] for arr in array_list)
    

def get_tensors_from_nii(nib_filename, destination_list, norm=None, mode=None):
    """ 
    Extracts 128x128x128 tensor from NIfTI file into a provided list 
    
    The data is normalized with a provided normalization constant. If not provided,
    the constant is calculated by setting the 99.9th percentile pixel value to 1
    
    Use mode='Shifted Proton MRI' to shift Proton MRI by 3 pixels
    """

    data = nib.as_closest_canonical(nib.load(nib_filename)).get_fdata()
    
    if norm is None:
        norm = np.percentile(data, 99.9)
        
    data /= norm

    if mode == 'Shifted Proton MRI':  # For sanity check
        data = np.roll(data, 3, axis=0)
    
    tensorTransform = Compose([ToTensor(), ConvertImageDtype(torch.float32)])
    destination_list.append(tensorTransform(data))
    
    return norm


#-------------------------------------------------------------------------------------------------
# Custom Dataset implementations

class TrainingDataset(Dataset):
    """ 
    Custom training Dataset for Sodium & Proton MRI inputs with AGR outputs. 
    
    The 'data' at an index is a random n x n x n sub-image from the 
    Na MRI #1, Na MRI #2, and Proton MRI tensors
    
    The 'label' at an index is the corresponding n x n x n sample from the AGR MRI tensor 
    """
    
    def __init__(self, training_files, batch_size, n):
        self.na_mr_1_tensors, self.na_mr_2_tensors, self.pro_mr_tensors, self.agr_tensors = [], [], [], []
        self.batch_size = batch_size
        self.n = n
        
        # Append all 128x128x128 tensors in given file groups to respective lists
        for file_group in training_files:
            na_mr1_file, na_mr2_file, pro_mr_file, agr_file = file_group
            
            # Normalize Na MRI tensors using 1st Sodium MRI
            na_scale = get_tensors_from_nii(na_mr1_file, self.na_mr_1_tensors)
            get_tensors_from_nii(na_mr2_file, self.na_mr_2_tensors, norm=na_scale)
            get_tensors_from_nii(pro_mr_file, self.pro_mr_tensors)
            get_tensors_from_nii(agr_file, self.agr_tensors, norm=na_scale)
         
    def __len__(self):
        return self.batch_size
    
    def __getitem__(self, idx):
        # Hijack idx to randomly select a tensor from tensor list
        idx = randint(0, len(self.agr_tensors) - 1)
        
        na_mr_1, na_mr_2 = self.na_mr_1_tensors[idx], self.na_mr_2_tensors[idx]
        pro_mr, agr_mr = self.pro_mr_tensors[idx], self.agr_tensors[idx]
        
        # Crop tensors using bounds of Target AGR
        bounds = find_bounds(agr_mr)
        na_mr_1, na_mr_2, pro_mr, agr_mr = crop_tensors((na_mr_1, na_mr_2, pro_mr, agr_mr), bounds)

        # Concatenate Na and Proton MRI tensors along channel dimension
        concat_tensor = torch.stack((na_mr_1, na_mr_2, pro_mr), dim=0)
        
        # Randomly select a n x n x n patch from the concatenated & agr tensor
        x, y, z = agr_mr.shape
        x, y, z = randint(0, x - self.n), randint(0, y - self.n), randint(0, z - self.n)
        
        # Ensure correct channel dimensions 
        sample_tensor = concat_tensor[:, x : x + self.n, y : y + self.n, z : z + self.n]
        agr_tensor = agr_mr[x : x + self.n, y : y + self.n, z : z + self.n].unsqueeze(dim=0)
        
        return sample_tensor, agr_tensor


class ValidationDataset(Dataset):
    """ 
    Custom validation dataset for Sodium & Proton MRI inputs 
    
    The 'data' at an index is the concatenation of the full 128 x 128 x 128 tensors for the
    Na MRI #1, Na MRI #2, and Proton MRI files 
    """

    def __init__(self, validation_files):
        self.na_mr_1_tensors, self.na_mr_2_tensors, self.pro_mr_tensors, self.agr_tensors = [], [], [], []
        
        # Append all 128x128x128 tensors in given file groups to respective lists
        for file_group in validation_files:
            na_mr1_file, na_mr2_file, pro_mr_file, agr_file = file_group
            
            # Normalize Na MRI tensors using 1st Sodium MRI
            na_scale = get_tensors_from_nii(na_mr1_file, self.na_mr_1_tensors)
            get_tensors_from_nii(na_mr2_file, self.na_mr_2_tensors, norm=na_scale)
            get_tensors_from_nii(pro_mr_file, self.pro_mr_tensors)
            get_tensors_from_nii(agr_file, self.agr_tensors, norm=na_scale)
   
    def __len__(self):
        return len(self.agr_tensors)
    
    def __getitem__(self, idx):  
        na_mr_1, na_mr_2 = self.na_mr_1_tensors[idx], self.na_mr_2_tensors[idx]
        pro_mr, agr_mr = self.pro_mr_tensors[idx], self.agr_tensors[idx]
        
        # Concatenate Na and Proton MR tensors along channel dimension
        concat_tensor = torch.stack((na_mr_1, na_mr_2, pro_mr), dim=0)        
        agr_tensor = agr_mr.unsqueeze(dim=0)  # Add channel dimension to agr tensor

        return concat_tensor, agr_tensor


class InferenceDataset(Dataset):
    """ 
    Custom dataset for inference on a 128x128x128 tensor from NIfTI files 
    """

    def __init__(self, na_mri_1, na_mri_2, proton_mri):
        # Lists to store different MRI-type tensors
        self.na_mr_1_tensors, self.na_mr_2_tensors, self.pro_mr_tensors = [], [], []
        
        na_scale = get_tensors_from_nii(na_mri_1, self.na_mr_1_tensors)
        get_tensors_from_nii(na_mri_2, self.na_mr_2_tensors, norm=na_scale)
        get_tensors_from_nii(proton_mri, self.pro_mr_tensors)
                        
    def __len__(self):
        return 1
    
    def __getitem__(self, idx):  
        na_mr_1 = self.na_mr_1_tensors[idx]
        na_mr_2 = self.na_mr_2_tensors[idx]
        pro_mr = self.pro_mr_tensors[idx]
        
        # Concatenate Na and Proton MR tensors along channel dimension
        concat_tensor = torch.stack((na_mr_1, na_mr_2, pro_mr), dim=0)

        return concat_tensor, torch.zeros(128, 128, 128)  # Label not necessary for inference


class SanityDataset(Dataset):
    """ 
    Custom dataset used for sanity checks. Modes:
    - Replace 2nd Sodium MRI with 1st Sodium MRI
    - Replace 2nd Sodium MRI with black image
    - Replace Proton MRI with black image
    - Shift Proton MRI by 3 pixels
    - Randomly rotate all MRIs
    """

    def __init__(self, folder_list, mode):
        if isinstance(folder_list, str): folder_list = [folder_list]
        
        # Lists to store different MRI-type tensors
        self.na_mr_1_tensors, self.na_mr_2_tensors, self.pro_mr_tensors, self.agr_tensors = [], [], [], []
        self.mode = mode
        
        # Append all 128x128x128 tensors for given patient profile(s) to respective lists
        for folder in folder_list:
            subject = Path(f'00_reu_sodium_cnn_data/{folder}')
            na_mr1_file, na_mr2_file = subject / 'input_01_na_mr_1st_echo.nii', subject / 'input_02_na_mr_2nd_echo.nii'
            pro_mr_file, agr_file = subject / 'input_03_proton_mr.nii', subject / 'target_na_mr_agr.nii'

            # These tensors are always unaltered
            na_scale = get_tensors_from_nii(na_mr1_file, self.na_mr_1_tensors)
            get_tensors_from_nii(agr_file, self.agr_tensors, norm=na_scale)  

            if mode == 'Duplicate Na MRI':  # Replace second Na MRI with the first Na MRI
                get_tensors_from_nii(na_mr1_file, self.na_mr_2_tensors, norm=na_scale)
            elif mode == 'No Na2 MRI':  # Inserts flat image for second sodium MRI
                self.na_mr_2_tensors.append(torch.zeros(128, 128, 128))  
            else: get_tensors_from_nii(na_mr2_file, self.na_mr_2_tensors, norm=na_scale)
            
            if mode == 'Random Proton MRI':  # Inserts flat MRI image
                self.pro_mr_tensors.append(torch.zeros(128, 128, 128))
            elif mode == 'Shifted Proton MRI':  # Shifts Proton MRI image by 3 pixels
                get_tensors_from_nii(pro_mr_file, self.pro_mr_tensors, mode='Shifted Proton MRI')
            else: get_tensors_from_nii(pro_mr_file, self.pro_mr_tensors)
                                        
    def __len__(self):
        return len(self.agr_tensors)
    
    def __getitem__(self, idx):
        na_mr_1, na_mr_2 = self.na_mr_1_tensors[idx], self.na_mr_2_tensors[idx]
        pro_mr, agr_mr = self.pro_mr_tensors[idx], self.agr_tensors[idx]
        
        if self.mode == 'Random Rotation':
            num_rots = randint(1, 3)
            
            na_mr_1 = torch.rot90(na_mr_1, k=num_rots, dims=[1, 2])
            na_mr_2 = torch.rot90(na_mr_2, k=num_rots, dims=[1, 2])
            pro_mr = torch.rot90(pro_mr, k=num_rots, dims=[1, 2])
            agr_mr = torch.rot90(agr_mr, k=num_rots, dims=[1, 2])
            
            concat_tensor = torch.stack((na_mr_1, na_mr_2, pro_mr), dim=0)        
            agr_tensor = agr_mr.unsqueeze(dim=0)

            return concat_tensor, agr_tensor
                
        # Concatenate Na and Proton MR tensors along 4-D channel dimension
        concat_tensor = torch.stack((na_mr_1, na_mr_2, pro_mr), dim=0)        
        agr_tensor = agr_mr.unsqueeze(dim=0)  # Add channel dimension to agr tensor

        return concat_tensor, agr_tensor


#-------------------------------------------------------------------------------------------------
# Miscellaneous utility functions for data visualization
 

def make_line_graph(x_label, x_list, y_label, y_list):
    """ 
    Helper function to make a line graph given 2 lists & list labels 
    """

    df = pd.DataFrame(data={x_label: x_list, y_label: y_list})
    fig = px.line(df, x=x_label, y=y_label, title=f'CNN {y_label} over {x_list[-1]} {x_label}', line_shape="linear", render_mode="auto")
    
    # image formatting
    fig.update_layout(title_font_size=25)  # formatting
    fig.update_xaxes(title_font_size=20, tickfont_size=20)
    fig.update_yaxes(title_font_size=20, tickfont_size=20)
    fig.show()
    

""" Initialize figure for animation """
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3))
disp_kws = dict(cmap='Greys_r', vmin=0, vmax=1, origin='lower')
ax1.set_title('CNN Prediction')
ax2.set_title('Target AGR')


def display_frame(t, pidx):
    """ 
    Display single frame (sliced along z-axis) for GIF animation 
    """

    nn_filename = f'3d_agr_niis/{pidx}_agr.nii'
    alg_filename = f'00_reu_sodium_cnn_data/{pidx}/target_na_mr_agr.nii'
    
    nn_image_data = nib.as_closest_canonical(nib.load(nn_filename)).get_fdata()
    alg_image_data = nib.as_closest_canonical(nib.load(alg_filename)).get_fdata() 
    
    ax1.imshow(nn_image_data[:,:,t].T, **disp_kws)
    ax2.imshow(alg_image_data[:,:,t].T, **disp_kws)
    
    
def animate_agr(patient_index):
    """ 
    Read CNN-produced NII file & animate as a video with AGR for comparison 
    """

    animation = anim.FuncAnimation(fig, partial(display_frame, pidx=patient_index), frames=128, interval=100)
    print(f'{patient_index} Animation Rendered ... ', end='', flush=True)
    
    animation.save(f'mri_vids/{patient_index}_anim.mp4')
    print(f'{patient_index} Animation Saved!')


def save_to_json(data, folder, filename, config=None):
    """ 
    Serialize a list to a file 
    """
    
    json_filename = f'{folder}/{filename}.json' if config == True else f'{folder}/{filename}.txt'

    with open(json_filename, 'w') as f:
        json.dump(data, f)