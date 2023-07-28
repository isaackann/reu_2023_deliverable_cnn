import argparse
import matplotlib.pyplot as plt
import nibabel as nib
import os
from time import sleep


#-------------------------------------------------------------------------------------------------
# Viewing function


def view_nii(filename, slice_index, name=None):
    data = nib.as_closest_canonical(nib.load(filename)).get_fdata()
    
    fig, ax = plt.subplots(1, 1)
    disp_kws = dict(cmap='Greys_r', vmin=0, vmax=1, origin='lower')
    
    ax.imshow(data[:, :, slice_index].T, **disp_kws)
    
    if name is None: name = filename
    
    fig.tight_layout()
    plt.savefig(f'{name}.png')


#-------------------------------------------------------------------------------------------------
# Parse user input

parser = argparse.ArgumentParser(description="View NII files.")
    
parser.add_argument("nii_filename", help="NII file to be viewed")
parser.add_argument("slice_index", help="Which layer of the brain (0 - 127) to view",
                    type=int)
parser.add_argument("--output_filename", help="Directory to save predicted AGR under",
                    default=None)

parser.add_argument("--anim", help="Display real-time NII animation in anim.png",
                    default=False)
parser.add_argument("--anim_delay", help="Frame delay in NII animation",
                    type=float, default=0.25)

args = parser.parse_args()

nii_filename = args.nii_filename
slice_index = args.slice_index
output_filename = args.output_filename

anim = args.anim
anim_delay = args.anim_delay


#-------------------------------------------------------------------------------------------------
# Execute

view_nii(nii_filename, slice_index, output_filename)

# Real-time animation
if anim:
    for i in range(128):
        sleep(anim_delay)
        view_nii(nii_filename, i, name='anim')
        
    os.remove('anim.png')    
