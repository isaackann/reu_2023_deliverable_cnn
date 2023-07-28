import argparse


""" Isaac Kan 7.27.23
    ----------------------------------------------------------------------------------------------
    This file uses a convolutional neural network to produce an anatomically-guided reconstruction
    of Sodium MRI images using a structural Proton MRI.
    
    The model accepts 2 Sodium MRI NIfTI files and one Proton MRI NIfTI files.
    A NIfTI file is produced for the reconstruction. 
"""


def main():
    #-------------------------------------------------------------------------------------------------
    # Get filename args from terminal
    
    parser = argparse.ArgumentParser(
        description="U-Net CNN Prediction of Anatomy-Guided Sodium MRI reconstruction.")
    
    parser.add_argument("na_mri_1", help="Absolute path of first Sodium MRI NII directory")
    parser.add_argument("na_mri_2", help="Absolute path of second Sodium MRI NII directory")
    parser.add_argument("proton_mri", help="Absolute path of Proton MRI NII directory")
    parser.add_argument("weights_filename", help="Absolute path of CNN weights file")
    parser.add_argument("--output_filename", help="Directory to save predicted AGR under",
                        default=None)
    
    args = parser.parse_args()
    
    #-------------------------------------------------------------------------------------------------
    # Load libraries and modules

    import torch
    import torch.nn as nn
    import nibabel as nib
    import numpy as np
    from model_modules import SuperResUNetCNN
    from model_data_structures import InferenceDataset
    from torch.utils.data import DataLoader
    from torchvision.transforms import ConvertImageDtype
    from datetime import datetime
    
    #-------------------------------------------------------------------------------------------------
    # Parse user input
    
    na_mri_1 = args.na_mri_1
    na_mri_2 = args.na_mri_2
    proton_mri = args.proton_mri
    
    weights_filename = args.weights_filename
    output_filename = args.output_filename
    
    if output_filename is None: 
        output_filename = f"prediction_{datetime.now().strftime('%m_%d')}"
    
    #-------------------------------------------------------------------------------------------------
    # Load trained model

    model = SuperResUNetCNN()
    model = nn.DataParallel(model)  # Supports multiple GPU usage
    
    model.load_state_dict(torch.load(weights_filename))
    model.eval()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)  # Handle device management
    
    #-------------------------------------------------------------------------------------------------
    # Load and preprocess data from input Sodium/Proton MRI and target Sodium AGR files
        
    dataset = InferenceDataset(na_mri_1,na_mri_2, proton_mri)
    dataloader = DataLoader(dataset, batch_size=1)
    
    #-------------------------------------------------------------------------------------------------
    # Make prediction
    
    prediction = None
    
    with torch.no_grad():
        tensors = next(iter(dataloader)) # Only one iteration for the only tensor in dataset
        concat_tensor = tensors[0].to(device)

        pred = model(concat_tensor)
        prediction = ConvertImageDtype(dtype=torch.float32)(pred[0][0])
            
    # Convert tensor to transposed numpy array to become a Nifti file
    pred_array = prediction.detach().cpu().numpy()
    pred_array = np.transpose(pred_array, (1, 2, 0))
        
    # Retrieve header and affine matrix for nifti file
    target_image = nib.load(na_mri_1)
    target_header, target_affine = target_image.header, target_image.affine
    
    # Create and save new Nifti image objects
    nifti_image = nib.Nifti1Image(pred_array, target_affine, header=target_header)
    nib.save(img=nifti_image, filename=output_filename)
    
    print("CNN AGR saved as NII!")
    

if __name__ == "__main__":
    main()