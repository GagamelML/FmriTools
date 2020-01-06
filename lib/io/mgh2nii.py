def mgh2nii(filename, out_type, path_output):
    """
    This function converts a volume file from freesurfer mgh to nifti format.
    Inputs:
        *filename: full path of the input file.
        *out_type: target type of file.
        *path_outupt: path where output is written.
    
    created by Daniel Haenelt
    Date created: 06-01-2020             
    Last modified: 06-01-2020
    """        
    import os
    from lib.io.get_filename import get_filename
    from nipype.interfaces.freesurfer.preprocess import MRIConvert

    # get filename
    path, name, ext = get_filename(filename)

    # convert volume to nifti format
    mc = MRIConvert()
    mc.inputs.in_file = filename
    mc.inputs.out_file = os.path.join(path_output,name+".nii")
    mc.inputs.in_type = ext[1:]
    mc.inputs.out_type = out_type
    mc.run()