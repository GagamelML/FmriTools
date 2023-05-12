import os

# Exceptions and Errors
class ShapeError(Exception):
    pass

# basic tools
def naming(source_full: str, new_name: str, new_dir: str, ext: str = '.nii', prefix: str = '', suffix: str = '', subfolder: str = None):
    """
    Figures out the desired path and location of a new file depending on variable input. If new_name and new_path
    combined already represent a complete file location, that will be used, disregarding any other input.
    Otherwise, the missing parts will be fill in from other input.
    Intended for use in file processing, where the new output needs to be saved somewhere.
    Parameters
    ----------
    source_full: str
        Full name and path of the original file. If new_name or new_path are None, they will be taken from this string.
    new_name: str
        Desired name of output file. Use 'None' to leave out. Can contain file extension that takes priority. Otherwise,
        the 'ext' parameter will be used as extension.
    new_dir: str
        Desired path of output file. Use 'None' to leave out.
    ext: str [Optional]
        Use this file extension if none is given in 'new_name'. Default is '.nii'
    prefix, suffix: str [Optional]
        Add this to name of source file if no 'new_name' is given. Default is ''. Note that if both parameters are left
        as '' and no 'new_name' or 'new_path' are given, 'out_full' may match the input file (depending on ext) and
        the source file may be overwritten.
    subfolder: str [Optional]
        If new_dir is none, subfolder will be appended to source dir as output dir
    Returns string with full combined path, name and ext of new file
    -------

    """

    source_path, source_name = os.path.split(source_full)
    source_base, source_ext = os.path.splitext(source_name)
    source_base, source_ext2 = os.path.splitext(source_base)

    if ext is None:
        ext = source_ext2 + source_ext

    if new_name is None:
        new_name = prefix + source_base + suffix + ext
    else:
        new_base, new_ext = os.path.splitext(new_name)
        new_base, new_ext2 = os.path.splitext(new_base)
        new_ext = new_ext2 + new_ext
        if new_ext == '':
            new_name += ext

    if new_dir is None:
        if subfolder is None:
            new_dir = source_path
        else:
            new_dir = os.path.join(source_path, subfolder)
    if not os.path.isdir(new_dir):
        os.mkdir(new_dir)

    if not os.path.isdir(new_dir):
        os.mkdir(new_dir)

    out_path = os.path.join(new_dir, new_name)

    return out_path
