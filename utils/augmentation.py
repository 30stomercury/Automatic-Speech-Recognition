import sox
import os
import numpy as np
from tqdm import tqdm

def SpeedAugmentation(filelist, target_folder, speed):
    """Speed Augmentation
    
    Args:
        file_list: Path to audio files.
        target_folder: Folder of augmented audios.
        speed: Speed for augmentation.
    """
    
    audio_path = []
    aug_generator = sox.Transformer()
    print("Total audios:", len(filelist))
    aug_generator.speed(speed)
    target_folder_ = target_folder+"_"+str(speed)
    if not os.path.exists(target_folder_):
        os.makedirs(target_folder_)
    for source_filename in tqdm(filelist):
        file_id = source_filename.split("/")[-1]
        save_filename = target_folder_+"/"+file_id.split(".")[0]+"_"+str(speed)+"."+file_id.split(".")[1] 
        if os.path.isfile(save_filename):
            print("File exist!")
        else:
            aug_generator.build(source_filename, save_filename)
        audio_path.append(save_filename)

    return audio_path

def VolumeAugmentation(filelist, target_folder, vol_range):
    """Volume Augmentation
    
    Args:
        file_list: Path to audio files.
        target_folder: Folder of augmented audios.
        volume_range: Range of volumes for augmentation.
    """

    audio_path = []
    aug_generator = sox.Transformer()
    print("Total audios:", len(filelist))
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)
    for source_filename in tqdm(filelist):
        volume = np.around(
                np.random.uniform(vol_range[0],vol_range[1]), 2)
        aug_generator.vol(volume)
        file_id = source_filename.split("/")[-1]
        save_filename = target_folder+"/"+file_id.split(".")[0]+"_"+str(volume)+"."+file_id.split(".")[1] 
        aug_generator.build(source_filename, save_filename)
        aug_generator.clear_effects()
        audio_path.append(save_filename)
    return audio_path

