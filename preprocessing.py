'''
It is a test written by Long trying to read the LO_DOSE_CT images from the data film
Oct 19, 2020
'''

import os
import sys
import SimpleITK as sitk
from PilLite import Image         # change to PilLite
import pydicom
import numpy as np
import matplotlib.pyplot as plt

#============================================  1. path  ===========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# c002 scan
dataset_dir_C002_0830201809658_full = os.path.abspath(os.path.join(BASE_DIR, "data", "LDCT-and-Projection-data", "C002", "08-30-2018-09658", "1.000000-Full dose images-31186"))

if not os.path.exists(dataset_dir_C002_0830201809658_full):
    raise Exception("\n{} dose not exists，please download dicom data and put it in \n{}".format(
        dataset_dir_C002_0830201809658_full, os.path.dirname(dataset_dir_C002_0830201809658_full)))

# Q: get one patient one time full dose scan  ? read 10 patients full dose files?

#============================================  2. read Dicom  ===========================

# load the scan, list in z direction and get the thickness
def load_scan(path):
    slices = [pydicom.read_file(path + '/' + s) for s in os.listdir(path)]
    slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))

    try:
        slice_thickness = np.abs(slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2])
    except:
        # if z data is missed, use this information
        slice_thickness = np.abs(slices[0].SliceLocation - slices[1].SliceLocation)

    for s in slices:
        s.SliceThickness = slice_thickness

    return slices


def get_pixels_hu(slices):
    image = np.stack([s.pixel_array for s in slices])
    # Convert to int16 (from sometimes int16),
    # should be possible as values should always be low enough (<32k)
    image = image.astype(np.int16)

    # Set outside-of-scan pixels to 0
    # The intercept is usually -1024, so air is approximately 0
    image[image == -2000] = 0

    # Convert to Hounsfield units (HU)
    for slice_number in range(len(slices)):

        intercept = slices[slice_number].RescaleIntercept
        slope = slices[slice_number].RescaleSlope

        if slope != 1:
            image[slice_number] = slope * image[slice_number].astype(np.float64)
            image[slice_number] = image[slice_number].astype(np.int16)

        image[slice_number] += np.int16(intercept)

    return np.array(image, dtype=np.int16)

patient_c002_full_slices = load_scan(dataset_dir_C002_0830201809658_full)
patient_c002_full_pixels = get_pixels_hu(patient_c002_full_slices)

plt.hist(patient_c002_full_pixels.flatten(), bins=80, color='c')
plt.xlabel("Hounsfield Units (HU)")
plt.ylabel("Frequency")
plt.show()

for s in range(10):
    plt.imshow(patient_c002_full_pixels[100+s], cmap=plt.cm.gray)
    plt.show()



'''
# Extracting data from the mri file
plan = pydicom.read_file(dataset_dir_C002)
shape = plan.pixel_array.shape
print(shape)

plt.imshow(plan.pixel_array, cmap=plt.cm.gray)
plt.show()
'''



