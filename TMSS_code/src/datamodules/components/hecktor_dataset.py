import os
from traceback import print_tb
from typing import Callable, Optional, Tuple
import sys

import SimpleITK as sitk
from pathlib import Path
import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np

import nibabel as nib
import pathlib
from einops import rearrange

from joblib import Parallel, delayed

from sklearn.preprocessing import scale

from torchmtlr.torchmtlr.utils import make_time_bins, encode_survival

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('max_colwidth', 2000)


def find_centroid(mask: sitk.Image):
    stats = sitk.LabelShapeStatisticsImageFilter()
    stats.Execute(mask)
    centroid_coords = stats.GetCentroid(1)
    centroid_idx = mask.TransformPhysicalPointToIndex(centroid_coords)

    return np.asarray(centroid_idx, dtype=np.float64)


def get_paths_to_patient_files(path_to_imgs, PatientID, append_mask=True):
    path_to_imgs = pathlib.Path(path_to_imgs)

    # patients = [p for p in PatientID if os.path.isdir(path_to_imgs / p)]
    paths = []
    for p in PatientID:
        path_to_ct = path_to_imgs / 'images' / (p + '_before.nii')
        path_to_pt = path_to_imgs / 'images' / (p + '_after.nii')

        if append_mask:
            path_to_mask = path_to_imgs / 'mask' / (p + '.nii')
            # paths.append((path_to_ct, path_to_pt, path_to_mask))
            paths.append((path_to_ct, path_to_mask))
        else:
            # paths.append((path_to_ct, path_to_pt))
            paths.append((path_to_ct))
    return paths


class HecktorDataset(Dataset):

    def __init__(self,
                 root_directory: str,
                 clinical_data_path: str,  # data_dir:  /home/sribd/Desktop/TMSS_EC_Sorted
                 patch_size: int = 50,
                 time_bins: int = 14,
                 cache_dir: str = "data_cropped/data_cache/",
                 transform: Optional[Callable] = None,
                 num_workers: int = 1
                 ):
        print(cache_dir)
        self.num_of_seqs = 1  # CT PT !!!

        self.root_directory = root_directory
        self.patch_size = patch_size

        self.transforms = transform
        self.num_workers = num_workers

        self.clinical_data = self.make_data(clinical_data_path)
        self.time_bins = make_time_bins(times=self.clinical_data["time"], num_bins=time_bins,
                                        event=self.clinical_data["event"])
        self.y = encode_survival(self.clinical_data["time"].values, self.clinical_data["event"].values,
                                 self.time_bins)  # single event

        self.cache_path = get_paths_to_patient_files(cache_dir, self.clinical_data['name'])

    def make_data(self, path):

        print('path in make_data:   ', path)
        df = pd.read_csv(path + '/EC.csv')
        # try:
        #    df = pd.read_csv(path + '/before.csv')
        # except:
        #    df = path
        clinical_data = df

        # clinical_data = clinical_data.rename(columns={"OS": "event", "OS(m)": "time"})
        # 标准化处理
        clinical_data["HGB_Before_Treatment"] = scale(clinical_data["HGB_Before_Treatment"])
        clinical_data["HGB_After_Treatment"] = scale(clinical_data["HGB_After_Treatment"])
        clinical_data["MWT_Before_Treatment"] = scale(clinical_data["MWT_Before_Treatment"])
        clinical_data["MWT_After_Treatment"] = scale(clinical_data["MWT_After_Treatment"])
        clinical_data["NS_Before_Treatment"] = scale(clinical_data["NS_Before_Treatment"])
        clinical_data["NS_After_Treatment"] = scale(clinical_data["NS_After_Treatment"])
        # clinical_data.iloc[:, 32:] = scale(clinical_data.iloc[:, 32:])

        # types = clinical_data.dtypes.to_dict()
        # for i, (col, dt) in enumerate(types.items()):
        #     print(f"{i}  Column '{col}': '{dt}'")
        # sys.exit()

        clinical_data = pd.get_dummies(clinical_data,
                                       columns=["Gender", "ECOG", "Smoking_and_Alcohol", "Family_History", "Location",
                                                "T", "N", "Supraclavicular_LN", "TNM", "Treatment_Response", "PTV_Dose",
                                                "GTV_Dose", "Concurrent_Chemotherapy"], drop_first=False, )
        # 转为独热编码
        columns_to_drop = ['LC', 'LC_m', 'LRFS', 'LRFS_m', 'OS', 'OS_m']
        clinical_data.drop(columns_to_drop, axis=1, inplace=True)

        # 填充缺失值
        columns_to_fill = ['Age', 'TL']
        clinical_data[columns_to_fill] = clinical_data[columns_to_fill].fillna(clinical_data[columns_to_fill].mean())
        # 填充缺失值

        # clinical_data["Age"] = scale(clinical_data["Age"])

        # print(clinical_data.isna().sum())
        # sys.exit()
        print('clinical_data:   ', clinical_data.info())
        return clinical_data

    def _prepare_data(self):
        pass

        # Parallel(n_jobs=self.num_workers)(
        #     delayed(self._preprocess_subject)(subject_id)
        #     for subject_id in self.clinical_data["name"]
        # )

    def _preprocess_subject(self, subject_id: str):

        print(self.root_directory)
        print('\n' + '---------------------------------------' + '\n')
        print('subject_id:    ', subject_id)

        # path = os.path.join(self.root_directory, "data/hecktor_nii/"
        #                     "{}",f"{subject_id}"+"{}"+".nii")
        #
        # image = sitk.ReadImage(path.format("images", "_ct"))
        # mask = sitk.ReadImage(path.format("masks", "_gtvt"))
        #
        # #crop the image to (patch_size)^3 patch around the tumor center
        # tumour_center = find_centroid(mask)
        # size = np.ceil(self.patch_size / np.asarray(image.GetSpacing())).astype(np.int) + 1
        # min_coords = np.floor(tumour_center - size / 2).astype(np.int64)
        # max_coords = np.floor(tumour_center + size / 2).astype(np.int64)
        # min_x, min_y, min_z = min_coords
        # max_x, max_y, max_z = max_coords
        # image = image[min_x:max_x, min_y:max_y, min_z:max_z]
        #
        # # resample to isotropic 1 mm spacing
        # reference_image = sitk.Image([self.patch_size]*3, sitk.sitkFloat32)
        # reference_image.SetOrigin(image.GetOrigin())
        # image = sitk.Resample(image, reference_image)
        #
        # # window image intensities to [-500, 1000] HU range
        # image = sitk.Clamp(image, sitk.sitkFloat32, -500, 1000)
        #
        # sitk.WriteImage(image, os.path.join(self.cache_path, f"{subject_id}.nii"), True)

    def __getitem__(self, idx: int):
        """Get an input-target pair from the dataset.

        The images are assumed to be preprocessed and cached.

        Parameters
        ----------
        idx
            The index to retrieve (note: this is not the subject ID).

        Returns
        -------
        tuple of torch.Tensor and int
            The input-target pair.
        """
        clin_name = self.clinical_data.iloc[idx]['name']
        try:  # training data
            # clin_var_data = self.clinical_data.drop(["target_binary", 'time', 'event', 'Study ID'], axis=1) # single event
            clin_var_data = self.clinical_data.drop(['name', 'event', 'time'], axis=1)
        except:  # test data
            clin_var_data = self.clinical_data.drop(['name'], axis=1)
        clin_var = clin_var_data.iloc[idx].to_numpy(dtype='float32')
        target = self.y[idx]
        labels = self.clinical_data.iloc[idx].to_dict()
        subject_id = self.clinical_data.iloc[idx]["name"]
        # path = self.cache_path, f"{subject_id}_ct.nii.gz")
        #         print('hi:', path)

        # image = sitk.ReadImage(path)
        # if self.transform is not None:
        #     image = self.transform(image)
        sample = dict()
        id_ = self.cache_path[idx][0].parent.stem
        sample['id'] = id_
        img = [self.read_data(self.cache_path[idx][i])[:, :, :] for i in range(self.num_of_seqs)]
        # if img[0].shape[2] != 48 or img[1].shape[2] != 48:
        #     print(clin_name)
        try:
            img = np.stack(img, axis=-1)  # 336 336 64 batch*2
        except:
            print(img[0].shape, img[1].shape, clin_name)
        img = rearrange(img, 'h w d c -> c h w d')
        sample['input'] = img  # np.expand_dims(img, axis=0)
        mask = self.read_data(self.cache_path[idx][-1])[:, :, :]
        # print(mask.shape)
        mask = np.expand_dims(mask, axis=3)  # 336 336 64 batch
        mask = rearrange(mask, 'h w d c->c h w d')
        sample['target_mask'] = mask
        # print('input  ', sample['input'].shape)                 #  (160, 160, 64, 1)
        # print('mask   ', sample['target_mask'].shape)           #  (160, 160, 64, 1)
        if self.transforms:
            sample = self.transforms(sample)
        return (sample, clin_var), target, labels

    def __len__(self) -> int:
        """Return the length of the dataset."""
        return len(self.clinical_data)

    @staticmethod
    def read_data(path_to_nifti, return_numpy=True):
        """Read a NIfTI image. Return a numpy array (default) or `nibabel.nifti1.Nifti1Image` object"""
        if return_numpy:
            return nib.load(str(path_to_nifti)).get_fdata()
        return nib.load(str(path_to_nifti))
