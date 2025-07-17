import pandas as pd
import torch
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

class ISIC2024:
    PATIENT_ID = 'patient_id'
    NON_FEATURES = ['isic_id', 'target', 'patient_id',
                'image_type', 'tbp_tile_type', 'attribution', 'copyright_license', 
                'lesion_id', 'iddx_full', 'iddx_1', 'iddx_2', 'iddx_3', 
                'iddx_4', 'iddx_5', 'mel_mitotic_index', 'mel_thick_mm'
                ]
    
    RAW_CATEGORICAL_FEATURES = [ 'sex', 'anatom_site_general', 'tbp_lv_location', 'tbp_lv_location_simple',]
    
    NUMERICAL_FEATURES = [
        'tbp_lv_eccentricity', 
        'tbp_lv_area_perim_ratio', 
        'tbp_lv_deltaB',
        'tbp_lv_norm_border',
        'tbp_lv_norm_color',
        'tbp_lv_symm_2axis',
        'tbp_lv_deltaA',
        'tbp_lv_perimeterMM',
        'tbp_lv_areaMM2',
        'tbp_lv_deltaL', 
        'tbp_lv_B',
        'tbp_lv_x',
        'age_approx',
        'tbp_lv_nevi_confidence',
        'tbp_lv_Cext', 
        'tbp_lv_dnn_lesion_confidence',
        'tbp_lv_deltaLB',
        'clin_size_long_diam_mm', 
        'tbp_lv_Lext', 'tbp_lv_L',
        'tbp_lv_stdL',
        'tbp_lv_z',
        'tbp_lv_Aext', 
        'tbp_lv_radial_color_std_max',
        'tbp_lv_symm_2axis_angle',
        'tbp_lv_Bext', 
        'tbp_lv_A',
        'tbp_lv_H', 
        'tbp_lv_C',
        'tbp_lv_stdLExt',
        'tbp_lv_y', 
        'tbp_lv_Hext',
        'tbp_lv_minorAxisMM',
        'tbp_lv_deltaLBnorm',
        'tbp_lv_color_std_mean'
    ]

    ENCODED_CATEGORICAL_FEATURES = [
        'sex_female', 'sex_male',
        'anatom_site_general_anterior torso', 'anatom_site_general_head/neck',
        'anatom_site_general_lower extremity',
        'anatom_site_general_posterior torso',
        'anatom_site_general_upper extremity', 'tbp_lv_location_Head & Neck',
        'tbp_lv_location_Left Arm', 'tbp_lv_location_Left Arm - Lower',
        'tbp_lv_location_Left Arm - Upper', 'tbp_lv_location_Left Leg',
        'tbp_lv_location_Left Leg - Lower', 'tbp_lv_location_Left Leg - Upper',
        'tbp_lv_location_Right Arm', 'tbp_lv_location_Right Arm - Lower',
        'tbp_lv_location_Right Arm - Upper', 'tbp_lv_location_Right Leg',
        'tbp_lv_location_Right Leg - Lower',
        'tbp_lv_location_Right Leg - Upper', 'tbp_lv_location_Torso Back',
        'tbp_lv_location_Torso Back Bottom Third',
        'tbp_lv_location_Torso Back Middle Third',
        'tbp_lv_location_Torso Back Top Third', 'tbp_lv_location_Torso Front',
        'tbp_lv_location_Torso Front Bottom Half',
        'tbp_lv_location_Torso Front Top Half', #'tbp_lv_location_Unknown',
        'tbp_lv_location_simple_Head & Neck', 'tbp_lv_location_simple_Left Arm',
        'tbp_lv_location_simple_Left Leg', 'tbp_lv_location_simple_Right Arm',
        'tbp_lv_location_simple_Right Leg', 'tbp_lv_location_simple_Torso Back',
        'tbp_lv_location_simple_Torso Front', #'tbp_lv_location_simple_Unknown'
    ]

    TARGET_COLUMN = 'target'
    FOLDER_COLUMN = 'folder'

    def __init__(self, path, folder, features=ENCODED_CATEGORICAL_FEATURES + NUMERICAL_FEATURES, train=True,
                 return_tensors=False, balance=False) -> None:
        self.metadata = pd.read_csv(path) if ISIC2024.metadata is None else ISIC2024.metadata
        ISIC2024.metadata = self.metadata.copy()
        mask = self.metadata[ISIC2024.FOLDER_COLUMN] == folder
        self.labels = self.metadata[ISIC2024.TARGET_COLUMN][~mask if train else mask]
        self.lbs = self.labels.copy().values
        self.metadata = self.metadata[~mask if train else mask][features]

        self.index_categorical_features = [list(self.metadata.columns).index(x) for x in ISIC2024.RAW_CATEGORICAL_FEATURES]
        self.index_numerical_features = [list(self.metadata.columns).index(x) for x in ISIC2024.NUMERICAL_FEATURES]

        if balance:
            sampler = RandomUnderSampler(sampling_strategy=0.01)
            self.metadata, self.labels = sampler.fit_resample(self.metadata, y=self.labels)
        elif balance == 'smote':
            smote = SMOTE()
            self.metadata, self.labels = smote.fit_resample(self.metadata, y=self.labels)
        self.return_tensors = return_tensors
        if return_tensors:
            self.metadata = torch.Tensor(self.metadata.values)
            self.labels = torch.LongTensor(self.labels.values)
    
    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, index):
        return self.metadata[index], self.labels[index]