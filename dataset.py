import pandas as pd
import torch
from imblearn.over_sampling import SMOTE
class ISIC2024:
    non_features = ['isic_id', 'target', 'patient_id',
                'image_type', 'tbp_tile_type', 'attribution', 'copyright_license', 
                'lesion_id', 'iddx_full', 'iddx_1', 'iddx_2', 'iddx_3', 
                'iddx_4', 'iddx_5', 'mel_mitotic_index', 'mel_thick_mm'
                ]

    categorical_features = ['sex','anatom_site_general',
                            'tbp_lv_location', 'tbp_lv_location_simple'
                        ]

    numerical_features = ['tbp_lv_eccentricity', 'tbp_lv_area_perim_ratio', 'tbp_lv_deltaB',
                            'tbp_lv_norm_border', 'tbp_lv_norm_color', 'tbp_lv_symm_2axis',
                            'tbp_lv_deltaA', 'tbp_lv_perimeterMM', 'tbp_lv_areaMM2', 'tbp_lv_deltaL', 
                            'tbp_lv_B', 'tbp_lv_x', 'age_approx', 'tbp_lv_nevi_confidence', 'tbp_lv_Cext', 
                            'tbp_lv_dnn_lesion_confidence', 'tbp_lv_deltaLB', 'clin_size_long_diam_mm', 
                            'tbp_lv_Lext', 'tbp_lv_L', 'tbp_lv_stdL', 'tbp_lv_z', 'tbp_lv_Aext', 
                            'tbp_lv_radial_color_std_max', 'tbp_lv_symm_2axis_angle', 'tbp_lv_Bext', 
                            'tbp_lv_A', 'tbp_lv_H', 'tbp_lv_C', 'tbp_lv_stdLExt', 'tbp_lv_y', 
                            'tbp_lv_Hext', 'tbp_lv_minorAxisMM', 'tbp_lv_deltaLBnorm', 'tbp_lv_color_std_mean'
                            ]

    encoded_categorical_features = [
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

    def __init__(self, path, folder, features=encoded_categorical_features + numerical_features, train=True,
                 return_tensors=False, balance=False) -> None:
        self.metadata = pd.read_csv(path)
        mask = self.metadata['folder'] == folder
        self.labels = self.metadata['target'][~mask if train else mask]
        self.metadata = self.metadata[~mask if train else mask][features]
        
        if balance:
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