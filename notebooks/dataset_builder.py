import numpy as np
import pandas as pd

class DatasetBuilder:
    """
    Attributes:
        dict (dict): Dictionary mapping region names to pandas DataFrames containing glacier data.
        surge_path (str): File path to the surge glacier CSV file.
        dc_path (str): File path to the debris-cover ratio CSV file.
        save (bool): Flag indicating whether to save processed datasets to disk.
        surge (pd.DataFrame): DataFrame containing surge glacier information.
        dc (pd.DataFrame): DataFrame containing debris-cover ratio information.
    Methods:
        __init__(region_dict, save=False)
            Initializes the DatasetBuilder with region data and loads auxiliary datasets.
        iter_regs()
            Iterates over all regions in the dictionary, processes each DataFrame,
            and optionally saves the processed dataset to a CSV file.
        _build_ds_from_df(c_df)
            Processes a single region DataFrame by trimming columns, filtering surge glaciers,
            adding debris-cover ratio, computing hypsometric index, and converting aspect.
        _load_surge_debris()
            Loads the surge glacier and debris-cover ratio datasets from their respective CSV files.
        _trim_df(reg_df)
            Selects and renames relevant columns from the input DataFrame for further processing.
        _filter_surge(reg_df)
            Removes glaciers identified as surge-type from the input DataFrame.
        _add_dc(reg_df)
            Joins the debris-cover ratio data to the input DataFrame based on glacier IDs.
        _add_HI(reg_df)
            Computes the hypsometric index (HI) for each glacier and applies a transformation
            for values less than 1.
        _conv_aspect(reg_df)
            Converts the 'Aspect' column from degrees to radians and adds sine and cosine
            representations as new columns.
    """

    def __init__(self, region_dict, save = False):
        self.dict = region_dict
        self.surge_path = "./data/other/surge_glaciers.csv"
        self.dc_path = "./data/other/dc_ratio.csv"
        self.save = save
        self._load_surge_debris()
        self.iter_regs()
        
    def iter_regs(self):

        for reg in self.dict.keys():
            c_df = self._build_ds_from_df(self.dict[reg])
            if self.save: c_df.to_csv(f"./data/MODEL_DATASETS/{reg}_dataset.csv", index = False)

    def _build_ds_from_df(self, c_df):
        c_df = self._trim_df(c_df)
        c_df = self._filter_surge(c_df)
        c_df = self._add_dc(c_df)
        c_df = self._add_HI(c_df)
        c_df = self._conv_aspect(c_df)
        return c_df

    def _load_surge_debris(self):
        self.surge = pd.read_csv(self.surge_path)
        self.dc = pd.read_csv(self.dc_path)

    def _trim_df(self, reg_df):
        cols = ['RGIId', 'GLIMSId','CenLon', 'CenLat', 'Area', 'Zmin', 'Zmax', 'Zmed', 'Slope', 'Aspect', 'dhdt_ma', 
        'dhdt_ma_si', 'mb_mwea', 'mb_mwea_si', 'RGIId_x']
        new_df = reg_df[cols].copy()
        new_df.rename(columns = {'RGIId_x': 'RGIId_Full'}, inplace = True)
        return new_df
        
    def _filter_surge(self, reg_df):
        reg_ids = reg_df['GLIMSId']
        surge_ids = self.surge['Glac_ID']
        overlap_ids = set(reg_ids).intersection(surge_ids)

        if len(overlap_ids) == 0:
            return reg_df
        else:
            reg_df_no_overlap = reg_df[~reg_df['GLIMSId'].isin(overlap_ids)]
            return reg_df_no_overlap


    def _add_dc(self, reg_df):
        new_df = reg_df.set_index('RGIId_Full').join(self.dc.set_index('RGIId')['dc_ratio'], lsuffix = '')
        new_df = new_df.reset_index()
        return new_df

    def _add_HI(self, reg_df):
        reg_df['HI'] = (reg_df['Zmax'] - reg_df['Zmed']) / (reg_df['Zmed'] - reg_df['Zmin'])
        reg_df['HI'] = reg_df['HI'].apply(lambda x: -1 / x if x <1 else x)
        return reg_df

    def _conv_aspect(self, reg_df):
        aspect = reg_df['Aspect']
        aspect2 = np.deg2rad(reg_df['Aspect'])
        reg_df['sin_Aspect'], reg_df['cos_Aspect'] = np.sin(aspect2), np.cos(aspect2)
        return reg_df
        