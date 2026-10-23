import os
import glob 
import json
import pandas as pd
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

import torch
from torchvision import transforms


def get_YYYYmmdd(timestamp):
    return pd.Timestamp(str(timestamp)).strftime('%Y-%m-%d')

def load_json_as_dict(json_path):
    assert os.path.exists(json_path)
    with open(json_path, "r") as f:
        d = json.load(f)
    return d

def get_timestamp(metadata_json_path):
    return load_json_as_dict(metadata_json_path)["properties"]["acquired"] #"YYYY-mm-ddTHH:MM:SS"

def replicate_numerical_metadata(input_dict, gsd_scaling=1., base_year=1980, base_lon=180, base_lat=90):
    
    metadata = input_dict # python dictionary with the json content
    coords = np.hstack([np.array(x) for x in metadata["geometry"]["coordinates"]]).squeeze()
    assert len(coords.shape) == 2 and coords.shape[-1] == 2
    lons, lats = coords[:, 0], coords[:, 1]

    lon = lons.mean()
    lat = lats.mean()
    cloud_cover = metadata["properties"]["cloud_cover"]
    base_gsd = metadata["properties"]["pixel_resolution"] # probably better to use the pixel resolution here, more stable
    # https://gis.stackexchange.com/questions/397523/what-is-the-difference-between-ground-sampling-distance-gsd-and-spatial-resolu
    gsd = base_gsd * gsd_scaling
    
    year = int(metadata["properties"]["acquired"].split("-")[0])
    month = int(metadata["properties"]["acquired"].split("-")[1])
    day = int(metadata["properties"]["acquired"].split("-")[-1][:2])

    # apply offsets. I am not sure why they are used, but they are needed for compatibility with metadata_normalize
    year = year - base_year
    lon = lon + base_lon
    lat = lat + base_lat
    
    return torch.tensor([lon, lat, gsd, cloud_cover, year, month, day])

def metadata_normalize(metadata, base_lon=180, base_lat=90, base_year=1980, max_gsd=1., scale=1000):
    # copy from diffusionsat.data_util.metadata_normalize, only to remove dependency
    lon, lat, gsd, cloud_cover, year, month, day = metadata
    lon = lon / (180 + base_lon) * scale
    lat = lat / (90 + base_lat) * scale
    gsd = gsd / max_gsd #*scale
    cloud_cover = cloud_cover * scale
    year = year / (2100 - base_year) * scale
    month = month / 12 * scale
    day = day / 31 * scale
    return torch.tensor([lon, lat, gsd, cloud_cover, year, month, day])

def get_valid_px_vals(px_vals, ):
    # this function ensures that input images are usable by the network
    # the image size has to be a multiple of 8 and must not contain nan values

    # px_vals is an array with shape (..., h, w)
    h, w = px_vals.shape[-2:]
    H = (h//8) * 8 # closest multiple of 8 that is not larger than the original height
    W = (w//8) * 8
    px_vals = px_vals[..., :H, :W]
    #px_vals[torch.isnan(px_vals)] = -1
    return px_vals
    

########################### WEATHER ################################
class GencatWeatherMeasures:
    def __init__(self, weather_data_path, start_day=None, end_day=None, method="lastN"):
        """
        Retrieve the weather measures corresponding to a given acquisition date
        
        Parameters
        ----------
        csv_path : str
            Path to the CSV file with weather data.
        start_day : str or datetime
            Start of the time window.
        end_day : str or datetime
            End of the time window.
        smooth_factor : float
            Smoothing parameter for the spline.
        method : str
            Method used for spline fitting.
        """
        assert os.path.exists(weather_data_path)
        self.indicators = ["temperatura", "humitat_relativa", "precipitacions", "irradiancia_solar"]
        #self.indicators = ["temperatura", "humitat_relativa", "irradiancia_solar"]
        self.csv_paths = sorted(glob.glob(weather_data_path + "/*.xls"))
        self._set_default_first_and_last_dates()
        # if there is an input start_day and end_day, check they are within the default timespan and update
        if (start_day is not None) and (end_day is not None):
            start_day = pd.Timestamp(str(start_day)).replace(tzinfo=None) # ensure date format is ok
            end_day = pd.Timestamp(str(end_day)).replace(tzinfo=None)
            diff_start = compute_day_diffs_between_timestamps([self.start_day, start_day])[1]
            diff_end = compute_day_diffs_between_timestamps([end_day, self.end_day])[1]
            assert diff_start >= 0, "error: input start_day is before the first date of the weather data"
            assert diff_end >= 0, "error: input end_day is after the last date of the weather data"
            self.start_day = start_day.strftime('%Y-%m-%d')
            self.end_day = end_day.strftime('%Y-%m-%d')

        self.method = method
        
        max_day_diff = compute_day_diffs_between_timestamps([self.start_day, self.end_day])[-1]
        self.day_diffs = list(np.arange(max_day_diff + 1))

        self.functions = {}
        self.raw_functions = {} 
        self._build_functions()


    def _set_default_first_and_last_dates(self):
        # get the unfiltered dataframe gencat_df and find the first and last dates
        gencat_df = get_metadata_timeseries_Gencat(
                self.csv_paths, "temperatura", start_day=None, end_day=None, return_dict=False 
            )
        df_first_idx = 0
        df_last_idx = len(gencat_df["Data"])-1 # for some reason -1 is not valid
        self.start_day = gencat_df["Data"][df_first_idx].strftime('%Y-%m-%d')
        self.end_day = gencat_df["Data"][df_last_idx].strftime('%Y-%m-%d')

    def _build_functions(self):
        for measure in self.indicators:
            gencat_d = get_metadata_timeseries_Gencat(
                self.csv_paths, measure, self.start_day, self.end_day, return_dict=True
            )
            # gentcat_d is a dictionary that contains per-day statistics
            # 'min', 'max', 'mean', 'median', 'count', 'timestamps'
            self.raw_functions[measure] = gencat_d.copy()
            # create a smoother version of the meteo stats
            for k in ["min", "max", "mean", "median", "sum"]:
                x, timestamps = gencat_d[k], gencat_d["timestamps"]
                #mu, sigma = x.mean(), x.std()
                #x = (x - mu) / (sigma + 1e-7)
                _, stat_F = fit_smoothed_spline(x, timestamps, N=10, method=self.method)
                #_, stat_F = fit_smoothed_spline(x, timestamps,, alpha=0.08, method="ema")
                #_, stat_F = fit_smoothed_spline(x, timestamps,, s=5e3, method="spline")
                gencat_d[k] = stat_F   #+ mu
            self.functions[measure] = gencat_d

    def get_weather_at_input_date(self, input_date, measures=None):
        """
        Retrieve the fitted spline function for a given measure.
        """
        #input date format must be a string -> 'YYYY-MM-DD'
        #measures is either None (retrieves all possible measures) or a list of measures (e.g. ["temperature",])
        #day_diff = input_date - STARTING_DATE
        if measures is None:
            measures = self.indicators
        else:
            for m in measures:
                assert m in self.indicators, f"error: unknown weather measure {m}"
        day_diff = compute_day_diffs_between_timestamps([self.start_day, input_date])[-1]
        max_day_diff = compute_day_diffs_between_timestamps([self.start_day, self.end_day])[-1]
        if not 0 <= day_diff <= max_day_diff:
            raise ValueError(
                f"Weather date {input_date} is outside "
                f"[{self.start_day}, {self.end_day}]"
            )
        vals = [
            self.functions[k]["sum" if k == "precipitacions" else "mean"][day_diff]
            for k in measures
        ]
        return dict(zip(measures, vals))
    
    def is_input_date_valid(self, input_date):
        input_day_diff = compute_day_diffs_between_timestamps([self.start_day, input_date])[-1]
        max_day_diff = compute_day_diffs_between_timestamps([self.start_day, self.end_day])[-1]
        return 0 <= input_day_diff <= max_day_diff

def get_metadata_timeseries_Gencat(excel_path, variable, start_day=None, end_day=None, return_dict=False):
    """
    Inputs:
        csv_path : string pointing to the gencat weather csv file
        variable : string, can be "temperatura", "humitat_relativa", "precipitacions" or "irradiancia_solar"
        start_day: string with the first date of interest in %Y-%m-%d format, e.g.: '2021-03-27'
        end_day:   string with the last date of interest in %Y-%m-%d format, e.g.: '2024-04-24'
    """

    if isinstance(excel_path, str):
        excel_paths = [excel_path]
        assert os.path.exists(excel_path)
    else:
        # excel_path is a list of paths, multiple files will be aggregated
        for p in excel_path:
            assert os.path.exists(p)
        excel_paths = excel_path
    assert variable in ["temperatura", "humitat_relativa", "precipitacions", "irradiancia_solar"]
    CODIS_VARIABLES = {"temperatura": "Temperatura",
                       "humitat_relativa": "Humitat relativa",
                       "precipitacions": "Precipitació",
                       "irradiancia_solar": "Irradiància solar global"}
    target_key = CODIS_VARIABLES[variable]

    # load csv
    df = pd.read_excel(excel_paths[0])
    for p in excel_paths[1:]:
        df_ = pd.read_excel(p)
        df = pd.concat([df, df_], ignore_index=True)
    filtered_df = df

    filtered_df['Data'] = pd.to_datetime(filtered_df['Data'], format='%d/%m/%Y') 
    # filter dates outside the interval [start_day, end_day]
    if (start_day is not None) and (end_day is not None):
        valid_dates = (filtered_df['Data'] >= pd.Timestamp(start_day)) & (filtered_df['Data'] <= pd.Timestamp(end_day))
        filtered_df = filtered_df[valid_dates]

    # sort according to the acquisition date
    filtered_df = filtered_df.sort_values('Data')

    # ignore non-numeric values like "--" (will be converted to NaN)
    filtered_df[target_key] = pd.to_numeric(filtered_df[target_key], errors="coerce")

    # extract the min, max, mean and median values per day
    grouped = filtered_df.groupby(filtered_df['Data'].dt.date)[target_key]
    stat_keys = ['min', 'max', 'mean', 'median', 'sum', 'count']
    daily_stats = grouped.agg(stat_keys).reset_index()

    # Functions are indexed by calendar-day offset. Reindexing makes that
    # relationship correct even when a station day is absent from the source.
    first_date = pd.Timestamp(start_day) if start_day is not None else filtered_df['Data'].min()
    last_date = pd.Timestamp(end_day) if end_day is not None else filtered_df['Data'].max()
    full_dates = pd.date_range(first_date, last_date, freq="D")
    daily_stats["Data"] = pd.to_datetime(daily_stats["Data"])
    daily_stats = (
        daily_stats.set_index("Data")
        .reindex(full_dates)
        .rename_axis("Data")
        .reset_index()
    )
    daily_stats["count"] = daily_stats["count"].fillna(0)

    if return_dict:
        stat_vals = np.array([daily_stats[k].values.tolist() for k in stat_keys])
        stat_dict = dict(zip(stat_keys, stat_vals))
        stat_dict["timestamps"] = [
            pd.Timestamp(v).strftime("%Y-%m-%dT00:00:00")
            for v in daily_stats["Data"]
        ]
        return stat_dict
    
    # (optional) export to excel
    # convert dates back to strings
    #daily_stats['Data'] = pd.to_datetime(daily_stats['Data']).dt.strftime("%Y-%m-%d")
    #df.to_excel("gencat_stats.xlsx", index=False)

    return daily_stats

def fit_smoothed_spline(values, timestamps, s=0.2, alpha=0.2, N=14, plot=False, label=None, method="spline"):
    from scipy.interpolate import UnivariateSpline

    day_diffs = compute_day_diffs_between_timestamps(timestamps)
    x = np.array(day_diffs) # temporal differences (days)
    y = np.asarray(values, dtype=float)
    if np.isnan(y).all():
        raise ValueError("Cannot smooth a weather series containing only NaN values")
    # Fill isolated missing station days before smoothing. In particular, this
    # prevents one NaN from poisoning all later values in cumulative operations.
    y = pd.Series(y).interpolate(limit_direction="both").to_numpy()

    if method=="ema":
        alpha = np.clip(alpha, 0, 1) # Smoothing factor -> smaller alpha implies more smoothing
        smoothed_y = np.zeros_like(y)
        smoothed_y[0] = y[0]
        for i in range(1, len(y)):
            smoothed_y[i] = alpha * y[i] + (1 - alpha) * smoothed_y[i - 1]
        smoothed_x = x
    elif method=="lastN":
        # N is the number of precedent samples taken into account, 14 = two weeks
        # for index i, we take the mean of y[max(0, i-N): i]
        n = len(y)
        c = np.cumsum(y)                      # c[i] = sum(y[:i+1])
        c_pad = np.concatenate(([0.0], c))    # so sums can start at 0 cleanly
        idx = np.arange(n)
        start = np.maximum(0, idx - N)
        sums = c_pad[idx] - c_pad[start]
        counts = idx - start              # 1..10
        counts[0] = 1 # no past data for idx=0
        smoothed_y = sums / counts
        smoothed_y[0] = y[0]
        smoothed_x = x
    elif method=="spline":
        spline = UnivariateSpline(x, y, s=s, k=3)  # s controls smoothness (0 is no smoothing)
        # suitable values for s explained below
        # https://stackoverflow.com/questions/8719754/scipy-interpolate-univariatespline-not-smoothing-regardless-of-parameters
        smoothed_x = np.arange(max(x) + 1)
        smoothed_y = spline(smoothed_x)
    else:
        raise ValueError(f"Unknown method value {method}")

    if plot:
        fig = plt.figure(figsize=(15,5))
        # plot original values
        plt.plot(day_diffs, values, 'o-', label=label)
        xticks = np.linspace(0, len(timestamps) - 1, num=20, dtype=int)
        plt.xticks(np.array(day_diffs)[xticks], [get_YYYYmmdd(timestamps[i]) for i in xticks], rotation=45)
        plt.xlabel("acquisition date")
        plt.ylabel(label)
        plt.legend(loc="upper right")
        #plot smoothed spline
        plt.plot(smoothed_x, smoothed_y, label=f'{label} smoothed', linewidth=2)
        if label is not None:
            plt.legend()
        plt.show()

    return smoothed_x, smoothed_y

def compute_day_diffs_between_timestamps(timestamps):
    # timestamps is a list of timestamps
    # this function computes the difference in days between each timestamp and the first timestamp of the list
    first_day = pd.Timestamp(str(timestamps[0])).replace(tzinfo=None) # remove timezone awareness
    day_diffs = [(pd.Timestamp(str(current_day)).replace(tzinfo=None) - first_day).days for current_day in timestamps]
    return day_diffs

def replicate_weather_metadata(input_dict, normalize=True, regularize=False, random=False):
    
    metadata = input_dict # python dictionary extracted with get_weather_at_input_date
    vec = torch.Tensor([*metadata.values()]) # vals for ["temperatura", "humitat_relativa", "precipitacions", "irradiancia_solar"]
    if normalize:
        return normalize_weather_vals(vec, regularize=regularize)
    if random:
        return torch.empty_like(vec).uniform_(0, 1000)
    return vec

def normalize_weather_vals(vec, scale=1.0, regularize=False):
    # Weather variables numerical range for Catalonia
    # Temperature: °C — typically -5 to 40; rare extremes about -20 to >45.
    # Relative Humidity: % — 0 to 100 (0 very dry, 100 saturation/fog/rain).
    # Precipitation: mm (== L/m²) — typically 0–20 mm/h; intense storms up to ~100+ mm/h.
    # Daily totals usually 0–50 mm; severe events can exceed ~200 mm/day.
    # Global Solar Irradiance: W/m² — 0 at night to ~800 W/m² at clear-sky midday (summer peak).
    mins = torch.tensor([-20., 0., 0., 0.])
    maxs = torch.tensor([50., 100., 50., 1000.])
    #mins = torch.tensor([-10, 0., 0.])
    #maxs = torch.tensor([50., 200., 800.])
    normalized = (vec - mins) / (maxs - mins)
    normalized = torch.clamp(normalized, 0.0, 1.0) # ensure vec is between 0 and 1, outlier extrema is clipped
    normalized = normalized * scale

    if regularize:
        # add some small noise to distort the actual values
        noise = torch.empty_like(normalized).uniform_(-0.02, 0.02) * scale
        normalized = normalized + noise

    return normalized
