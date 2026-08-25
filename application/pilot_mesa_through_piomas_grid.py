"""
pilot_mesa_through_piomas_grid.py

Cheap pilot test for the "train on MESACLIP instead of/with FOSI" proposal.
Question: is the texture MESA/FOSI have that real PIOMAS lacks (Figure 14)
explained *simply* by PIOMAS's native grid being coarser than MESA/FOSI's
(~22km vs. ~10km POP_tx0.1v2), or does real PIOMAS lose additional detail
beyond that (e.g. from assimilation smoothing) -- which would mean training
on MESA regridded through PIOMAS's own native grid wouldn't actually expose
the network to realistically PIOMAS-like input, even though it would still
be a resolution downgrade from MESA's own native grid.

Method: for one representative RCP8.5 member (.004) and 6 sample dates
(the same ones used in Figure 14), take MESA's *native* (un-regridded,
~10km POP_tx0.1v2) hi_d and pass it through TWO conservative regrids:

  MESA native (~10km) --conservative--> PIOMAS native T-grid (~22km,
      real corners from grid.dat.pop) --conservative--> common 1deg grid

vs. MESA's own existing direct regrid (native -> 1deg in one step, already
built as X_MESA_HR_RCP85_daily_avg.nc) and vs. real PIOMAS's own v3
conservative-regridded 1deg field (X_PIOMAS_obs_2020_daily_v3_avg.nc) for
the same calendar dates.

If "MESA through PIOMAS's native grid" looks like real PIOMAS, the
resolution gap is just PIOMAS's native grid being coarse -- training on
MESA (even downsampled through PIOMAS's own grid) still exposes the network
to sharper detail than real PIOMAS ever has, which was the original worry.
If it still looks much sharper than real PIOMAS, PIOMAS itself is smoothing
away detail beyond its native grid resolution (e.g. assimilation), and no
amount of "downsample MESA to PIOMAS's grid" training will fully replicate
what real PIOMAS input looks like.

This is a feasibility pilot (1 member, 6 dates), not a training-data build.
"""

import warnings

import cftime
import numpy as np
import pop_tools
import xarray as xr
import xesmf as xe

warnings.filterwarnings("ignore", message="Latitude is outside of \\[-90, 90\\]")
warnings.filterwarnings("ignore", message="Input array is not C_CONTIGUOUS")

DATA_DIR = "/glade/derecho/scratch/skygale/Downscaling_Data"
WEIGHTED_GRIDS_DIR = "/glade/work/skygale/_projects/SeaIceDownscaling/weighted_grids"
OUT_FIG_DIR = "/glade/work/skygale/_projects/SeaIceDownscaling/Version5/results/PIOMAS_obs_2020"

MEMBER_DIR = "/glade/campaign/collections/gdex/data/d651009/b.e13.BRCP85C5.ne120_t12.cesm-ihesp-hires1.0.44-2006-2100.004"
MEMBER_ENSEMBLE_INDEX = 0  # .004 is ensemble index 0 in X_MESA_HR_RCP85_daily_avg.nc (sorted, .002/.003/.009 excluded)

DATE_STRS = ["2020-12-01", "2020-01-01", "2020-02-01", "2020-03-01", "2020-04-01", "2020-05-01"]

GRID_DAT_PATH = "/glade/campaign/cgd/ccr/yeager/OBS/seaice/PIOMAS/grid.dat"
GRID_DAT_POP_PATH = "/glade/derecho/scratch/skygale/PIOMAS_daily/utilities/grid.dat.pop"
NLAT_PIOMAS, NLON_PIOMAS = 120, 360

bbox_regrid = {"lon_min": -200, "lon_max": -130, "lat_min": 55, "lat_max": 85}
lon_min_regrid = bbox_regrid["lon_min"] % 360
lon_max_regrid = bbox_regrid["lon_max"] % 360
bbox = {"lon_min": -190, "lon_max": -140, "lat_min": 60, "lat_max": 80}
lon_min = bbox["lon_min"] % 360
lon_max = bbox["lon_max"] % 360

# ---------------------------------------------------------------------------
# 1. MESA native ice grid (POP_tx0.1v2), with real corners -- identical
#    construction to build_X_Y_from_MESA-HR_daily_rcp85.py
# ---------------------------------------------------------------------------

nat_ice_hr = pop_tools.get_grid("POP_tx0.1v2")
ice_lon = nat_ice_hr.TLONG % 360

mask_ice_hr = np.any(
    (nat_ice_hr.TLAT >= bbox_regrid["lat_min"]) & (nat_ice_hr.TLAT <= bbox_regrid["lat_max"])
    & (ice_lon >= lon_min_regrid) & (ice_lon <= lon_max_regrid),
    axis=1,
)

tlon_full_m = (nat_ice_hr.TLONG.values % 360)
tlat_full_m = nat_ice_hr.TLAT.values
ulon_full_m = (nat_ice_hr.ULONG.values % 360)
ulat_full_m = nat_ice_hr.ULAT.values
ny_full_m = tlon_full_m.shape[0]

rows_m = np.where(mask_ice_hr)[0]
r0_m, r1_m = rows_m.min(), rows_m.max()
if not np.array_equal(rows_m, np.arange(r0_m, r1_m + 1)):
    raise RuntimeError("MESA mask rows not contiguous.")

ulon_i_m = np.pad(ulon_full_m, ((0, 0), (1, 0)), mode="wrap")
ulat_i_m = np.pad(ulat_full_m, ((0, 0), (1, 0)), mode="wrap")
lon_b_mesa = ulon_i_m[r0_m - 1:r1_m + 1, :]
lat_b_mesa = ulat_i_m[r0_m - 1:r1_m + 1, :]

grid_mesa_conserv = xr.Dataset({
    "lat": (["nlat", "nlon"], tlat_full_m[r0_m:r1_m + 1, :]),
    "lon": (["nlat", "nlon"], tlon_full_m[r0_m:r1_m + 1, :]),
    "lat_b": (["nlat_b", "nlon_b"], lat_b_mesa),
    "lon_b": (["nlat_b", "nlon_b"], lon_b_mesa),
})
print("MESA native conservative source grid built.")

# ---------------------------------------------------------------------------
# 2. PIOMAS native T-grid, with real corners from grid.dat.pop -- identical
#    construction to build_X_Y_PIOMAS_obs_2020_daily_v3_conservative.py
# ---------------------------------------------------------------------------

grid_vals = np.loadtxt(GRID_DAT_PATH, dtype=np.float32).reshape(2, NLAT_PIOMAS, NLON_PIOMAS)
tlon_full_p = grid_vals[0] % 360
tlat_full_p = grid_vals[1]

pop_vals = np.loadtxt(GRID_DAT_POP_PATH).reshape(7, NLAT_PIOMAS, NLON_PIOMAS)
ulat_full_p = pop_vals[0].astype(np.float32)
ulon_full_p = pop_vals[1].astype(np.float32) % 360

mask_piomas = np.any(
    (tlat_full_p >= bbox_regrid["lat_min"]) & (tlat_full_p <= bbox_regrid["lat_max"])
    & (tlon_full_p >= lon_min_regrid) & (tlon_full_p <= lon_max_regrid),
    axis=1,
)
rows_p = np.where(mask_piomas)[0]
r0_p, r1_p = rows_p.min(), rows_p.max()
if not np.array_equal(rows_p, np.arange(r0_p, r1_p + 1)):
    raise RuntimeError("PIOMAS mask rows not contiguous.")

ulon_i_p = np.pad(ulon_full_p, ((0, 0), (1, 0)), mode="wrap")
ulat_i_p = np.pad(ulat_full_p, ((0, 0), (1, 0)), mode="wrap")
lon_b_piomas = ulon_i_p[r0_p - 1:r1_p + 1, :]
lat_b_piomas = ulat_i_p[r0_p - 1:r1_p + 1, :]

grid_piomas_conserv = xr.Dataset({
    "lat": (["nlat", "nlon"], tlat_full_p[mask_piomas]),
    "lon": (["nlat", "nlon"], tlon_full_p[mask_piomas]),
    "lat_b": (["nlat_b", "nlon_b"], lat_b_piomas),
    "lon_b": (["nlat_b", "nlon_b"], lon_b_piomas),
})
print("PIOMAS native conservative dest grid built.")

# ---------------------------------------------------------------------------
# 3. Common 1deg dest grid (identical to every other build in this project)
# ---------------------------------------------------------------------------

dst_1deg = xr.Dataset({
    "lat": ("lat", np.arange(bbox_regrid["lat_min"], bbox_regrid["lat_max"] + 1, 1.0)),
    "lon": ("lon", np.arange(lon_min_regrid, lon_max_regrid + 1, 1.0)),
})
dst_lat_c, dst_lon_c = dst_1deg.lat.values, dst_1deg.lon.values
dst_lat_b = np.concatenate([[dst_lat_c[0] - 0.5], dst_lat_c + 0.5])
dst_lon_b = np.concatenate([[dst_lon_c[0] - 0.5], dst_lon_c + 0.5])
dst_1deg_b = xr.Dataset({
    "lat": ("lat", dst_lat_c), "lon": ("lon", dst_lon_c),
    "lat_b": ("lat_b", dst_lat_b), "lon_b": ("lon_b", dst_lon_b),
})

# ---------------------------------------------------------------------------
# 4. Regridders: MESA-native -> PIOMAS-native, and PIOMAS-native -> 1deg
#    (the latter reuses the exact cached weights from the real PIOMAS build,
#    since it's the same source/dest grid pair)
# ---------------------------------------------------------------------------

regridder_mesa_to_piomasnative = xe.Regridder(
    grid_mesa_conserv, grid_piomas_conserv, method="conservative", periodic=True,
    filename=f"{WEIGHTED_GRIDS_DIR}/mesa_native_to_piomas_native_conservative.nc",
    reuse_weights=False,
)
print(" >> Built regridder_mesa_to_piomasnative.")

regridder_piomas_to_1deg = xe.Regridder(
    grid_piomas_conserv, dst_1deg_b, method="conservative", periodic=True,
    filename=f"{WEIGHTED_GRIDS_DIR}/piomas_to_1deg_conservative.nc", reuse_weights=True,
)
print(" >> Loaded cached regridder_piomas_to_1deg (same weights as the real PIOMAS build).")

# ---------------------------------------------------------------------------
# 5. Load MESA's native hi_d for the 6 sample dates (member .004)
# ---------------------------------------------------------------------------

import glob
ice_file = sorted(glob.glob(f"{MEMBER_DIR}/ice/proc/tseries/day_1/*.hi_d.20160102-20260101.nc"))
if not ice_file:
    raise RuntimeError("Expected hi_d chunk covering 2016-2026 not found.")
ds_ice = xr.open_dataset(ice_file[0])
print(f"Opened {ice_file[0]}, time range {ds_ice.time.values[0]} to {ds_ice.time.values[-1]}")

target_dates = [cftime.DatetimeNoLeap(int(d[:4]), int(d[5:7]), int(d[8:10])) for d in DATE_STRS]
hi_native_dates = ds_ice["hi_d"].sel(time=target_dates, method="nearest")
hi_native_dates = hi_native_dates.rename({"nj": "nlat", "ni": "nlon"}).isel(nlat=mask_ice_hr)
hi_native_dates = hi_native_dates.load()
ds_ice.close()
print("MESA native hi_d loaded for sample dates:", hi_native_dates.shape)

# ---------------------------------------------------------------------------
# 6. Chain: MESA native -> PIOMAS native -> 1deg
# ---------------------------------------------------------------------------

mesa_on_piomasnative = regridder_mesa_to_piomasnative(hi_native_dates, skipna=True)
mesa_through_piomas_1deg = regridder_piomas_to_1deg(mesa_on_piomasnative, skipna=True)
mesa_through_piomas_1deg = mesa_through_piomas_1deg.sel(
    lat=slice(bbox["lat_min"], bbox["lat_max"]), lon=slice(lon_min, lon_max)
).fillna(0).astype(np.float32)
print("MESA-through-PIOMAS-grid, 1deg:", mesa_through_piomas_1deg.shape)

# ---------------------------------------------------------------------------
# 7. Comparison fields: MESA's own direct 1deg regrid, and real PIOMAS's own
#    1deg conservative field, same dates
# ---------------------------------------------------------------------------

X_mesa_direct = xr.open_dataset(f"{DATA_DIR}/X_MESA_HR_RCP85_daily_avg.nc")["X"].sel(channel="hi_d").isel(ensemble=MEMBER_ENSEMBLE_INDEX)
mesa_direct_dates = X_mesa_direct.sel(time=target_dates, method="nearest")

X_piomas_real = xr.open_dataset(f"{DATA_DIR}/X_PIOMAS_obs_2020_daily_v3_avg.nc")["X"].sel(channel="hi_d")
if "ensemble" in X_piomas_real.dims:
    X_piomas_real = X_piomas_real.isel(ensemble=0)
piomas_real_dates = X_piomas_real.sel(time=target_dates, method="nearest")

# ---------------------------------------------------------------------------
# 8. Quantitative texture check: spatial std (coastal band roughness proxy)
#    at each stage, per date
# ---------------------------------------------------------------------------

print("\n=== Domain spatial std (m), per date -- includes large-scale gradient, NOT texture-isolated ===")
print(f"{'date':<12}{'MESA direct':>14}{'MESA/PIOMAS-grid':>18}{'PIOMAS real':>14}")
for i, d in enumerate(DATE_STRS):
    s_direct = float(mesa_direct_dates.isel(time=i).std())
    s_through = float(mesa_through_piomas_1deg.isel(time=i).std())
    s_real = float(piomas_real_dates.isel(time=i).std())
    print(f"{d:<12}{s_direct:>14.3f}{s_through:>18.3f}{s_real:>14.3f}")

mean_direct = float(mesa_direct_dates.std(dim=["lat", "lon"]).mean())
mean_through = float(mesa_through_piomas_1deg.std(dim=["lat", "lon"]).mean())
mean_real = float(piomas_real_dates.std(dim=["lat", "lon"]).mean())
print(f"\nMean spatial std across dates: MESA-direct={mean_direct:.3f}, "
      f"MESA-through-PIOMAS-grid={mean_through:.3f}, PIOMAS-real={mean_real:.3f}")
print("NOTE: plain std conflates domain-scale gradient magnitude (e.g. overall thickness range "
      "across the domain) with fine-scale texture -- real PIOMAS 2020 and this MESA RCP8.5 "
      "member's simulated '2020' are different physical realizations (a free-running scenario "
      "isn't constrained to match the real world at any given date), so a higher plain std for "
      "PIOMAS-real here is NOT by itself evidence of more fine-scale texture. See the high-pass "
      "metric below, which isolates small-scale roughness from the large-scale gradient.")

# ---------------------------------------------------------------------------
# 8b. Texture-isolated roughness: high-pass each field (subtract a 3x3
#     uniform-filtered version) before taking std, so large-scale gradients
#     (e.g. overall thickness trend across the domain) don't dominate the
#     comparison -- isolates the fine-scale "texture" the visual comparison
#     (Figure 14) was actually about.
# ---------------------------------------------------------------------------

from scipy.ndimage import uniform_filter


def highpass_std(da):
    vals = da.values
    smoothed = uniform_filter(vals, size=3, mode="nearest")
    residual = vals - smoothed
    return float(np.std(residual))


print("\n=== High-pass (texture-only) std (m), per date ===")
print(f"{'date':<12}{'MESA direct':>14}{'MESA/PIOMAS-grid':>18}{'PIOMAS real':>14}")
hp_direct_list, hp_through_list, hp_real_list = [], [], []
for i, d in enumerate(DATE_STRS):
    hp_direct = highpass_std(mesa_direct_dates.isel(time=i))
    hp_through = highpass_std(mesa_through_piomas_1deg.isel(time=i))
    hp_real = highpass_std(piomas_real_dates.isel(time=i))
    hp_direct_list.append(hp_direct)
    hp_through_list.append(hp_through)
    hp_real_list.append(hp_real)
    print(f"{d:<12}{hp_direct:>14.3f}{hp_through:>18.3f}{hp_real:>14.3f}")

hp_mean_direct = float(np.mean(hp_direct_list))
hp_mean_through = float(np.mean(hp_through_list))
hp_mean_real = float(np.mean(hp_real_list))
print(f"\nMean high-pass std across dates: MESA-direct={hp_mean_direct:.4f}, "
      f"MESA-through-PIOMAS-grid={hp_mean_through:.4f}, PIOMAS-real={hp_mean_real:.4f}")
print(f"Texture lost by PIOMAS-grid downsampling alone (MESA-direct -> MESA/PIOMAS-grid): "
      f"{1 - hp_mean_through / hp_mean_direct:.1%}")
print(f"Texture PIOMAS-real has beyond/below MESA-through-PIOMAS-grid: "
      f"{(hp_mean_real / hp_mean_through - 1):+.1%}")

# ---------------------------------------------------------------------------
# 9. Figure: 3-row panel, 6 columns (dates), same layout style as Figure 14
# ---------------------------------------------------------------------------

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, axes = plt.subplots(3, 6, figsize=(22, 10), constrained_layout=True)
row_data = [
    ("MESA native\n(direct -> 1deg)", mesa_direct_dates.squeeze()),
    ("MESA native\n(through PIOMAS grid)", mesa_through_piomas_1deg.squeeze()),
    ("PIOMAS real\n(conservative, v3)", piomas_real_dates.squeeze()),
]
vmax = 4.0
for row_i, (label, da) in enumerate(row_data):
    for col_i, d in enumerate(DATE_STRS):
        ax = axes[row_i, col_i]
        im = ax.pcolormesh(da.lon, da.lat, da.isel(time=col_i).values, vmin=0, vmax=vmax, cmap="viridis")
        if row_i == 0:
            ax.set_title(d, fontsize=10)
        if col_i == 0:
            ax.set_ylabel(label, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
fig.colorbar(im, ax=axes, shrink=0.6, label="SIT (m)")
fig.suptitle("Pilot: does PIOMAS's native grid alone explain the MESA/PIOMAS texture gap?\n"
             "(member .004, 6 sample dates -- same dates as Figure 14)", fontsize=12)
fig_path = f"{OUT_FIG_DIR}/15_pilot_mesa_through_piomas_grid.png"
fig.savefig(fig_path, dpi=150)
print(f"\nSaved figure to: {fig_path}")

print("\nDone.")
