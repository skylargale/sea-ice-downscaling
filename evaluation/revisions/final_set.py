"""
Which model is "the final model" for the revision scripts and figure builders (2026-11-14).

FINAL_SET=final1             the submitted paper's model: 4 consecutive training windows (EngressNet + EMA), seed 0
FINAL_SET=final2  (default)  the revised final model (user decision 2026-11-14): one EngressNet trained on 1825 random days
                             from 2000-2020 (seed 0), matched det-MSE on the same days, 10-model seed ensemble on the same days

Usage in a script:  from final_set import FS   ->  FS["MESA"]["eng"], FS["MESA"]["windows"], FS["piomas2010"], ...
Run e.g.:  FINAL_SET=final1 python evaluation/revisions/community_table4.py   (old model; default since 2026-11-15 is final2)
Output names depend on the model, not on the default: final1 outputs have no suffix, final2 outputs end in _final2.
"""
import os

SETS = {
    "final1": {
        "MESA": {"eng": "MESA_final_ema", "det": "MESA_detmse_ema", "windows": ["2000-2005", "2005-2010", "2010-2015", "2015-2020"],
                 "seedens": ["MESA_detmse_ema", "REV_E1_MESA_detmse_seed1", "REV_E1_MESA_detmse_seed2"]},
        "FOSI": {"eng": "FOSI_final_ema", "det": "FOSI_detmse_ema", "windows": ["2000-2005", "2005-2010", "2010-2015", "2015-2020"],
                 "seedens": ["FOSI_detmse_ema", "REV_E1_FOSI_detmse_seed1", "REV_E1_FOSI_detmse_seed2"]},
        "map_window": "2015-2020",                       # window used for single-run map figures (Figs 3, 5)
        "piomas2010": "PIOMAS_2010-2023_cdrv6_emaFOSI_4win", "piomas_window": "2015-2020",
        "cells_eng": "ema_piomas", "cells_det": "detmse_ema_piomas",
        "label": "4 consecutive training windows (2000-05 ... 2015-20), seed 0",
    },
    "final2": {
        "MESA": {"eng": "FINAL2_MESA_random1825", "det": "FINAL2_MESA_detmse_random1825", "windows": ["2000-2020"],
                 "seedens": ["FINAL2_MESA_detmse_random1825", "FINAL2_MESA_detmse_seedens"]},
        "FOSI": {"eng": "FINAL2_FOSI_random1825", "det": "FINAL2_FOSI_detmse_random1825", "windows": ["2000-2020"],
                 "seedens": ["FINAL2_FOSI_detmse_random1825", "FINAL2_FOSI_detmse_seedens"]},
        "map_window": "2000-2020",
        "piomas2010": "PIOMAS_2010-2023_cdrv6_FINAL2_FOSI", "piomas_window": "2000-2020",
        "cells_eng": "final2_piomas", "cells_det": "final2_detmse_piomas",
        "label": "one model trained on 1825 random days from 2000-2020, seed 0",
    },
}
NAME = os.environ.get("FINAL_SET", "final2")
FS = SETS[NAME]
# MESACLIP is named "MESA" in batch names; tables use "MESACLIP"
DS_NAMES = {"MESACLIP": "MESA", "FOSI": "FOSI"}
