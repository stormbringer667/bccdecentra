# -*- coding: utf-8 -*-
import os, glob
import pandas as pd
from typing import Dict, Tuple

def load_clients(path="data/clients.csv") -> pd.DataFrame:
    df = pd.read_csv(path)
    return df

def _extract_id(p: str) -> int:
    base = os.path.basename(p)
    try:
        return int(base.split("_")[1])
    except Exception:
        return None

def load_client_tables(tx_glob="data/client_*_transactions_3m.csv",
                       tr_glob="data/client_*_transfers_3m.csv") -> Dict[int, Dict[str, pd.DataFrame]]:
    data = {}
    for p in glob.glob(tx_glob):
        cid = _extract_id(p)
        if cid is None: 
            continue
        df = pd.read_csv(p)
        data.setdefault(cid, {})["tx"] = df

    for p in glob.glob(tr_glob):
        cid = _extract_id(p)
        if cid is None:
            continue
        df = pd.read_csv(p)
        data.setdefault(cid, {})["tr"] = df
    return data
