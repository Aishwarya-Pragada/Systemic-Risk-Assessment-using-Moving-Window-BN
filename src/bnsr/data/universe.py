"""The 60 HSI constituent stocks and their sectors (paper, S1 Appendix, Table 4).

The raw constituents file contains 103 tickers (every stock that was in the HSI at some point
between 2006 and 2021). The paper builds its networks only on the 60 stocks listed in S1 Appendix;
the node count grows from 46 to 60 because 14 of them were listed after 2 Jan 2008.
"""
import pandas as pd

# (constituent name, stock symbol, sector)
UNIVERSE = [
    # ---- Commerce (29) ----
    ("CKH HOLDINGS", "0001.HK", "Commerce"),
    ("GALAXY ENT", "0027.HK", "Commerce"),
    ("MTR CORPORATION", "0066.HK", "Commerce"),
    ("GEELY AUTO", "0175.HK", "Commerce"),
    ("ALI HEALTH", "0241.HK", "Commerce"),
    ("CITIC", "0267.HK", "Commerce"),
    ("WH GROUP", "0288.HK", "Commerce"),
    ("SINOPEC CORP", "0386.HK", "Commerce"),
    ("TECHTRONIC IND", "0669.HK", "Commerce"),
    ("CHINA UNICOM", "0762.HK", "Commerce"),
    ("PETROCHINA", "0857.HK", "Commerce"),
    ("XINYI GLASS", "0868.HK", "Commerce"),
    ("CNOOC", "0883.HK", "Commerce"),
    ("CHINA MOBILE", "0941.HK", "Commerce"),
    ("XINYI SOLAR", "0968.HK", "Commerce"),
    ("HENGAN INT'L", "1044.HK", "Commerce"),
    ("CSPC PHARMA", "1093.HK", "Commerce"),
    ("SINO BIOPHARM", "1177.HK", "Commerce"),
    ("BYD COMPANY", "1211.HK", "Commerce"),
    ("BUD APAC", "1876.HK", "Commerce"),
    ("SANDS CHINA LTD", "1928.HK", "Commerce"),
    ("AAC TECH", "2018.HK", "Commerce"),
    ("ANTA SPORTS", "2020.HK", "Commerce"),
    ("WUXI BIO", "2269.HK", "Commerce"),
    ("SHENZHOU INTL", "2313.HK", "Commerce"),
    ("MENGNIU DAIRY", "2319.HK", "Commerce"),
    ("LI NING", "2331.HK", "Commerce"),
    ("SUNNY OPTICAL", "2382.HK", "Commerce"),
    ("HAIDILAO", "6862.HK", "Commerce"),
    # ---- Finance (11) ----
    ("HSBC HOLDINGS", "0005.HK", "Finance"),
    ("HANG SENG BANK", "0011.HK", "Finance"),
    ("HKEX", "0388.HK", "Finance"),
    ("CCB", "0939.HK", "Finance"),
    ("AIA", "1299.HK", "Finance"),
    ("ICBC", "1398.HK", "Finance"),
    ("PING AN", "2318.HK", "Finance"),
    ("BOC HONG KONG", "2388.HK", "Finance"),
    ("CHINA LIFE", "2628.HK", "Finance"),
    ("CM BANK", "3968.HK", "Finance"),
    ("BANK OF CHINA", "3988.HK", "Finance"),
    # ---- Information technology (4) ----
    ("TENCENT", "0700.HK", "Information technology"),
    ("XIAOMI-W", "1810.HK", "Information technology"),
    ("MEITUAN-W", "3690.HK", "Information technology"),
    ("BABA-SW", "9988.HK", "Information technology"),
    # ---- Properties (12) ----
    ("HENDERSON LAND", "0012.HK", "Properties"),
    ("SHK PPT", "0016.HK", "Properties"),
    ("NEW WORLD DEV", "0017.HK", "Properties"),
    ("HANG LUNG PPT", "0101.HK", "Properties"),
    ("CHINA OVERSEAS", "0688.HK", "Properties"),
    ("LINK REIT", "0823.HK", "Properties"),
    ("LONGFOR GROUP", "0960.HK", "Properties"),
    ("CHINA RES LAND", "1109.HK", "Properties"),
    ("CK ASSET", "1113.HK", "Properties"),
    ("WHARF REIC", "1997.HK", "Properties"),
    ("COUNTRY GARDEN", "2007.HK", "Properties"),
    ("CG SERVICES", "6098.HK", "Properties"),
    # ---- Utilities (4) ----
    ("CLP HOLDINGS", "0002.HK", "Utilities"),
    ("HK & CHINA GAS", "0003.HK", "Utilities"),
    ("POWER ASSETS", "0006.HK", "Utilities"),
    ("CKI HOLDINGS", "1038.HK", "Utilities"),
]

SECTOR_COUNTS = {
    "Commerce": 29,
    "Finance": 11,
    "Information technology": 4,
    "Properties": 12,
    "Utilities": 4,
}


def universe_frame() -> pd.DataFrame:
    return pd.DataFrame(UNIVERSE, columns=["name", "symbol", "sector"])


def symbols() -> list:
    return [s for _, s, _ in UNIVERSE]


def sector_map() -> dict:
    return {s: sec for _, s, sec in UNIVERSE}
