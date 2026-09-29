from json import dumps
from glob import glob
from os.path import exists

metadata_table = [
    ("A1", "T2N0M0", "1", "IIA"),
    ("A2", "T2N0M0", "1", "IIA"),
    ("A3", "T2N1M0", "2", "IIB"),
    ("A4", "T2N0M0", "1", "IIA"),
    ("A5", "T2N1M0", "2", "IIB"),
    ("A6", "T2N2M0", "2", "IIIA"),
    ("A7", "T4N0M0", "2", "IIIB"),
    ("A8", "T2N0M0", "2", "IIA"),
    ("A9", "T3N0M0", "2", "IIB"),
    ("A10", "T2N1M0", "2", "IIB"),
    ("B1", "T3N2M0", "2", "IIIA"),
    ("B2", "T2N0M0", "2", "IIA"),
    ("B3", "T3N0M0", "2", "IIB"),
    ("B4", "T2N0M0", "2", "IIA"),
    ("B5", "T2N0M0", "2", "IIA"),
    ("B6", "T2N0M0", "2", "IIA"),
    ("B7", "T2N2M0", "2", "IIIA"),
    ("B8", "T2N0M0", "*", "IIA"),
    ("B9", "T2N1M0", "2", "IIB"),
    ("B10", "T2N0M0", "2", "IIA"),
    ("C1", "T2N0M0", "2", "IIA"),
    ("C2", "T3N1M0", "2", "IIIA"),
    ("C3", "T2N0M0", "2", "IIA"),
    ("C4", "T2N0M0", "2", "IIA"),
    ("C5", "T2N0M0", "2", "IIA"),
    ("C6", "T2N2M0", "2", "IIIA"),
    ("C7", "T2N0M0", "2", "IIA"),
    ("C8", "T3N1M0", "2", "IIIA"),
    ("C9", "T3N0M0", "2", "IIB"),
    ("C10", "T2N1M0", "2", "IIB"),
    ("D1", "T2N0M0", "2", "IIA"),
    ("D2", "T2N2M0", "2", "IIIA"),
    ("D3", "T2N0M0", "2", "IIA"),
    ("D4", "T2N0M0", "2", "IIA"),
    ("D5", "T2N0M0", "2", "IIA"),
    ("D6", "T2N0M0", "2", "IIA"),
    ("D7", "T2N1M0", "2", "IIB"),
    ("D8", "T2N0M0", "2", "IIA"),
    ("D9", "T2N0M0", "2", "IIA"),
    ("D10", "T2N0M0", "2", "IIA"),
    ("E1", "T2N1M0", "2", "IIB"),
    ("E2", "T2N0M0", "2", "IIA"),
    ("E3", "T2N0M0", "2", "IIA"),
    ("E4", "T2N2aM0", "2", "IIIA"),
    ("E5", "T2N0M0", "2", "IIA"),
    ("E6", "T2N0M0", "2", "IIA"),
    ("E7", "T2N0M0", "2", "IIA"),
    ("E8", "T2N2M0", "2", "IIIA"),
    ("E9", "T2N0M0", "2", "IIA"),
    ("E10", "T3N0M0", "2", "IIB"),
    ("F1", "T1N0M0", "2", "IA"),
    ("F2", "T2N0M0", "2", "IIA"),
    ("F3", "T2N1M0", "2", "IIB"),
    ("F4", "T4N0M0", "2", "IIIB"),
    ("F5", "T4N1M0", "2", "IIIB"),
    ("F6", "T2N2M0", "2", "IIIA"),
    ("F7", "T4N0M0", "2", "IIIB"),
    ("F8", "T2N2M0", "2", "IIIA"),
    ("F9", "T2N1M0", "2", "IIB"),
    ("F10", "T2N2M0", "2", "IIIA"),
    ("G1", "T2N1M0", "2", "IIB"),
    ("G2", "T2N2M0", "2", "IIIA"),
    ("G3", "T2N0M0", "2", "IIA"),
    ("G4", "T3N1M0", "-", "IIIA"),
    ("G5", "T3N0M0", "2", "IIB"),
    ("G6", "T2N0M0", "3", "IIA"),
    ("G7", "T1N1M0", "3", "IIA"),
    ("G8", "T2N3M0", "3", "IIIC"),
    ("G9", "T2N0M0", "3", "IIA"),
    ("G10", "T2N1M0", "3", "IIB"),
    ("H1", "T2N0M0", "2", "IIA"),
    ("H2", "T2N1M0", "3", "IIB"),
    ("H3", "T2N0M0", "3", "IIA"),
    ("H4", "T2N0M0", "3", "IIA"),
    ("H5", "T2N2M0", "3", "IIIA"),
    ("H6", "T2N0M0", "3", "IIA"),
    ("H7", "T3N0M0", "3", "IIB"),
    ("H8", "T4N0M0", "3", "IIIB"),
    ("H9", "T2N0M0", "2", "IIA"),
    ("H10", "T3N0M0", "3", "IIB"),
    ("I1", "T2N0M0", "3", "IIA"),
    ("I2", "T2N0M0", "3", "IIA"),
    ("I3", "T3N1M0", "*", "IIIA"),
    ("I4", "T2N0M0", "3", "IIA"),
    ("I5", "T3N2M0", "3", "IIIA"),
    ("I6", "T2N2aM0", "3", "IIIA"),
    ("I7", "T2N0M0", "3", "IIA"),
    ("I8", "T2N0M0", "3", "IIA"),
    ("I9", "T2N0M0", "3", "IIA"),
    ("I10", "T2N0M0", "3", "IIA"),
    ("J1", "T2N0M0", "3", "IIA"),
    ("J2", "T4N0M0", "3", "IIIB"),
    ("J3", "T2N0M0", "3", "IIA"),
    ("J4", "T3N0M0", "3", "IIB"),
    ("J5", "T2N0M0", "3", "IIA"),
    ("J6", "T4N0M0", "3", "IIIB"),
    ("J7", "T3N0M0", "3", "IIB"),
    ("J8", "T2N0M0", "3", "IIA"),
    ("J9", "T3N1M0", "3", "IIIA"),
    ("J10", "T2N1M0", "3", "IIB"),
]


def load_metadata(base_path) -> dict:
    from json import load

    data = None
    with open(base_path + "metadata.json", "r") as fhandle:
        data = load(fhandle)
    return data


def save_metadata(base_path, data):
    from json import dumps

    with open(base_path + "metadata.json", "w") as fhandle:
        fhandle.write(dumps(data, indent=2))


for table_entry in metadata_table:
    location = f"../Datasets/HT-{table_entry[0]}/"
    try:
        meta = load_metadata(location)
        meta["TNM"] = table_entry[1]
        meta["grade"] = table_entry[2]
        try:
            meta["grade_i"] = int(table_entry[2]) - 1
        except:
            meta["grade_i"] = -1
        try:
            del meta["tn_ohe"]
        except:
            pass
        try:
            del meta["stage_basic_ohe"]
        except:
            pass
        try:
            del meta["stage_advanced_ohe"]
        except:
            pass
        save_metadata(location, meta)
    except:
        pass
