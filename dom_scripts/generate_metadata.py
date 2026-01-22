#!/usr/bin/env python
import pandas as pd
import argparse
import os
# Manually filled out just for the HILIC data provided by Katherine
# Setup ID: 0000

# PLEASE TYPE OUT YOUR VALUES FOR YOUR SET UP. I KNOW IT"S PAINFUL BUT BEAR WITH US FOR NOW

def get_args():

    parser = argparse.ArgumentParser(description="Generate Metadata.tsv of Chromatography Setup")
    parser.add_argument('--id')
    parser.add_argument('-o', '--output_dir')
    
    return parser.parse_args()

# set global vars
args = get_args()

id = args.id
dir = args.output_dir

os.makedirs(dir, exist_ok=True)

# Column Headers
header = """id	column.name	column.usp.code	column.length	column.id	column.particle.size	column.temperature	column.flowrate	column.t0	eluent.A.h2o	eluent.A.meoh	eluent.A.acn	eluent.A.iproh	eluent.A.acetone	eluent.A.hex	eluent.A.chcl3	eluent.A.ch2cl2	eluent.A.hept	eluent.A.formic	eluent.A.formic.unit	eluent.A.acetic	eluent.A.acetic.unit	eluent.A.trifluoroacetic	eluent.A.trifluoroacetic.unit	eluent.A.phosphor	eluent.A.phosphor.unit	eluent.A.nh4ac	eluent.A.nh4ac.unit	eluent.A.nh4form	eluent.A.nh4form.unit	eluent.A.nh4carb	eluent.A.nh4carb.unit	eluent.A.nh4bicarb	eluent.A.nh4bicarb.unit	eluent.A.nh4f	eluent.A.nh4f.unit	eluent.A.nh4oh	eluent.A.nh4oh.unit	eluent.A.trieth	eluent.A.trieth.unit	eluent.A.triprop	eluent.A.triprop.unit	eluent.A.tribut	eluent.A.tribut.unit	eluent.A.nndimethylhex	eluent.A.nndimethylhex.unit	eluent.A.medronic	eluent.A.medronic.unit	eluent.A.pH	eluent.A.heptafluorobutyric	eluent.A.heptafluorobutyric.unit	eluent.B.h2o	eluent.B.meoh	eluent.B.acn	eluent.B.iproh	eluent.B.acetone	eluent.B.hex	eluent.B.chcl3	eluent.B.ch2cl2	eluent.B.hept	eluent.B.formic	eluent.B.formic.unit	eluent.B.acetic	eluent.B.acetic.unit	eluent.B.trifluoroacetic	eluent.B.trifluoroacetic.unit	eluent.B.phosphor	eluent.B.phosphor.unit	eluent.B.nh4ac	eluent.B.nh4ac.unit	eluent.B.nh4form	eluent.B.nh4form.unit	eluent.B.nh4carb	eluent.B.nh4carb.unit	eluent.B.nh4bicarb	eluent.B.nh4bicarb.unit	eluent.B.nh4f	eluent.B.nh4f.unit	eluent.B.nh4oh	eluent.B.nh4oh.unit	eluent.B.trieth	eluent.B.trieth.unit	eluent.B.triprop	eluent.B.triprop.unit	eluent.B.tribut	eluent.B.tribut.unit	eluent.B.nndimethylhex	eluent.B.nndimethylhex.unit	eluent.B.medronic	eluent.B.medronic.unit	eluent.B.pH	eluent.B.heptafluorobutyric	eluent.B.heptafluorobutyric.unit	eluent.C.h2o	eluent.C.meoh	eluent.C.acn	eluent.C.iproh	eluent.C.acetone	eluent.C.hex	eluent.C.chcl3	eluent.C.ch2cl2	eluent.C.hept	eluent.C.formic	eluent.C.formic.unit	eluent.C.acetic	eluent.C.acetic.unit	eluent.C.trifluoroacetic	eluent.C.trifluoroacetic.unit	eluent.C.phosphor	eluent.C.phosphor.unit	eluent.C.nh4ac	eluent.C.nh4ac.unit	eluent.C.nh4form	eluent.C.nh4form.unit	eluent.C.nh4carb	eluent.C.nh4carb.unit	eluent.C.nh4bicarb	eluent.C.nh4bicarb.unit	eluent.C.nh4f	eluent.C.nh4f.unit	eluent.C.nh4oh	eluent.C.nh4oh.unit	eluent.C.trieth	eluent.C.trieth.unit	eluent.C.triprop	eluent.C.triprop.unit	eluent.C.tribut	eluent.C.tribut.unit	eluent.C.nndimethylhex	eluent.C.nndimethylhex.unit	eluent.C.medronic	eluent.C.medronic.unit	eluent.C.pH	eluent.C.heptafluorobutyric	eluent.C.heptafluorobutyric.unit	eluent.D.h2o	eluent.D.meoh	eluent.D.acn	eluent.D.iproh	eluent.D.acetone	eluent.D.hex	eluent.D.chcl3	eluent.D.ch2cl2	eluent.D.hept	eluent.D.formic	eluent.D.formic.unit	eluent.D.acetic	eluent.D.acetic.unit	eluent.D.trifluoroacetic	eluent.D.trifluoroacetic.unit	eluent.D.phosphor	eluent.D.phosphor.unit	eluent.D.nh4ac	eluent.D.nh4ac.unit	eluent.D.nh4form	eluent.D.nh4form.unit	eluent.D.nh4carb	eluent.D.nh4carb.unit	eluent.D.nh4bicarb	eluent.D.nh4bicarb.unit	eluent.D.nh4f	eluent.D.nh4f.unit	eluent.D.nh4oh	eluent.D.nh4oh.unit	eluent.D.trieth	eluent.D.trieth.unit	eluent.D.triprop	eluent.D.triprop.unit	eluent.D.tribut	eluent.D.tribut.unit	eluent.D.nndimethylhex	eluent.D.nndimethylhex.unit	eluent.D.medronic	eluent.D.medronic.unit	eluent.D.pH	eluent.D.heptafluorobutyric	eluent.D.heptafluorobutyric.unit	gradient.start.A	gradient.start.B	gradient.start.C	gradient.start.D	gradient.end.A	gradient.end.B	gradient.end.C	gradient.end.D"""

cols = header.split("\t")
print("Number of columns:", len(cols))  # should be 185


row = {c: "" for c in cols}

# Fill in HILIC Info

# Column / hardware
row["id"] = "0000"
row["column.name"] = "Agilent InfinityLab Poroshell 120 HILIC-Z"
row["column.usp.code"] = "L114"
row["column.length"] = "150"          # mm
row["column.id"] = "2.1"              # mm
row["column.particle.size"] = "2.7"   # µm
row["column.temperature"] = "40"      # °C
row["column.flowrate"] = "0.45"       # mL/min
row["column.t0"] = "0.5"              # void time (you can change to measured)

# Mobile phase A: 99.8% H2O, 0.2% acetic acid, 5 mM NH4OAc, 5 µM medronic
row["eluent.A.h2o"] = "99.8"
row["eluent.A.meoh"] = "0"
row["eluent.A.acn"] = "0"
row["eluent.A.iproh"] = "0"
row["eluent.A.acetone"] = "0"
row["eluent.A.hex"] = "0"
row["eluent.A.chcl3"] = "0"
row["eluent.A.ch2cl2"] = "0"
row["eluent.A.hept"] = "0"

row["eluent.A.formic"] = "0"
row["eluent.A.formic.unit"] = ""

row["eluent.A.acetic"] = "0.2"
row["eluent.A.acetic.unit"] = "%"

row["eluent.A.trifluoroacetic"] = "0"
row["eluent.A.trifluoroacetic.unit"] = ""

row["eluent.A.phosphor"] = "0"
row["eluent.A.phosphor.unit"] = ""

row["eluent.A.nh4ac"] = "5"
row["eluent.A.nh4ac.unit"] = "mM"

row["eluent.A.nh4form"] = "0"
row["eluent.A.nh4form.unit"] = ""

row["eluent.A.nh4carb"] = "0"
row["eluent.A.nh4carb.unit"] = ""

row["eluent.A.nh4bicarb"] = "0"
row["eluent.A.nh4bicarb.unit"] = ""

row["eluent.A.nh4f"] = "0"
row["eluent.A.nh4f.unit"] = ""

row["eluent.A.nh4oh"] = "0"
row["eluent.A.nh4oh.unit"] = ""

row["eluent.A.trieth"] = "0"
row["eluent.A.trieth.unit"] = ""

row["eluent.A.triprop"] = "0"
row["eluent.A.triprop.unit"] = ""

row["eluent.A.tribut"] = "0"
row["eluent.A.tribut.unit"] = ""

row["eluent.A.nndimethylhex"] = "0"
row["eluent.A.nndimethylhex.unit"] = ""

row["eluent.A.medronic"] = "5"
row["eluent.A.medronic.unit"] = "uM"  # ASCII µM
row["eluent.A.pH"] = ""

row["eluent.A.heptafluorobutyric"] = "0"
row["eluent.A.heptafluorobutyric.unit"] = ""

# Mobile phase B: 95% ACN / 5% H2O, same additives
row["eluent.B.h2o"] = "5"
row["eluent.B.meoh"] = "0"
row["eluent.B.acn"] = "95"
row["eluent.B.iproh"] = "0"
row["eluent.B.acetone"] = "0"
row["eluent.B.hex"] = "0"
row["eluent.B.chcl3"] = "0"
row["eluent.B.ch2cl2"] = "0"
row["eluent.B.hept"] = "0"

row["eluent.B.formic"] = "0"
row["eluent.B.formic.unit"] = ""

row["eluent.B.acetic"] = "0.2"
row["eluent.B.acetic.unit"] = "%"

row["eluent.B.trifluoroacetic"] = "0"
row["eluent.B.trifluoroacetic.unit"] = ""

row["eluent.B.phosphor"] = "0"
row["eluent.B.phosphor.unit"] = ""

row["eluent.B.nh4ac"] = "5"
row["eluent.B.nh4ac.unit"] = "mM"

row["eluent.B.nh4form"] = "0"
row["eluent.B.nh4form.unit"] = ""

row["eluent.B.nh4carb"] = "0"
row["eluent.B.nh4carb.unit"] = ""

row["eluent.B.nh4bicarb"] = "0"
row["eluent.B.nh4bicarb.unit"] = ""

row["eluent.B.nh4f"] = "0"
row["eluent.B.nh4f.unit"] = ""

row["eluent.B.nh4oh"] = "0"
row["eluent.B.nh4oh.unit"] = ""

row["eluent.B.trieth"] = "0"
row["eluent.B.trieth.unit"] = ""

row["eluent.B.triprop"] = "0"
row["eluent.B.triprop.unit"] = ""

row["eluent.B.tribut"] = "0"
row["eluent.B.tribut.unit"] = ""

row["eluent.B.nndimethylhex"] = "0"
row["eluent.B.nndimethylhex.unit"] = ""

row["eluent.B.medronic"] = "5"
row["eluent.B.medronic.unit"] = "uM"
row["eluent.B.pH"] = ""

row["eluent.B.heptafluorobutyric"] = "0"
row["eluent.B.heptafluorobutyric.unit"] = ""

# Eluents C and D not used → all 0 / empty
for phase in ["C", "D"]:
    for comp in [
        "h2o","meoh","acn","iproh","acetone","hex","chcl3","ch2cl2","hept",
        "formic","acetic","trifluoroacetic","phosphor",
        "nh4ac","nh4form","nh4carb","nh4bicarb","nh4f","nh4oh",
        "trieth","triprop","tribut","nndimethylhex","medronic"
    ]:
        key = f"eluent.{phase}.{comp}"
        if key in row:
            row[key] = "0"
        unit_key = f"eluent.{phase}.{comp}.unit"
        if unit_key in row:
            row[unit_key] = ""
    if f"eluent.{phase}.pH" in row:
        row[f"eluent.{phase}.pH"] = ""
    if f"eluent.{phase}.heptafluorobutyric" in row:
        row[f"eluent.{phase}.heptafluorobutyric"] = "0"
    if f"eluent.{phase}.heptafluorobutyric.unit" in row:
        row[f"eluent.{phase}.heptafluorobutyric.unit"] = ""

# Gradient summary (matches your gradients.tsv HILIC program)
row["gradient.start.A"] = "0"
row["gradient.start.B"] = "100"
row["gradient.start.C"] = "0"
row["gradient.start.D"] = "0"
row["gradient.end.A"] = "80"
row["gradient.end.B"] = "20"
row["gradient.end.C"] = "0"
row["gradient.end.D"] = "0"

# 4. Make DataFrame and write metadata.tsv
df = pd.DataFrame([row], columns=cols)
df.to_csv(f"{dir}{id}_metadata.tsv", sep="\t", index=False)
print("Wrote metadata.tsv with", df.shape[1], "columns.")
