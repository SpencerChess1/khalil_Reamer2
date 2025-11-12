
# streamlit_app.py
# Run: streamlit run streamlit_app.py
# Requires: pip install streamlit plotly pandas

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(page_title="Strip-Log Dashboard (Elapsed Time)", layout="wide")

@st.cache_data(show_spinner=False)
def load_data(csv_path: Path):
    df = pd.read_csv(csv_path)
    # Build elapsed time (min) from DATE+TIME if possible; fallback to index
    def parse_dt(row):
        d = str(row.get("DATE",""))
        t = str(row.get("TIME",""))
        try:
            return pd.to_datetime(d + " " + t, errors="coerce")
        except Exception:
            return pd.NaT
    ts = df.apply(parse_dt, axis=1)
    if ts.isna().all():
        elapsed = np.arange(len(df), dtype=float)
    else:
        ts = pd.to_datetime(ts)
        elapsed = (ts - ts.iloc[0]).dt.total_seconds()/60.0
    df.insert(0, "Elapsed Time (min)", np.round(elapsed, 3))
    return df

CSV_NAME = "ENHANCED.csv"
csv_path = Path(__file__).parent / CSV_NAME
if not csv_path.exists():
    st.error(f"Couldn't find {CSV_NAME} next to this script. Put it in the same folder.")
    st.stop()

df = load_data(csv_path)
cols = set(df.columns)

# Helper: add computed Vibration Severity magnitude if the three AVG axes exist
def add_vib_magnitude(df):
    need = {"AVGX","AVGY","AVGZ"}
    if need.issubset(df.columns):
        df["VIB Avg Magnitude"] = np.sqrt(df["AVGX"]**2 + df["AVGY"]**2 + df["AVGZ"]**2)
    return df
df = add_vib_magnitude(df)

# Color palette
palette = {
    "hook":"#1f77b4", "block":"#ff7f0e", "rop":"#2ca02c",
    "wob":"#9467bd", "doc_wob_avg":"#8c564b","doc_wob_max":"#e377c2",
    "flow":"#17becf","spp":"#bcbd22","torque":"#7f7f7f","doc_tq_avg":"#d62728","doc_tq_max":"#9467bd",
    "mud":"#1f77b4","emw":"#ff7f0e","pwd_pumpoff":"#2ca02c","ann_pwd":"#d62728","pwd_int":"#8c564b","pwd_diff":"#e377c2","sk_bp":"#17becf",
    "gp_temp":"#7f7f7f","surf_motor_rpm":"#1f77b4","gp_rpm_max":"#ff7f0e","gp_rpm_min":"#2ca02c","vib_mag":"#d62728",
    "surf_rpm":"#1f77b4","stick":"#bcbd22","ddsr_max":"#ff7f0e","ddsr_avg":"#9467bd","ddsr_min":"#2ca02c",
    "gp_deflect":"#e377c2","gp_toolface":"#1f77b4","gp_inc":"#ff7f0e","doc_dir":"#d62728","doc_bend_avg":"#2ca02c",
    "doc1_avgx":"#1f77b4","doc1_avgy":"#ff7f0e","doc1_avgz":"#2ca02c",
    "doc1_peakx":"#d62728","doc1_peaky":"#9467bd","doc1_peakz":"#8c564b",
    "gp_avgx":"#1f77b4","gp_avgy":"#ff7f0e","gp_avgz":"#2ca02c",
    "gp_peakx":"#d62728","gp_peaky":"#9467bd","gp_peakz":"#8c564b",
}

# Map requested friendly names to actual CSV columns (best-effort based on provided file)
# We will keep only columns that exist; missing ones will be skipped.
map_cols = {
    # Track 1
    "hookload": "Hookload Avg(klb)",
    "block_position": "Block Position(ft)",
    "avg_rop": "ROP Avg(fph)",
    "surface_wob": "WOB Avg(klb)",
    "doc_avg_wob": "DrillDOC Avg Weight on Bit",  # best match
    "doc_max_wob": "DCWX(klb)",                  # best available proxy
    # Track 2
    "total_flow": "Total Flow(gpm)",
    "spp": "Pres Xducer 1(psig)",                # standpipe pressure proxy
    "surface_torque": "Torque Abs Avg",
    "doc_avg_torque": "DCTX(f-p)",
    "doc_max_torque": "DCTA(f-p)",               # if this is avg bend moment in your nomenclature, we can swap
    # Track 3
    "mud_density_in": "Dens Mud In Avg(ppg)",
    "ann_emw_pwd": "Ann Pres",                   # no EMW-PWD column in file; using Ann Pres as proxy
    "pwd_pump_off": None,
    "ann_pres_pwd": "Ann Pres",
    "pwd_internal_press": None,
    "diff_pres_pwd": None,
    "sk_surface_back_pressure": None,
    # Track 4
    "gp_temp": None,                              # not present
    "surf_plus_motor_rpm": None,                  # rendered as two series below
    "rpm_surface": "RPM(rpm)",
    "gp_rpm": "GP Mean RPM",
    "gp_rpm_max": "GP Max RPM",
    "gp_rpm_min": "GP Min RPM",
    "vibration_severity": "VIB Avg Magnitude",    # computed from AVGX/AVGY/AVGZ if available
    # Track 5
    "surface_rpm_avg": "RPM Surface Avg(rpm)",
    "stick_slip": "StickSlip Ind(NONE)",
    "ddsr_max_rpm": None,
    "ddsr_avg_rpm": None,
    "ddsr_min_rpm": None,
    # Track 6
    "gp_bit_deflection_rt": None,
    "gp_toolface": None,
    "gp_inclination": None,
    "doc_bend_mom_dir_deg": "DCD(deg)",
    "doc_avg_bend_mom": "DCTA(f-p)",
    # Track 7
    "doc1_avg_x": "DDS Avg X(g)",
    "doc1_avg_y": "DDS Avg Y(g)",
    "doc1_avg_z": "DDS Avg Z(g)",
    # Track 8
    "doc1_peak_x": "DDS Peak X(g)",
    "doc1_peak_y": "DDS Peak Y(g)",
    "doc1_peak_z": "DDS Peak Z(g)",
    # Track 9
    "gp_avg_x": "DDS Avg X(g).1",
    "gp_avg_y": "DDS Avg Y(g).1",
    "gp_avg_z": "DDS Avg Z(g).1",
    # Track 10
    "gp_peak_x": "DDS Peak X(g).1",
    "gp_peak_y": "DDS Peak Y(g).1",
    "gp_peak_z": "DDS Peak Z(g).1",
}

def exists(key):
    col = map_cols.get(key)
    return (col in df.columns) if col else False

# --- Build the user's requested tracks ---
# axis: 'L' or 'R' for left/right Y axis; we try to separate pressure-like vs torque-like when useful
tracks = {
    "01: Hookload / Block / ROP / WOB (DOC)": [
        (map_cols["hookload"], "Hookload", "L", "hook"),
        (map_cols["block_position"], "Block Position", "L", "block"),
        (map_cols["avg_rop"], "Avg ROP", "R", "rop"),
        (map_cols["surface_wob"], "Surface WOB", "L", "wob"),
        (map_cols["doc_avg_wob"], "DOC Avg WOB", "R", "doc_wob_avg"),
        (map_cols["doc_max_wob"], "DOC Max WOB", "R", "doc_wob_max"),
    (map_cols["rpm_surface"], "Surface RPM", "L", "surf_rpm"),
    (map_cols["gp_rpm"], "Motor RPM (GP Mean)", "R", "gp_rpm"),

    ],
    "02: Flow / SPP / Torque (DOC)": [
        (map_cols["total_flow"], "Total Flow", "L", "flow"),
        (map_cols["spp"], "SPP", "R", "spp"),
        (map_cols["surface_torque"], "Surface Torque", "L", "torque"),
        (map_cols["doc_avg_torque"], "DOC Avg Torque", "R", "doc_tq_avg"),
        (map_cols["doc_max_torque"], "DOC Max Torque", "R", "doc_tq_max"),
    ],
    "03: Mud Density / PWD & Pressures": [
        (map_cols["mud_density_in"], "Mud Density In", "L", "mud"),
        (map_cols["ann_emw_pwd"], "Ann EMW-PWD (proxy)", "R", "emw"),
        (map_cols["pwd_pump_off"], "PWD Pump Off", "R", "pwd_pumpoff"),
        (map_cols["ann_pres_pwd"], "Ann Pres - PWD", "R", "ann_pwd"),
        (map_cols["pwd_internal_press"], "PWD Internal Press", "R", "pwd_int"),
        (map_cols["diff_pres_pwd"], "Diff Pres - PWD", "R", "pwd_diff"),
        (map_cols["sk_surface_back_pressure"], "SK Surface Back Pressure", "R", "sk_bp"),
    ],
    "04: GP Temp / RPMs / Vib Severity": [
        (map_cols["gp_temp"], "GP Temp", "L", "gp_temp"),
        (map_cols["rpm_surface"], "Surface RPM", "L", "surf_motor_rpm"),
        (map_cols["gp_rpm"], "Motor RPM (GP Mean)", "R", "surf_motor_rpm"),
        (map_cols["gp_rpm_max"], "GP RPM Max", "R", "gp_rpm_max"),
        (map_cols["gp_rpm_min"], "GP RPM Min", "R", "gp_rpm_min"),
        (map_cols["vibration_severity"], "Vibration Severity (Avg Magnitude)", "L", "vib_mag"),
    ],
    "05: Surface RPM / Stick-Slip / DDSR RPMs": [
        (map_cols["surface_rpm_avg"], "Surface RPM (Avg)", "L", "surf_rpm"),
        (map_cols["stick_slip"], "Stick-Slip Ind", "R", "stick"),
        (map_cols["ddsr_max_rpm"], "DDSR Max RPM", "R", "ddsr_max"),
        (map_cols["ddsr_avg_rpm"], "DDSR Avg RPM", "R", "ddsr_avg"),
        (map_cols["ddsr_min_rpm"], "DDSR Min RPM", "R", "ddsr_min"),
    ],
    "06: GP Deflection / Toolface / Incl / DOC Bending": [
        (map_cols["gp_bit_deflection_rt"], "GP Bit Deflection RT", "L", "gp_deflect"),
        (map_cols["gp_toolface"], "GP Toolface", "L", "gp_toolface"),
        (map_cols["gp_inclination"], "GP Inclination", "L", "gp_inc"),
        (map_cols["doc_bend_mom_dir_deg"], "DOC Bend Mom Dir (deg)", "R", "doc_dir"),
        (map_cols["doc_avg_bend_mom"], "DOC Avg Bend Mom", "L", "doc_bend_avg"),
    ],
    "07: DDSR-DOC1 Averages (X/Y/Z)": [
        (map_cols["doc1_avg_x"], "DOC1 Avg X", "L", "doc1_avgx"),
        (map_cols["doc1_avg_y"], "DOC1 Avg Y", "L", "doc1_avgy"),
        (map_cols["doc1_avg_z"], "DOC1 Avg Z", "L", "doc1_avgz"),
    ],
    "08: DDSR-DOC1 Peaks (X/Y/Z)": [
        (map_cols["doc1_peak_x"], "DOC1 Peak X", "R", "doc1_peakx"),
        (map_cols["doc1_peak_y"], "DOC1 Peak Y", "R", "doc1_peaky"),
        (map_cols["doc1_peak_z"], "DOC1 Peak Z", "R", "doc1_peakz"),
    ],
    "09: DDSR-GP Averages (X/Y/Z)": [
        (map_cols["gp_avg_x"], "GP Avg X", "L", "gp_avgx"),
        (map_cols["gp_avg_y"], "GP Avg Y", "L", "gp_avgy"),
        (map_cols["gp_avg_z"], "GP Avg Z", "L", "gp_avgz"),
    ],
    "10: DDSR-GP Peaks (X/Y/Z)": [
        (map_cols["gp_peak_x"], "GP Peak X", "R", "gp_peakx"),
        (map_cols["gp_peak_y"], "GP Peak Y", "R", "gp_peaky"),
        (map_cols["gp_peak_z"], "GP Peak Z", "R", "gp_peakz"),
    ],
}

# Drop any None or missing columns from each track
for k in list(tracks.keys()):
    cleaned = []
    for col, name, ax, color in tracks[k]:
        if col and (col in df.columns):
            cleaned.append((col, name, ax, color))
    tracks[k] = cleaned
    if not tracks[k]:
        del tracks[k]

# Sidebar controls
xmin = float(df["Elapsed Time (min)"].min())
xmax = float(df["Elapsed Time (min)"].max())
st.sidebar.header("Filters")
time_range = st.sidebar.slider("Elapsed Time Range (min)", min_value=round(xmin,2), max_value=round(xmax,2),
                               value=(round(xmin,2), round(xmax,2)), step=0.1)
selected_tracks = st.sidebar.multiselect("Tracks to show", list(tracks.keys()), default=list(tracks.keys()))
show_secondary_grid = st.sidebar.checkbox("Show right Y-axis gridlines", value=False)

df_view = df.query("`Elapsed Time (min)` >= @time_range[0] and `Elapsed Time (min)` <= @time_range[1]")

def build_track_fig(title, series_specs, df_section):
    use_secondary = any(ax == "R" for _,_,ax,_ in series_specs)
    fig = make_subplots(rows=1, cols=1, specs=[[{"secondary_y": use_secondary}]])
    x = df_section["Elapsed Time (min)"]
    for col, disp, ax, color_key in series_specs:
        if col not in df_section.columns: 
            continue
        y = df_section[col]
        fig.add_trace(
            go.Scatter(x=x, y=y, mode="lines", name=disp, line=dict(color=palette.get(color_key))),
            row=1, col=1, secondary_y=(ax=="R")
        )
    fig.update_layout(
        height=280, margin=dict(l=40,r=40,t=40,b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0)
    )
    fig.update_xaxes(title_text="Elapsed Time (min)")
    if use_secondary:
        fig.update_yaxes(title_text="Left axis", secondary_y=False, showgrid=True)
        fig.update_yaxes(title_text="Right axis", secondary_y=True, showgrid=show_secondary_grid)
    else:
        fig.update_yaxes(title_text="", showgrid=True)
    fig.update_layout(title=title)
    return fig

st.title("Strip-Log Dashboard — Custom Tracks")
st.caption("Overlayed charts vs Elapsed Time (min) from ENHANCED.csv")

missing_notes = []
# Render charts
for t in selected_tracks:
    fig = build_track_fig(t, tracks[t], df_view)
    st.plotly_chart(fig, use_container_width=True)

# Report any of the requested friendly items that were not found in the CSV
requested_cols = {k:v for k,v in map_cols.items() if v}
not_found = [v for v in requested_cols.values() if v not in df.columns]
if not_found:
    with st.expander("Note: Some requested inputs not found in the CSV (skipped)"):
        st.write(sorted(not_found))
