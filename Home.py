import streamlit as st
from datetime import datetime
import streamlit as st
import geopandas as gpd
import pandas as pd
import folium
from streamlit_folium import st_folium
import matplotlib.cm as cm
import matplotlib.colors as colors
import plotly.express as px
import numpy as np
from utils import apply_filters, plot_projects, plot_swarm, set_font, plot_contractors

st.markdown(set_font(), unsafe_allow_html=True)

st.set_page_config(
    page_title="PH Flood Control Project Tracker",
    page_icon="👀",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.sidebar.title(" ")
# st.sidebar.success("Use the pages in the sidebar to explore ➜")

st.title("👀 PH Flood Control Project Tracker")
st.write(
    "The Philippines has long been vulnerable to powerful typhoons and widespread flooding, events that have repeatedly " \
    "disrupted the lives of millions of Filipinos. " \
    "Despite significant government investments in flood control projects, the country continues to face severe flooding challenges." \
    " This project takes a closer look at historical data of flood control projects from www.sumbongsapangulo.ph " \
    "to better understand how government investments were distributed and utilized." 
)



st.set_page_config(layout="wide")

st.markdown("## Heavy investment in flood control projects started in 2022.")
st.write("This tracker is interactive, select from the filters on the left side panel to explore the data.")

df = gpd.read_file('flood_control_projects.geojson')
df["StartDate"] = pd.to_datetime(df["StartDate"], errors="coerce")
df["StartYear"] = df["StartDate"].dt.year.astype("Int64")
df["lon"] = df["geometry"].x
df["lat"] = df["geometry"].y
df['ContractCost_normalized'] = (df['ContractCost']-df['ContractCost'].min())/(df['ContractCost'].max()-df['ContractCost'].min()) * 10000

norm = colors.Normalize(vmin=df["ContractCost"].min(), vmax=df["ContractCost"].max())
colormap = cm.ScalarMappable(norm=norm, cmap="Reds")  


custom_order = [
                "Cordillera Administrative Region",
                "National Capital Region",
                "Region I", 
                "Region II",
                "Region III",
                "Region IV-A",
                "Region IV-B",
                "Region V",
                "Region VI",
                "Region VII",
                "Region VIII",
                "Region IX",
                "Region X",
                "Region XI",
                "Region XII",
                "Region XIII",
                ]

df["Region"] = pd.Categorical(
    df["Region"],
    categories=custom_order,
    ordered=True
)
regions_sorted = df["Region"].cat.categories.tolist()


region_values = st.sidebar.selectbox(
    "Region",
    regions_sorted,
    index=None,
)

    # Second dropdown: Province depends on Region
if region_values is None:
    province_options = sorted(df["Province"].unique())
else:
    province_options = sorted(df.loc[df["Region"] == region_values, "Province"].unique())
    municipality_sorted = sorted(
        df.loc[df["Province"] == region_values, "Municipality"].unique(),
        key=lambda x: (x is None, x)   # puts None last
    )
    municipality_options = municipality_sorted

province_values = st.sidebar.selectbox(
    "Province",
    province_options,
    index=None,
)


if province_values is None:
    municipality_sorted = sorted(
        df["Municipality"].unique(),
        key=lambda x: (x is None, x)   # puts None last
    )
    municipality_options = municipality_sorted
else:
    municipality_sorted = sorted(
        df.loc[df["Province"] == province_values, "Municipality"].unique(),
        key=lambda x: (x is None, x)   # puts None last
    )
    municipality_options = municipality_sorted

municipality_values = st.sidebar.selectbox(
    "Municipality",
    municipality_options,
    index=None,
)

TypeofWork_values = st.sidebar.selectbox(
    "Type of Work",
    df["TypeofWork"].unique(),
    index=None,
)

Contractor_values = st.sidebar.selectbox(
    "Contractor",
    df["Contractor"].unique(),
    index=None,
)

start_year_values = st.sidebar.slider("Start Year", 
                df['StartYear'].min(), 
                df['StartYear'].max(),
                (df['StartYear'].min(), df['StartYear'].max()),
                step=1)

completion_year_values = st.sidebar.slider("Completion Year", 
                df['CompletionYear'].min(), 
                df['CompletionYear'].max(),
                (df['CompletionYear'].min(), df['CompletionYear'].max()),
                step=1)



equals = {
    "Region": region_values,
    "Province": province_values,
    "Municipality": municipality_values,
    "TypeofWork": TypeofWork_values,
    "Contractor": Contractor_values
}

num_ranges = {
    "StartYear": start_year_values,
    "CompletionYear": completion_year_values,
}


# Build pills
pills = []
if region_values:
    pills.append(f"<span class='pill'>Region: {region_values}</span>")
if province_values:
    pills.append(f"<span class='pill'>Province: {province_values}</span>")    
if municipality_values:
    pills.append(f"<span class='pill'>Municipality: {municipality_values}</span>")
if TypeofWork_values:
    pills.append(f"<span class='pill'>Type of Work: {TypeofWork_values}</span>")
if start_year_values:
    pills.append(f"<span class='pill'>Start Year: {start_year_values}</span>")
if completion_year_values:
    pills.append(f"<span class='pill'>Completion Year: {completion_year_values}</span>")
if Contractor_values:
    pills.append(f"<span class='pill'>Contractor: {Contractor_values}</span>")

# Wrap in a container
st.markdown(
    f"""
    <div class="pill-container">
        {' '.join(pills)}
    </div>
    """,
    unsafe_allow_html=True
)

# Add CSS once
st.markdown(
    """
    <style>
    .pill-container {
        background-color: #f9fafc;
        padding: 10px;
        border-radius: 10px;
        border: 1px solid #e0e0e0;
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
        margin-bottom: 1rem;
    }
    .pill {
        background-color: #19535F;
        color: white;
        padding: 5px 12px;
        border-radius: 20px;
        font-size: 0.9em;
        white-space: nowrap;
    }
    </style>
    """,
    unsafe_allow_html=True
)


norm = colors.LogNorm(
    vmin=df["ContractCost"].min(),
    vmax=df["ContractCost"].max()
)

colormap = cm.ScalarMappable(norm=norm, cmap="Reds")

df["color"] = df["ContractCost"].apply(lambda v: colors.to_hex(colormap.to_rgba(v)))

df_filtered = apply_filters(
    df,
    equals=equals,
    num_ranges=num_ranges,
    # date_ranges={"Date": (start_date, end_date)}  # if you have dates
)

map = folium.Map(location=[df_filtered["lat"].mean(), df_filtered["lon"].mean()],
                         zoom_start=5.5,
                        tiles='CartoDB positron'
                         )
feature_group = folium.FeatureGroup("Locations")

for _, row in df_filtered.iterrows():
    
    folium.Circle(
        location=[row["lat"], row["lon"]],
        # radius=row["radius"],
        # radius=10000,
        color=row["color"],
        fill=True,
        fill_color=row["color"],
        fill_opacity=0.7,
        tooltip=f"""
        <div style="font-size:13px; line-height:1.4;">
        <b>Location:</b> {row['Municipality']} <br>
        <b>Type of Work:</b> {row['TypeofWork']} <br> 
        <b>Cost:</b> Php {row['ContractCost']:,.0f} <br>
        <b>Start Year:</b> {row['StartYear']} <br>
        <b>Completion Year:</b> {row['CompletionYear']} <br>
        <b>Contractor:</b> {row['Contractor']}
        """
    ).add_to(map)

# ---- Auto-zoom to filtered data ----
if not df_filtered.empty:
    # drop rows with missing coords for bounds
    _coords = df_filtered[["lat", "lon"]].dropna()
    if len(_coords) == 1:
        # single point: center there and pick a reasonable zoom
        lat, lon = _coords.iloc[0]
        map.location = [lat, lon]
        map.zoom_start = 12
    else:
        sw = [_coords["lat"].min(), _coords["lon"].min()]  # south-west
        ne = [_coords["lat"].max(), _coords["lon"].max()]  # north-east
        map.fit_bounds([sw, ne], padding=(30, 30))


# Optional: add legend
# from branca.colormap import LinearColormap
# legend = LinearColormap(
#     colors=["#fee5d9", "#fcae91", "#fb6a4a", "#de2d26", "#a50f15"],
#     vmin=df["ContractCost"].min(),
#     vmax=df["ContractCost"].max(),
#     caption="Cost"
# )
# legend.add_to(map)

counts = df_filtered.groupby("StartYear").size().reset_index(name="metric")
total_cost = df_filtered.groupby("StartYear", as_index=False).agg(metric=("ContractCost", "sum"))
avearge_cost = df_filtered.groupby("StartYear", as_index=False).agg(metric=("ContractCost", "mean"))

fig_total_projects = plot_projects(counts)
fig_total_cost = plot_projects(total_cost, currency=True)
# fig_average_cost = plot_projects(avearge_cost, currency=True)
config = {"displayModeBar": False}

contractors_by_cost = (
    df_filtered.groupby("Contractor", as_index=False)
    .agg(metric=("ContractCost", "sum"))
    .sort_values("metric", ascending=False)
)
# add % of total
total_cost = contractors_by_cost["metric"].sum()
contractors_by_cost["pct_of_total"] = (
    contractors_by_cost["metric"] / total_cost * 100
).round(2)  # 2 decimals
contractors_by_cost = contractors_by_cost.head(20) #contractors_by_cost[contractors_by_cost['metric']>=1_000_000_000]
text_pct_cost = f"{int(round(contractors_by_cost["pct_of_total"].sum(), 0))}% of the contracts are awarded to these contractors."

contractors_by_size = (
    df_filtered.groupby("Contractor")
    .size()
    .reset_index(name="metric")
    .sort_values("metric", ascending=False)
)
total_size = contractors_by_size["metric"].sum()
contractors_by_size["pct_of_total"] = (
    contractors_by_size["metric"] / total_size * 100
).round(2)  # 2 decimals
contractors_by_size = contractors_by_size.head(20) #contractors_by_cost[contractors_by_cost['metric']>=1_000_000_000]
text_pct_size = f"{int(round(contractors_by_size["pct_of_total"].sum(), 0))}% of the contracts are awarded to these contractors."

fig_contractors_cost = plot_contractors(contractors_by_cost, currency=True)
fig_contractors_size = plot_contractors(contractors_by_size)


st.markdown("""
<style>
/* Target bordered containers */
div[data-testid="stContainer"] > div {
    padding: 1rem;
    border-radius: 10px;
    background-color: #f9fafc;
    border: 1px solid #e0e0e0;
    margin-bottom: 1rem;
}
</style>
""", unsafe_allow_html=True)

# st.subheader("Flood control projects over the years")
col1, col2 = st.columns([1,1])
with col1:
    with st.container(border=True):
        st.markdown("##### Total Projects by Start Year")
        st.plotly_chart(fig_total_projects, use_container_width=False, config=config)

with col2:
    with st.container(border=True):
        st.markdown('##### Total Project Cost by Start Year (Php)')
        st.plotly_chart(fig_total_cost, use_container_width=False, config=config)
# with col3:
#     with st.container(border=True):
#         st.markdown('##### Average Project Cost by Start Year (Php)')
#         st.plotly_chart(fig_average_cost, use_container_width=False, config=config)


with st.container(border=True):
    if municipality_values != None:
        text = f"Contract Cost by Municipality"
    elif province_values != None and municipality_values != None:
        text = f"Contract Cost by Municipality"
    elif province_values != None:
        text = f"Contract Cost by Municipality"
    elif region_values != None:
        text = f"Contract Cost by Province"
    else:
        text = f"Contract Cost by Region"
    
    st.markdown('##### Flood Control Projects across the Philippines')
    col1, col2 = st.columns([1,1])

    with col1:
        st_map = st_folium(map, height=800, width=800)

    with col2:
        st.markdown(f"##### {text}")

        options = list(range(1_000_000, 291_000_000, 1_000_000))
        labels = [f"{x//1_000_000}M" for x in options]
        label_to_value = dict(zip(labels, options))

        selected_label = st.select_slider(
            "Threshold",
            options=labels,
            value="100M"
        )

        threshold = label_to_value[selected_label]
        print(threshold)

        if municipality_values != None:
            fig_projects = plot_swarm(df_filtered, custom_order, "Municipality", threshold)
        elif province_values != None and municipality_values != None:
            fig_projects = plot_swarm(df_filtered, custom_order, "Municipality", threshold)
        elif province_values != None:
            fig_projects = plot_swarm(df_filtered, custom_order, "Municipality", threshold)
        elif region_values != None:
            fig_projects = plot_swarm(df_filtered, custom_order, "Province", threshold)  
        else:
            fig_projects = plot_swarm(df_filtered, custom_order, "Region", threshold)

        
        st.plotly_chart(fig_projects, use_container_width=True, config=config)
        

with st.container(border=True):
    st.markdown('##### Top 20 Contractors engaged in Flood Control Projects')

    tab1, tab2 = st.tabs(['By Contract Cost', 'By Number of Projects'])
    with tab1:
        st.write(text_pct_cost)
        st.plotly_chart(fig_contractors_cost, use_container_width=True, config=config)
    with tab2:
        st.write(text_pct_size)
        st.plotly_chart(fig_contractors_size, use_container_width=True, config=config)

st.caption(f"Loaded at {datetime.now():%Y-%m-%d %H:%M:%S}")
