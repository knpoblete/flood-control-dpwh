import streamlit as st
from datetime import datetime
import geopandas as gpd
import pandas as pd
import folium
from streamlit_folium import st_folium
import matplotlib.cm as cm
import matplotlib.colors as colors
import plotly.express as px
import numpy as np
from utils import apply_filters, plot_projects, plot_swarm, set_font, plot_contractors



# ---------- Page config (set once) ----------
st.set_page_config(
    page_title="PH Flood Control Project Tracker",
    page_icon="👀",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown(set_font(), unsafe_allow_html=True)
st.sidebar.title(" ")




# ---------- Helpers for summary ----------
def _fmt_currency(n):
    if pd.isna(n) or n is None:
        return "—"
    return f"₱{n:,.0f}"

def _top_n(series, n=5):
    s = series.value_counts().head(n)
    return ", ".join([f"{idx} ({val})" for idx, val in s.items()]) if not s.empty else "—"

def summarize(df: pd.DataFrame) -> str:
    if df.empty:
        return "No projects match the current filters."
    total_projects = len(df)
    total_cost = df["ContractCost"].sum(skipna=True) if "ContractCost" in df.columns else 0
    mean_cost = df["ContractCost"].mean(skipna=True) if "ContractCost" in df.columns else None
    lines = [
        f"**Snapshot of your selection**",
        f"- **Projects:** {total_projects}",
        f"- **Total cost:** {_fmt_currency(total_cost)}",
        f"- **Average cost:** {_fmt_currency(mean_cost)}",
    ]
    if "Region" in df.columns and df["Region"].nunique() > 1:
        lines.append(f"- Top Regions: {_top_n(df['Region'])}")
    if "Contractor" in df.columns:
        lines.append(f"- Top Contractors: {_top_n(df['Contractor'])}")
    return "\n".join(lines)

# ---------- Summary panel (on the right side) ----------
left_col, right_col = st.columns([2, 1])  # 2/3 width for main content, 1/3 for summary

with left_col:
    st.title("👀 PH Flood Control Project Tracker")
    st.write(
        "The Philippines has long been vulnerable to powerful typhoons and widespread flooding, events that have repeatedly "
        "disrupted the lives of millions of Filipinos. "
        "Despite significant government investments in flood control projects, the country continues to face severe flooding challenges."
        " This project takes a closer look at historical data of flood control projects from www.sumbongsapangulo.ph "
        "to better understand how government investments were distributed and utilized."
    )
    st.markdown("## Large investment in flood control projects started in 2022.")
    st.write("This tracker is interactive, select from the filters on the left side panel to explore the data.")

    # ---------- Data loading ----------
    @st.cache_data(show_spinner=False)
    def load_data(path: str):
        df = gpd.read_file(path)

        # Dates
        df["StartDate"] = pd.to_datetime(df["StartDate"], errors="coerce")
        df["StartYear"] = df["StartDate"].dt.year.astype("Int64")   # nullable int

        # Geometry -> lon/lat
        df["lon"] = df.geometry.x.astype("float")
        df["lat"] = df.geometry.y.astype("float")

        # ContractCost as float, CompletionYear as Int
        if "ContractCost" in df.columns:
            df["ContractCost"] = pd.to_numeric(df["ContractCost"], errors="coerce").astype("float32")
        if "CompletionYear" in df.columns:
            df["CompletionYear"] = pd.to_numeric(df["CompletionYear"], errors="coerce").astype("Int64")  # nullable integer

        return df

    @st.cache_resource
    def get_colormap(vmin, vmax):
        # integer bounds for stability with LogNorm
        vmin = int(max(vmin, 1))
        vmax = int(vmax) if vmax and vmax > vmin else vmin + 1
        norm = colors.LogNorm(vmin=vmin, vmax=vmax)
        return cm.ScalarMappable(norm=norm, cmap="Reds")

    df = load_data("flood_control_projects.geojson")
    cmap = get_colormap(df["ContractCost"].min(), df["ContractCost"].max())

    # ---------- Region order & categorical ----------
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
    df["Region"] = pd.Categorical(df["Region"], categories=custom_order, ordered=True)

    # ---------- Sidebar form (batched filters) ----------
    with st.sidebar.form("filters"):
        # Region list: use categorical order if present, else sorted uniques
        regions_sorted = (
            list(df["Region"].cat.categories)
            if "Region" in df.columns and hasattr(df["Region"], "cat")
            else sorted(df["Region"].dropna().unique())
        )
        region_values = st.selectbox("Region", regions_sorted, index=None)

        # Province depends on Region
        if region_values is None:
            province_options = sorted(df["Province"].dropna().unique())
        else:
            province_options = sorted(df.loc[df["Region"] == region_values, "Province"].dropna().unique())
        province_values = st.selectbox("Province", province_options, index=None)

        # Municipality depends on Province (not Region)
        if province_values is None:
            municipality_options = sorted(df["Municipality"].dropna().unique())
        else:
            municipality_options = sorted(df.loc[df["Province"] == province_values, "Municipality"].dropna().unique())
        municipality_values = st.selectbox("Municipality", municipality_options, index=None)

        # Other filters
        TypeofWork_values = st.selectbox(
            "Type of Work",
            sorted(df["TypeofWork"].dropna().unique()),
            index=None,
        )
        Contractor_values = st.selectbox(
            "Contractor",
            sorted(df["Contractor"].dropna().unique()),
            index=None,
        )

        # Year sliders (guard against NA)
        start_year_series = pd.to_numeric(df["StartYear"], errors="coerce").dropna()
        completion_year_series = pd.to_numeric(df["CompletionYear"], errors="coerce").dropna()

        start_year_min = int(start_year_series.min()) if not start_year_series.empty else 2000
        start_year_max = int(start_year_series.max()) if not start_year_series.empty else 2025

        completion_year_min = int(completion_year_series.min()) if not completion_year_series.empty else start_year_min
        completion_year_max = int(completion_year_series.max()) if not completion_year_series.empty else start_year_max

        start_year_values = st.slider(
            "Start Year",
            start_year_min, start_year_max,
            (start_year_min, start_year_max),
            step=1
        )
        completion_year_values = st.slider(
            "Completion Year",
            completion_year_min, completion_year_max,
            (completion_year_min, completion_year_max),
            step=1
        )

        submitted = st.form_submit_button("Apply filters")

    # ---------- Persist readiness; avoid post-submit blank page ----------
    if "filters_ready" not in st.session_state:
        st.session_state["filters_ready"] = True  # render on first load

    if submitted:
        st.session_state["filters_ready"] = True  # keep rendering after apply

    if not st.session_state["filters_ready"]:
        st.info("Adjust filters and click Apply to load the dashboard.")
        st.stop()

    # ---------- Pills summary ----------
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

    st.markdown(
        f"""
        <div class="pill-container">
            {' '.join(pills)}
        </div>
        """,
        unsafe_allow_html=True
    )
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

    # ---------- Colors for map points ----------
    norm = colors.LogNorm(
        vmin=max(df["ContractCost"].min(), 1),
        vmax=df["ContractCost"].max()
    )
    colormap = cm.ScalarMappable(norm=norm, cmap="Reds")
    df["color"] = df["ContractCost"].apply(lambda v: colors.to_hex(colormap.to_rgba(v)))

    # ---------- Build filters for your existing helpers ----------
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

    # ---------- Apply filters ----------
    df_filtered = apply_filters(
        df,
        equals=equals,
        num_ranges=num_ranges,
    )

    # ---------- Folium map (safe center + empty guard) ----------
    center_lat = float(df_filtered["lat"].mean()) if not df_filtered.empty else 12.8797
    center_lon = float(df_filtered["lon"].mean()) if not df_filtered.empty else 121.7740

    _map = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=5.5,
        tiles='CartoDB positron'
    )

    if not df_filtered.empty:
        for _, row in df_filtered.iterrows():
            folium.Circle(
                location=[row["lat"], row["lon"]],
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
                </div>
                """
            ).add_to(_map)

        # Auto-zoom to filtered data
        _coords = df_filtered[["lat", "lon"]].dropna()
        if len(_coords) == 1:
            lat, lon = _coords.iloc[0]
            _map.location = [lat, lon]
            _map.zoom_start = 12
        elif len(_coords) > 1:
            sw = [_coords["lat"].min(), _coords["lon"].min()]
            ne = [_coords["lat"].max(), _coords["lon"].max()]
            _map.fit_bounds([sw, ne], padding=(30, 30))

    # ---------- Summary charts ----------
    counts = df_filtered.groupby("StartYear").size().reset_index(name="metric")
    total_cost_year = df_filtered.groupby("StartYear", as_index=False).agg(metric=("ContractCost", "sum"))
    avearge_cost = df_filtered.groupby("StartYear", as_index=False).agg(metric=("ContractCost", "mean"))

    fig_total_projects = plot_projects(counts)
    fig_total_cost = plot_projects(total_cost_year, currency=True)
    config = {"displayModeBar": False}

    contractors_by_cost = (
        df_filtered.groupby("Contractor", as_index=False)
        .agg(metric=("ContractCost", "sum"))
        .sort_values("metric", ascending=False)
    )
    total_cost_val = contractors_by_cost["metric"].sum()
    contractors_by_cost["pct_of_total"] = (
        contractors_by_cost["metric"] / total_cost_val * 100 if total_cost_val else 0
    ).round(2)
    contractors_by_cost = contractors_by_cost.head(20)
    text_pct_cost = f"{int(round(contractors_by_cost['pct_of_total'].sum(), 0))}% of the contracts were awarded to these contractors."

    contractors_by_size = (
        df_filtered.groupby("Contractor")
        .size()
        .reset_index(name="metric")
        .sort_values("metric", ascending=False)
    )
    total_size = contractors_by_size["metric"].sum()
    contractors_by_size["pct_of_total"] = (
        contractors_by_size["metric"] / total_size * 100 if total_size else 0
    ).round(2)
    contractors_by_size = contractors_by_size.head(20)
    text_pct_size = f"{int(round(contractors_by_size['pct_of_total'].sum(), 0))}% of the contracts were awarded to these contractors."

    fig_contractors_cost = plot_contractors(contractors_by_cost, currency=True)
    fig_contractors_size = plot_contractors(contractors_by_size)

    # ---------- Layout & charts ----------
    st.markdown("""
    <style>
    /* Give bordered containers a gentle card look */
    div[data-testid="stContainer"] > div {
        padding: 1rem;
        border-radius: 10px;
        background-color: #f9fafc;
        border: 1px solid #e0e0e0;
        margin-bottom: 1rem;
    }

    /* === Floating chat styles === */
    #chat-launcher-wrap,
    #chat-box-wrap {
        position: fixed;
        z-index: 10000;
    }
    #chat-launcher-wrap {
        bottom: 18px;
        right: 18px;
    }
    #chat-box-wrap {
        bottom: 84px; /* sits above the launcher */
        right: 18px;
        width: 360px;
        max-height: 70vh;
        overflow: auto;
        background: #ffffff;
        border: 1px solid #e0e0e0;
        border-radius: 12px;
        box-shadow: 0 8px 30px rgba(0,0,0,.15);
        padding: 10px 12px;
    }
    #chat-title {
        font-weight: 600;
        margin-bottom: 6px;
    }
    </style>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1])
    with col1:
        with st.container(border=True):
            st.markdown("##### Total Projects by Start Year")
            st.plotly_chart(fig_total_projects, use_container_width=False, config=config)

    with col2:
        with st.container(border=True):
            st.markdown('##### Total Project Cost by Start Year (Php)')
            st.plotly_chart(fig_total_cost, use_container_width=False, config=config)

    with st.container(border=True):
        if municipality_values is not None:
            text = "Contract Cost by Municipality"
        elif province_values is not None and municipality_values is not None:
            text = "Contract Cost by Municipality"
        elif province_values is not None:
            text = "Contract Cost by Municipality"
        elif region_values is not None:
            text = "Contract Cost by Province"
        else:
            text = "Contract Cost by Region"

        st.markdown('##### Flood Control Projects across the Philippines')
        col1, col2 = st.columns([1, 1])

        with col1:
            st_map = st_folium(_map, height=800, width=800)

        with col2:
            st.markdown(f"##### {text}")

            # ---------- Threshold control (inline, original section) ----------
            options = list(range(1_000_000, 291_000_000, 1_000_000))
            labels = [f"{x//1_000_000}M" for x in options]
            label_to_value = dict(zip(labels, options))

            selected_label = st.select_slider(
                "Threshold",
                options=labels,
                value="100M"
            )
            threshold = label_to_value[selected_label]

            if df_filtered.empty:
                st.info("No projects match the current filters.")
            else:
                if municipality_values is not None:
                    fig_projects = plot_swarm(df_filtered, custom_order, "Municipality", threshold)
                elif province_values is not None and municipality_values is not None:
                    fig_projects = plot_swarm(df_filtered, custom_order, "Municipality", threshold)
                elif province_values is not None:
                    fig_projects = plot_swarm(df_filtered, custom_order, "Municipality", threshold)
                elif region_values is not None:
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


with right_col:   # everything here stays on the right
    with st.container(border=True):
        st.markdown("##### Summary of Current Selection")
        st.markdown(summarize(df_filtered))

        # c1, c2, c3, c4 = st.columns(4)
        # with c1:
        #     st.metric("Projects", f"{len(df_filtered):,}")
        # with c2:
        #     total_cost_val = float(df_filtered["ContractCost"].sum(skipna=True)) if "ContractCost" in df_filtered.columns else 0.0
        #     st.metric("Total Cost (₱)", f"{total_cost_val:,.0f}")
        # with c3:
        #     mean_cost_val = float(df_filtered["ContractCost"].mean(skipna=True)) if "ContractCost" in df_filtered.columns else 0.0
        #     st.metric("Avg. Cost (₱)", f"{mean_cost_val:,.0f}")
        # with c4:
        #     unique_ctr = df_filtered["Contractor"].nunique(dropna=True) if "Contractor" in df_filtered.columns else 0
        #     st.metric("Unique Contractors", f"{unique_ctr:,}")

        # st.divider()

        # colA, colB = st.columns(2)
        # with colA:
        #     if "Region" in df_filtered.columns and not df_filtered.empty:
        #         st.markdown("**Top Regions (by project count)**")
        #         top_regions = (
        #             df_filtered["Region"]
        #             .value_counts()
        #             .rename_axis("Region")
        #             .reset_index(name="Projects")
        #             .head(10)
        #         )
        #         st.dataframe(top_regions, use_container_width=True, hide_index=True)
        # with colB:
        #     if "Contractor" in df_filtered.columns and not df_filtered.empty:
        #         st.markdown("**Top Contractors (by total cost)**")
        #         top_contractors = (
        #             df_filtered.groupby("Contractor", as_index=False)
        #             .agg(TotalCost=("ContractCost","sum"),
        #                  Projects=("ContractCost","size"))
        #             .sort_values("TotalCost", ascending=False)
        #             .head(10)
        #         )
        #         top_contractors["TotalCost"] = top_contractors["TotalCost"].round(0)
        #         st.dataframe(top_contractors, use_container_width=True, hide_index=True)

        # ---------- Mini Q&A ----------
        st.divider()
        st.markdown("###### Ask a quick question about the current selection")

        if "qa_history" not in st.session_state:
            st.session_state.qa_history = []

        with st.form("summary_qa", clear_on_submit=True):
            prompt = st.text_input("Question", placeholder="e.g., 'Top contractors?' or 'Total cost?'")
            sent = st.form_submit_button("Send")

        if sent and prompt.strip():
            q = prompt.strip().lower()
            # ... your existing Q&A logic here ...
            st.session_state.qa_history.append({"q": prompt, "a": answer})

        if st.session_state.qa_history:
            st.markdown("**Recent Q&A**")
            for item in st.session_state.qa_history[-5:]:
                st.markdown(f"- **You:** {item['q']}")
                st.markdown(item["a"])
                st.markdown("---")

        if not df_filtered.empty:
            csv = df_filtered.to_csv(index=False).encode("utf-8")
            st.download_button(
                "⬇️ Download filtered data (CSV)",
                data=csv,
                file_name="flood_control_projects_filtered.csv",
                mime="text/csv",
            )
