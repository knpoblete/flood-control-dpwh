import geopandas as gpd
import folium
import plotly.express as px
import plotly.io as pio

pio.templates["montserrat"] = pio.templates["plotly_white"]

pio.templates["montserrat"].layout.update(
    font=dict(
        family="Montserrat, sans-serif",
        size=12,
        color="black"
    ),
    title=dict(font=dict(family="Montserrat, sans-serif", size=14)),
    hoverlabel=dict(font=dict(family="Montserrat, sans-serif", size=12)),
    legend=dict(font=dict(family="Montserrat, sans-serif", size=12)),
    xaxis=dict(title_font=dict(family="Montserrat, sans-serif", size=12)),
    yaxis=dict(title_font=dict(family="Montserrat, sans-serif", size=12)),
)

# Make it the default template
pio.templates.default = "montserrat"

def set_font():
    return """
        <style>
        /* Load Montserrat from Google Fonts */
        @import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@300;400;600;700&display=swap');

        /* Apply Montserrat everywhere */
        html, body, [class*="css"] {
            font-family: 'Montserrat', sans-serif !important;
        }

        /* Streamlit widgets: inputs, labels, dropdowns, sliders */
        .stSelectbox label, .stMultiSelect label, .stRadio label, .stCheckbox label,
        .stTextInput label, .stNumberInput label, .stSlider label, .stDateInput label {
            font-family: 'Montserrat', sans-serif !important;
            font-weight: 500;
        }

        .stSelectbox div, .stMultiSelect div, .stRadio div, .stCheckbox div,
        .stTextInput input, .stNumberInput input, .stTextArea textarea,
        .stSlider div, .stDateInput input {
            font-family: 'Montserrat', sans-serif !important;
            font-weight: 400;
        }

        /* Markdown headings */
        .stMarkdown h1, .stMarkdown h2, .stMarkdown h3,
        .stMarkdown h4, .stMarkdown h5, .stMarkdown h6 {
            font-family: 'Montserrat', sans-serif !important;
            font-weight: 600;
            letter-spacing: 0.2px;
        }

        /* Dropdown options */
        div[role="listbox"] div[role="option"] {
            font-family: 'Montserrat', sans-serif !important;
            font-weight: 400;
        }

        /* Markdown & st.write text */
        .stMarkdown p, .stMarkdown, p {
            font-family: 'Montserrat', sans-serif !important;
            font-weight: 400;
            font-size: 15px;  /* optional, tweak if you want */
            line-height: 1.5; /* optional */
        }

        </style>

    """

def set_text(text, size="h3"):
    outstring = f"""<{size} style="color:#151E3F;">{text}</{size}>"""
    return outstring

def read_data(filepath):
    data = gpd.read_file(filepath)
    data['lon'] = data['geometry'].get_coordinates(ignore_index=True).x
    data['lat'] = data['geometry'].get_coordinates(ignore_index=True).y
    data['location'] = list(zip(data['lat'],data['lon']))
    hospitals = data[data['amenity']=='hospital']
    return hospitals

def show_map_marker(df):
    map = folium.Map(location=[13, 122], 
                 zoom_start=5,
                #  width=500,
                #  height=500,
                 tiles='https://api.mapbox.com/styles/v1/knpoblete/clgi3uciu001801ln65r8ors6/tiles/256/{z}/{x}/{y}@2x?access_token=pk.eyJ1Ijoia25wb2JsZXRlIiwiYSI6ImNsZ2k5NGE4ZTBpc2IzY2xmemgwdWl5NXkifQ.UQ55c1MEZb-mQV-x66gKIQ',
                 attr='Mapbox Light')

    feature_group = folium.FeatureGroup("Locations")

    for lat, lon, name in zip(list(df['lat']), list(df['lon']), list(df['location'])):
        feature_group.add_child(folium.Marker(location=[lat,lon],
                                            popup=name,
                                            icon=folium.Icon(color="red", icon="glyphicon glyphicon-plus", size=0.5)))

    map.add_child(feature_group)

    return map

def show_map_circle(df):
    map = folium.Map(location=[13, 122], 
                 zoom_start=5,
                #  width=500,
                 height=600,
                 tiles='https://api.mapbox.com/styles/v1/knpoblete/clgi3uciu001801ln65r8ors6/tiles/256/{z}/{x}/{y}@2x?access_token=pk.eyJ1Ijoia25wb2JsZXRlIiwiYSI6ImNsZ2k5NGE4ZTBpc2IzY2xmemgwdWl5NXkifQ.UQ55c1MEZb-mQV-x66gKIQ',
                 attr='Mapbox Light')

    feature_group = folium.FeatureGroup("Locations")

    for lat, lon, name in zip(list(df['lat']), list(df['lon']), list(df['location'])):
        feature_group.add_child(folium.Circle(location=[lat,lon],
                                            popup=name,
                                            color='red'))

    map.add_child(feature_group)

    return map

import pandas as pd
import numpy as np

def apply_filters(
    df: pd.DataFrame,
    *,
    equals: dict | None = None,        # {"Region": "EMEA"} or {"Region": ["EMEA","APAC"]}
    contains: dict | None = None,      # {"Name": "inc"}  (case-insensitive substring)
    num_ranges: dict | None = None,    # {"Cost": (min_val, max_val)}
    date_ranges: dict | None = None    # {"Date": (start_date, end_date)}
) -> pd.DataFrame:
    """
    Returns a filtered copy of df. Any None/empty filters are ignored.
    - equals: value or list of values -> exact match / isin
    - contains: substring (case-insensitive)
    - num_ranges: inclusive [min, max] (use None to leave one side open)
    - date_ranges: inclusive [start, end] (strings ok; will be to_datetime)
    """
    mask = pd.Series(True, index=df.index)

    # equals / isin
    if equals:
        for col, val in equals.items():
            if val is None or (isinstance(val, (list, tuple, set)) and len(val) == 0):
                continue
            if isinstance(val, (list, tuple, set, np.ndarray)):
                mask &= df[col].isin(val)
            else:
                mask &= df[col].eq(val)

    # contains (case-insensitive)
    if contains:
        for col, substr in contains.items():
            if not substr:
                continue
            mask &= df[col].astype(str).str.contains(str(substr), case=False, na=False)

    # numeric ranges
    if num_ranges:
        for col, (min_v, max_v) in num_ranges.items():
            if min_v is not None:
                mask &= df[col] >= min_v
            if max_v is not None:
                mask &= df[col] <= max_v

    # date ranges
    if date_ranges:
        for col, (start, end) in date_ranges.items():
            # ensure datetime
            series = pd.to_datetime(df[col], errors="coerce")
            if start is not None:
                start = pd.to_datetime(start)
                mask &= series >= start
            if end is not None:
                end = pd.to_datetime(end)
                mask &= series <= end

    return df.loc[mask].copy()

def plot_projects(df, currency=False):
    fig = px.bar(df, x="StartYear", y="metric", text="metric", color_discrete_sequence=["#7B2D26"] )
    fig.update_xaxes(
        tickmode="linear",   # linear tick spacing
        dtick=1              # step size of 1 → only integers
    )
    fig.update_yaxes(visible=False)

    if currency==True:
        fig.update_traces(
                # texttemplate="%{text:.0f}",   # format as integer with commas
                text=[f"{val/1_000_000:,.1f}M" for val in df["metric"]],  # override with M-format
                textposition="outside",
                textfont=dict(size=12)
            )
    else:
        fig.update_traces(
            texttemplate="%{text:,.0f}",  # format with commas
            textposition="outside",
            textfont_size=12,
        )
    fig.update_layout(
    font=dict(size=14),       # fixed font for chart elements
    yaxis=dict(showgrid=False),
    uniformtext_minsize=10,
    xaxis_title=None
    # uniformtext_mode="hide"
    )
    
    return fig

def hex_to_rgba(hex_color: str, alpha: float = 0.3) -> str:
    """Convert hex like '#RRGGBB' to an rgba string with given alpha (0–1)."""
    hex_color = hex_color.lstrip('#')
    r, g, b = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    return f"rgba({r}, {g}, {b}, {alpha})"

def plot_swarm(df, custom_order, category, threshold):

    df["color"] = np.where(
    df["ContractCost"] < threshold,
    "#D3D3D3",   # light gray hex
    df["color"]  # keep original color
)
   

    if category == "Region":
        sort = {category: custom_order}
    else:
        sort = {category: sorted(df[category].dropna().unique())}
   
    categories = df[category].unique().tolist()

    fig = px.strip(
        df,
        x="ContractCost",
        y=category,
        category_orders=sort,
        color="color",  # column of hex codes
        color_discrete_map={c: c for c in df["color"].unique()},
        custom_data=[category, "TypeofWork", "ContractCost", "StartYear", "CompletionYear", "Contractor"]
    )

    fig.update_traces(
        jitter=0.4, 
        marker=dict(size=8, opacity=0.7),
        hovertemplate=(
            "<b>Location:</b> %{customdata[0]}<br>"
            "<b>Type of Work:</b> %{customdata[1]}<br>"
            "<b>Cost:</b> Php %{customdata[2]:,}<br>"
            "<b>Start Year:</b> %{customdata[3]}<br>"
            "<b>Completion Year:</b> %{customdata[4]}<br>"
            "<b>Contractor:</b> %{customdata[5]}"
            "<extra></extra>"
        )
    )
    
    for i, cat in enumerate(categories):
        if i % 2 == 0:  # stripe every other row
            fig.add_hrect(
                y0=i - 0.5, y1=i + 0.5,   # span of category band
                fillcolor="lightgrey",
                opacity=0.2,
                line_width=0,
                layer="below"
            )

    fig.add_vline(
        x=threshold,
        line_width=2,
        line_dash="dot",
        line_color="#0B7A75",
        annotation_text=f"> Php {threshold // 1_000_000}M",
        annotation_position="top"
    )

    fig.update_layout(
        font=dict(size=14),       # fixed font for chart elements
        yaxis=dict(showgrid=False),
        uniformtext_minsize=10,
        xaxis_title="Contract Cost",
        yaxis_title=None,
        showlegend=False,
        height=800,
        hoverlabel=dict(
            font_size=13,          # tooltip font size
            font_color="black",
            bgcolor="white",
            bordercolor="#ccc"
        )
    )
    return fig

def plot_contractors(df, currency=False):

    if currency==True:
        df_sorted = df.sort_values("metric", ascending=True).copy()
        df_sorted["metric_millions"] = df_sorted["metric"] / 1_000_000
        custom_data=df_sorted["metric_millions"]
        template = "%{customdata:,.1f}M"
    else: 
        df_sorted = df.sort_values("metric", ascending=True).copy()
        # df_sorted["metric_billions"] = df_sorted["metric"] / 1_000_000_000
        custom_data=df['metric'].tolist()[::-1]
        template = "%{customdata:,.0f}"

    fig = px.bar(
        df_sorted,
        x="metric",
        y="Contractor",
        orientation="h",
        text="metric",  # placeholder, will override
        category_orders={"Contractor": df_sorted["Contractor"].tolist()[::-1]},
        color_discrete_sequence=["#7B2D26"]
    )

    fig.update_traces(
        texttemplate=template,   # comma + 1 decimal + B suffix
        customdata=custom_data,
        textposition="outside",
        cliponaxis=False,
    )

    fig.update_layout(autosize=True, 
                    height=800,
                    xaxis_title=None,
                    yaxis_title=None,
                    )

    return fig