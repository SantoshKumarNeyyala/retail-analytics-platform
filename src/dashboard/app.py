import streamlit as st
from streamlit_option_menu import option_menu

from pages import (
    executive_overview,
    demand_intelligence,
    customer_hub,
    inventory_monitor,
    mlops_monitor,
)

# =========================
# PAGE CONFIG
# =========================

st.set_page_config(
    page_title="Enterprise Retail Analytics",
    layout="wide",
)

# =========================
# SIDEBAR MENU
# =========================

with st.sidebar:
    selected = option_menu(
        "Retail Analytics",
        [
            "Executive Overview",
            "Demand Intelligence",
            "Customer Hub",
            "Inventory Monitor",
            "MLOps Monitor",
        ],
        icons=[
            "bar-chart",
            "graph-up",
            "people",
            "boxes",
            "cpu",
        ],
        default_index=0,
    )

# =========================
# PAGE ROUTING
# =========================

if selected == "Executive Overview":
    executive_overview.show_page()

elif selected == "Demand Intelligence":
    demand_intelligence.show_page()

elif selected == "Customer Hub":
    customer_hub.show_page()

elif selected == "Inventory Monitor":
    inventory_monitor.show_page()

elif selected == "MLOps Monitor":
    mlops_monitor.show_page()
