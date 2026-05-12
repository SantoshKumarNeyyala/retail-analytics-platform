import streamlit as st
from streamlit_option_menu import option_menu

st.set_page_config(page_title="Enterprise Retail Analytics", layout="wide")

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
        icons=["bar-chart", "graph-up", "people", "boxes", "cpu"],
        default_index=0,
    )

if selected == "Executive Overview":
    from pages.executive_overview import show_page

    show_page()

elif selected == "Demand Intelligence":
    from pages.demand_intelligence import show_page

    show_page()

elif selected == "Customer Hub":
    from pages.customer_hub import show_page

    show_page()

elif selected == "Inventory Monitor":
    st.title("📦 Inventory Monitor")
    st.info("Coming in Day 18")

elif selected == "MLOps Monitor":
    st.title("⚙️ MLOps Monitor")
    st.info("Coming in Day 20")
