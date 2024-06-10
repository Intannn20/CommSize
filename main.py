import streamlit as st
from streamlit_option_menu import option_menu
import pages.home as home
import pages.application as application
import pages.about as about
from pathlib import Path


def main():
    st.set_page_config(page_title="Foot Measurement App", layout="wide")

    # Load CSS
    st.markdown(
        f'<style>{(Path(__file__).parent / "styles.css").read_text()}</style>',
        unsafe_allow_html=True,
    )

    # Navbar di sidebar
    with st.sidebar:
        selected = option_menu(
            "Main Menu",
            ["Home", "Application", "About"],
            icons=["house", "app-indicator", "info-circle"],
            menu_icon="cast",
            default_index=0,
        )

    # Mengatur halaman yang sesuai
    if selected == "Home":
        home.show()
    elif selected == "Application":
        application.show()
    elif selected == "About":
        about.show()


if __name__ == "__main__":
    main()
