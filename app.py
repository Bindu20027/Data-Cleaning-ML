import streamlit as st

def add_bg_design():
    st.markdown(
        """
        <style>
        /* Background image */
        .stApp {
            background-image: url('https://images.unsplash.com/photo-1504384308090-c894fdcc538d');
            background-size: cover;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }

        /* Glass card effect for all content */
        .block-container {
            background: rgba(255, 255, 255, 0.15);
            border-radius: 15px;
            padding: 20px;
            backdrop-filter: blur(10px);
            -webkit-backdrop-filter: blur(10px);
            border: 2px solid rgba(255, 255, 255, 0.4);
        }

        /* Text styling */
        h1, h2, h3, p, label, span {
            color: white !important;
        }

        /* Buttons */
        .stButton>button {
            background-color: rgba(255, 255, 255, 0.8);
            color: black;
            border-radius: 10px;
            padding: 0.6rem 1.2rem;
            border: none;
            font-weight: bold;
            transition: 0.3s;
        }
        .stButton>button:hover {
            background-color: rgba(255, 255, 255, 1);
            transform: scale(1.05);
        }
        </style>
        """,
        unsafe_allow_html=True
    )

# Activate the design
add_bg_design()
