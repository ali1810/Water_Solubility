import streamlit as st

# Initialize session state if not already set
if 'footer_button_clicked' not in st.session_state:
    st.session_state['footer_button_clicked'] = None

# Function to render footer with buttons
def render_footer():
    footer_css = """
    <style>
    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background-color: #f1f1f1;
        color: black;
        text-align: center;
        padding: 10px;
        box-shadow: 0 -1px 3px rgba(0,0,0,0.2);
        z-index: 1000;
    }
    .footer .stButton button {
        margin: 0 15px;
        border: none;
        background-color: transparent;
        color: black;
        padding: 8px 16px;
        font-size: 16px;
        cursor: pointer;
        border-radius: 5px;
    }
    .footer .stButton button:hover {
        color: gray;
    }
    </style>
    <div class="footer">
        <div class="stButton">
            <button onclick="sendMessage('about')">About</button>
            <button onclick="sendMessage('directive')">Directive</button>
            <button onclick="sendMessage('data_privacy')">Data Privacy</button>
        </div>
    </div>
    <script>
    function sendMessage(buttonId) {
        const message = {type: 'footer_button_click', buttonId: buttonId};
        window.parent.postMessage(message, '*');
    }
    window.addEventListener('message', (event) => {
        if (event.data.type === 'footer_button_click') {
            const buttonId = event.data.buttonId;
            Streamlit.setComponentValue(buttonId);
        }
    });
    </script>
    """
    st.markdown(footer_css, unsafe_allow_html=True)

# Display footer with buttons
#render_footer()

# Handle footer button click
if st.session_state['footer_button_clicked']:
    if st.session_state['footer_button_clicked'] == 'about':
        st.write("About content: This is the about section of the website.")
    elif st.session_state['footer_button_clicked'] == 'directive':
        st.write("Directive content: This is the directive section of the website.")
    elif st.session_state['footer_button_clicked'] == 'data_privacy':
        st.write("Data Privacy content: This is the data privacy section of the website.")
else:
    st.write("Welcome! Please click a footer button to display content.")

# Function to set the session state from JavaScript message