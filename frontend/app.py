import os
import requests
import datetime
import streamlit as st

# Configure page metadata
st.set_page_config(
    page_title="PersonaHire AI | Piyush Bhardwaj Representative",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Backend API Configuration
# Falls back to localhost if not specified in environment
API_BASE_URL = os.getenv("BACKEND_API_URL", "http://localhost:8000")

# Premium Dark Glassmorphic Custom Styling
CUSTOM_CSS = """
<style>
    /* Global Styles */
    body {
        color: #e0e0e0;
        background-color: #0b0c10;
    }
    
    .stApp {
        background: radial-gradient(circle at top right, #1f2833 0%, #0b0c10 80%);
    }

    /* Glassmorphic Cards */
    .glass-card {
        background: rgba(31, 40, 51, 0.45);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        box-shadow: 0 4px 30px rgba(0, 0, 0, 0.5);
    }
    
    .highlight-text {
        color: #66fcf1;
        font-weight: 600;
    }
    
    /* Headings */
    h1, h2, h3 {
        font-family: 'Inter', sans-serif;
        color: #ffffff !important;
        font-weight: 700 !important;
    }
    
    .main-title {
        background: linear-gradient(90deg, #66fcf1 0%, #45a29e 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.8rem !important;
        margin-bottom: 0.5rem !important;
    }

    /* Buttons */
    .stButton>button {
        background-color: #45a29e !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 8px !important;
        transition: all 0.3s ease !important;
        font-weight: 500 !important;
    }
    .stButton>button:hover {
        background-color: #66fcf1 !important;
        color: #0b0c10 !important;
        box-shadow: 0 0 15px rgba(102, 252, 241, 0.4) !important;
        transform: translateY(-2px);
    }
    
    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background-color: #121824 !important;
        border-right: 1px solid rgba(255, 255, 255, 0.05);
    }

    /* Sidebar quick question buttons */
    .suggestion-btn {
        display: block;
        width: 100%;
        background: rgba(69, 162, 158, 0.1);
        border: 1px solid rgba(69, 162, 158, 0.3);
        color: #66fcf1 !important;
        padding: 0.5rem;
        border-radius: 6px;
        margin-bottom: 0.5rem;
        text-align: left;
        font-size: 0.85rem;
        cursor: pointer;
        transition: all 0.2s ease;
    }
    .suggestion-btn:hover {
        background: rgba(102, 252, 241, 0.2);
        border-color: #66fcf1;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ----------------------------------------------------
# Sidebar Details
# ----------------------------------------------------
with st.sidebar:
    st.markdown("<h2 style='text-align: center; color: #ffffff;'>PersonaHire AI</h2>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #45a29e; font-size:0.9rem;'>Piyush Bhardwaj Representative</p>", unsafe_allow_html=True)
    st.markdown("---")
    
    # Representative Status Badge
    st.markdown(
        """
        <div class="glass-card" style="padding:1rem; text-align:center;">
            <p style="margin:0; font-size:0.8rem; color:#888;">AI Representative Status</p>
            <p style="margin:5px 0 0 0; color:#66fcf1; font-weight:bold; font-size:1.1rem;">● ONLINE & GROUNDED</p>
        </div>
        """, 
        unsafe_allow_html=True
    )
    
    # Voice Agent Details
    st.markdown("### 🎙️ Test Voice Representative")
    st.markdown(
        """
        Our representative supports direct phone call screening. To test Vapi voice capabilities:
        1. Access the Vapi config dashboard.
        2. Set backend webhook to this API's public URL.
        3. Experience low-latency (under 2s) speech interactions.
        """
    )
    
    # Quick Links
    st.markdown("### 🔗 Connect with Piyush")
    st.markdown("- 📧 [Email](mailto:piyushbhardwaj634@gmail.com)")
    st.markdown("- 🐙 [GitHub Profile](https://github.com/piyushxbhardwaj)")
    st.markdown("- 💼 [LinkedIn Profile](https://www.linkedin.com/in/piyushxbhardwaj/)")
    
    # Cost Tracker Summary
    st.markdown("---")
    st.markdown("<p style='font-size:0.75rem; color:#666;'>Powered by GPT-4o-mini & ChromaDB Hybrid search.</p>", unsafe_allow_html=True)

# ----------------------------------------------------
# Main Layout
# ----------------------------------------------------
st.markdown("<h1 class='main-title'>PersonaHire AI</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='color:#c0c0c0; margin-top:-10px; font-weight:300;'>AI Representative & Autonomous Interview Scheduling Agent</h4>", unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)

# Define Main Tabs
tab_chat, tab_book, tab_portfolio = st.tabs([
    "💬 Chat Representative", 
    "📅 Schedule Interview", 
    "📄 Candidate Portfolio"
])

# ----------------------------------------------------
# TAB 1: Chat Interface
# ----------------------------------------------------
with tab_chat:
    st.markdown(
        """
        Ask Piyush's representative questions about his **skills, internships, published deepfake research (AetherGuard), 
        projects (MeetMindAI, PictoAI, SplitEase), commit history, or request a meeting schedule.**
        """
    )
    
    # Initialize message history
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Hello! I am Piyush Bhardwaj's AI Representative. I can discuss Piyush's background, education, specific coding projects, commit histories, or guide you to book a calendar interview with him. What would you like to know?"}
        ]
        
    # Generate unique session ID for memory storage
    if "session_id" not in st.session_state:
        st.session_state.session_id = f"session_{int(datetime.datetime.now().timestamp())}"

    # Suggestion Chips
    st.write("💡 **Try asking:**")
    col1, col2, col3 = st.columns(3)
    
    suggested_query = None
    with col1:
        if st.button("What is PictoAI?", key="sug_1"):
            suggested_query = "What is PictoAI?"
    with col2:
        if st.button("Why was ChromaDB selected?", key="sug_2"):
            suggested_query = "Why was ChromaDB selected for PictoAI?"
    with col3:
        if st.button("Tell me about Piyush's education", key="sug_3"):
            suggested_query = "Tell me about Piyush's education, degree, and GPA."

    # Render previous messages
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # User Input
    user_query = st.chat_input("Type your question here...")
    
    # Handle suggested chips click
    if suggested_query:
        user_query = suggested_query

    if user_query:
        # Display user message
        st.session_state.messages.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)
            
        # Call Backend API
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            message_placeholder.markdown("*Searching context and synthesizing answer...*")
            
            try:
                response = requests.post(
                    f"{API_BASE_URL}/api/chat",
                    json={"query": user_query, "session_id": st.session_state.session_id},
                    timeout=20
                )
                if response.status_code == 200:
                    data = response.json()
                    answer = data.get("answer", "")
                    message_placeholder.markdown(answer)
                    # Save assistant message
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                else:
                    err_msg = f"API Error (Status {response.status_code}): {response.text}"
                    message_placeholder.error(err_msg)
            except Exception as e:
                # Local check fallback warning
                message_placeholder.error(
                    f"Could not connect to FastAPI server at {API_BASE_URL}.\n"
                    f"Please check if the backend is running by running `uvicorn backend.main:app` locally.\n"
                    f"Details: {e}"
                )

# ----------------------------------------------------
# TAB 2: Interview Booking Interface
# ----------------------------------------------------
with tab_book:
    st.header("📅 Coordinate Interview Availability")
    st.write(
        """
        This booking form directly interacts with Piyush's Google Calendar. 
        It double-checks availability to prevent overlapping reservations.
        """
    )
    
    col_left, col_right = st.columns([1, 1])
    
    with col_left:
        st.subheader("1. Check Open Availability")
        
        # Pick Date (Limit to today onwards, up to 14 days in future)
        min_date = datetime.date.today()
        max_date = min_date + datetime.timedelta(days=14)
        selected_date = st.date_input("Select Interview Date:", value=min_date, min_value=min_date, max_value=max_date)
        
        date_str = selected_date.isoformat()
        
        # Fetch Slots
        slots = []
        try:
            res = requests.get(f"{API_BASE_URL}/api/calendar/slots?date={date_str}", timeout=10)
            if res.status_code == 200:
                slots = res.json().get("slots", [])
            else:
                st.error("Failed to fetch slots from API.")
        except Exception as e:
            st.warning(f"Backend Server offline at {API_BASE_URL}. Operating with fallback local schedule.")
            # Local fallback simulator for streamlit-only preview
            slots = ["09:00-10:00", "11:00-12:00", "13:00-14:00", "15:00-16:00"]
            
        if not slots:
            st.info(f"No available slots found for {date_str}. Please select another date.")
        else:
            st.success(f"Found {len(slots)} open slots for {date_str}.")
            
    with col_right:
        st.subheader("2. Submit Booking Info")
        
        if slots:
            selected_slot = st.selectbox("Choose a Time Slot:", options=slots)
            recruiter_name = st.text_input("Recruiter Name:", placeholder="Jane Doe")
            recruiter_email = st.text_input("Recruiter Corporate Email:", placeholder="jane.doe@company.com")
            
            if st.button("Confirm Interview Booking"):
                if not recruiter_name or not recruiter_email:
                    st.error("Please fill in both Name and Email to book.")
                else:
                    with st.spinner("Locking calendar slot and creating invitation..."):
                        try:
                            book_res = requests.post(
                                f"{API_BASE_URL}/api/calendar/book",
                                json={
                                    "date": date_str,
                                    "slot": selected_slot,
                                    "name": recruiter_name,
                                    "email": recruiter_email
                                },
                                timeout=15
                            )
                            if book_res.status_code == 200:
                                details = book_res.json().get("event", {})
                                st.markdown(
                                    f"""
                                    <div class="glass-card" style="border-color:#66fcf1; background:rgba(102, 252, 241, 0.05);">
                                        <h4 style="color:#66fcf1 !important; margin:0;">🎉 BOOKING SUCCESSFUL!</h4>
                                        <p style="margin:10px 0 0 0; font-size:0.95rem;">
                                            <b>Interview:</b> {details.get('slot')} on {details.get('date')}<br>
                                            <b>Attendee:</b> {details.get('attendee')} ({details.get('email')})<br>
                                            <b>Event ID:</b> <code style="color:#66fcf1;">{details.get('event_id')}</code><br>
                                            <b>Mode:</b> {details.get('type')}<br>
                                        </p>
                                    </div>
                                    """, 
                                    unsafe_allow_html=True
                                )
                                # Force page rerun to update slot list
                                st.balloons()
                            else:
                                err_details = book_res.json().get("detail", "Slot is already taken.")
                                st.error(f"Booking Conflict: {err_details}")
                        except Exception as e:
                            st.error(f"Error booking interview: {e}. If testing without running backend, this form requires the FastAPI server.")
        else:
            st.info("Select a date on the left that contains open slots to enable booking.")

# ----------------------------------------------------
# TAB 3: Candidate Portfolio & PDF Resume Viewer
# ----------------------------------------------------
with tab_portfolio:
    st.header("📄 Piyush Bhardwaj Portfolio")
    st.write(
        """
        Below is a summary of Piyush's credentials extracted from his official profile.
        """
    )
    
    col_p1, col_p2 = st.columns([1, 1])
    
    with col_p1:
        st.markdown(
            """
            <div class="glass-card">
                <h3>🎓 Education</h3>
                <p>
                    <b>B.E. Computer Science</b><br>
                    Chitkara University (2022 - 2026)<br>
                </p>
                <h3>🏆 Key Certifications</h3>
                <ul>
                    <li>Generative AI - Google Cloud</li>
                    <li>Cyber Security - Cisco</li>
                    <li>Google Analytics - Google</li>
                    <li>AI Foundations - IBM</li>
                </ul>
            </div>
            
            <div class="glass-card">
                <h3>🛠️ Core Technical Skills</h3>
                <p>
                    <b>Languages:</b> Python, Java, C++, SQL, JavaScript<br>
                    <b>Backend:</b> FastAPI, Django, Spring Boot, Node.js, REST APIs<br>
                    <b>Frontend:</b> React, Next.js, HTML5, CSS3, Tailwind CSS<br>
                    <b>AI & Tools:</b> Hugging Face, PyTorch, FAISS, SentenceTransformers, Scikit-Learn, Git & GitHub, Docker, MongoDB, PostgreSQL
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )
        
    with col_p2:
        st.markdown(
            """
            <div class="glass-card">
                <h3>🚀 Featured Projects</h3>
                <p>
                    <b>MeetMindAI:</b> TypeScript meeting summarizer extracting action items.<br>
                    <b>PictoAI:</b> Stable Diffusion Flask image platform using MongoDB caching.<br>
                    <b>AetherGuard:</b> Deep learning deepfake forensics detector published in <i>IJAMRED Journal</i> (May 2026).<br>
                    <b>SplitEase:</b> Next.js 16 bill splitter using Supabase.<br>
                    <b>KnowledgeFlow AI:</b> FAISS + SentenceTransformers semantic search board.
                </p>
            </div>
            
            <div class="glass-card">
                <h3>📄 Offline Resume Source</h3>
                <p>
                    The vector store holds the full parsed resume of Piyush Bhardwaj. 
                    You can inspect or download the original file inside the repository path <code>data/resume.pdf</code>.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )
