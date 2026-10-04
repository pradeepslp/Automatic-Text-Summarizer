import streamlit as st
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
import string
import heapq
import re

# -----------------------------------------------------------------------------
# Page Configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Automatic Text Summarizer",
    page_icon="📝",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------------------------------------------------------
# Custom Styling (Clean, Modern Aesthetic)
# -----------------------------------------------------------------------------
st.markdown("""
<style>
    /* Global Styles & Font Adjustments */
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
    
    /* Metrics Box */
    .metric-card {
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 1rem 1.25rem;
        text-align: center;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #2563EB;
    }
    .metric-label {
        font-size: 0.85rem;
        font-weight: 600;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    /* Text Display Cards */
    .text-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 1.25rem;
        min-height: 250px;
        font-size: 1rem;
        line-height: 1.65;
        color: #334155;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
    }
    
    .summary-card {
        background: #F0FDF4;
        border: 1px solid #BBF7D0;
        border-radius: 12px;
        padding: 1.25rem;
        min-height: 250px;
        font-size: 1.02rem;
        line-height: 1.7;
        color: #166534;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
    }
    
    .highlighted-text {
        background-color: #FEF08A;
        font-weight: 500;
        padding: 2px 4px;
        border-radius: 4px;
    }
    
    /* Badge tags */
    .badge {
        display: inline-block;
        padding: 0.25rem 0.6rem;
        font-size: 0.75rem;
        font-weight: 600;
        border-radius: 9999px;
        background-color: #EFF6FF;
        color: #1D4ED8;
        margin-right: 0.5rem;
        margin-bottom: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# Initialize NLTK Resources (Cached for performance)
# -----------------------------------------------------------------------------
@st.cache_resource
def setup_nltk():
    """Ensure required NLTK corpora are downloaded."""
    packages = ["punkt", "punkt_tab", "stopwords"]
    for pkg in packages:
        try:
            nltk.download(pkg, quiet=True)
        except Exception as e:
            st.warning(f"Note: Could not automatically verify NLTK package '{pkg}': {e}")

setup_nltk()


# -----------------------------------------------------------------------------
# Sample Texts for Quick Testing
# -----------------------------------------------------------------------------
SAMPLE_TEXTS = {
    "Artificial Intelligence (AI)": (
        "Artificial Intelligence (AI) is transforming the modern world at an unprecedented pace. "
        "From healthcare and education to finance and transportation, AI technologies are revolutionizing the way humans live and work. "
        "Machine learning algorithms can analyze vast amounts of data to identify patterns and make predictions with incredible accuracy. "
        "In medicine, AI systems assist doctors in diagnosing diseases earlier and discovering life-saving drugs in record time. "
        "Autonomous vehicles leverage computer vision and deep neural networks to navigate city roads safely. "
        "However, the rapid advancement of AI also presents significant ethical and societal challenges, such as algorithmic bias, job displacement, and data privacy concerns. "
        "As intelligent systems become more capable, governments and industry leaders must collaborate to establish clear regulatory frameworks. "
        "Ensuring the responsible and ethical use of artificial intelligence will remain one of humanity's most critical responsibilities in the coming decades."
    ),
    "Renewable Energy & Climate": (
        "The global transition toward renewable energy is accelerating rapidly as nations strive to combat climate change. "
        "Solar and wind power have emerged as the most cost-effective sources of new electricity generation across many regions worldwide. "
        "Advances in energy storage technology, particularly lithium-ion and solid-state batteries, are solving the intermittency challenges of solar and wind. "
        "Governments are enacting ambitious policies and providing tax incentives to decarbonize transportation and heavy industries. "
        "Electric vehicles have gained widespread adoption, significantly reducing reliance on fossil fuels. "
        "Despite this remarkable momentum, substantial investments in modernizing the power grid and expanding transmission capacity remain essential. "
        "A diversified energy mix combining solar, wind, hydro, and geothermal energy will be critical to achieving a sustainable zero-carbon future."
    ),
    "Space Exploration & Mars": (
        "Humanity stands on the verge of a new era in interplanetary exploration. "
        "Space agencies and private aerospace enterprises are developing next-generation heavy-lift rockets capable of transporting crew and cargo to deep space. "
        "NASA's Artemis program aims to return humans to the Moon and establish a sustainable lunar research base by the end of the decade. "
        "The Moon will serve as a vital testing ground for technologies and life-support systems needed for future crewed missions to Mars. "
        "Robotic rovers on Mars, such as Perseverance and Curiosity, continue to uncover compelling geological evidence of past water and potential habitable environments. "
        "However, long-duration human spaceflight presents formidable obstacles, including prolonged exposure to cosmic radiation and the psychological strains of isolation. "
        "Overcoming these challenges will require groundbreaking innovations in propulsion, habitat design, and closed-loop life support systems."
    )
}


# -----------------------------------------------------------------------------
# Extractive Summarization Core Engine
# -----------------------------------------------------------------------------
def clean_word(word: str) -> str:
    """Normalize word by stripping punctuation and converting to lowercase."""
    return word.strip(string.punctuation).lower()


def extractive_summarize(text: str, target_sentences: int):
    """
    Summarize text using NLTK sentence scoring based on normalized word frequencies.
    
    Returns:
        dict: Contains summary text, sentence list, scores, metrics, and top keywords.
    """
    # 1. Tokenize into sentences
    raw_sentences = sent_tokenize(text.strip())
    total_sentences = len(raw_sentences)
    
    if total_sentences == 0:
        return None
    
    # Handle single sentence or small texts where target exceeds total
    num_to_extract = min(max(1, target_sentences), total_sentences)
    
    # 2. Tokenize words and filter stopwords & punctuation
    stop_words = set(stopwords.words("english"))
    words = word_tokenize(text)
    
    # 3. Calculate word frequencies
    word_frequencies = {}
    for word in words:
        w_clean = clean_word(word)
        if w_clean and len(w_clean) > 1 and w_clean not in stop_words and not w_clean.isnumeric():
            word_frequencies[w_clean] = word_frequencies.get(w_clean, 0) + 1
            
    if not word_frequencies:
        # Fallback if no content words found (e.g. only stop words or numbers)
        selected_indices = list(range(num_to_extract))
        summary_sentences = [raw_sentences[i] for i in selected_indices]
        return {
            "summary_text": " ".join(summary_sentences),
            "original_sentences": raw_sentences,
            "selected_indices": set(selected_indices),
            "sentence_scores": {i: 0.0 for i in range(total_sentences)},
            "top_keywords": [],
            "total_sentences": total_sentences,
            "summary_sentence_count": len(summary_sentences)
        }
    
    # 4. Normalize word frequencies (scale from 0 to 1)
    max_freq = max(word_frequencies.values())
    normalized_frequencies = {w: freq / max_freq for w, freq in word_frequencies.items()}
    
    # Top keywords for display
    top_keywords = sorted(word_frequencies.items(), key=lambda x: x[1], reverse=True)[:6]
    
    # 5. Score each sentence based on normalized word weights
    sentence_scores = {}
    for idx, sentence in enumerate(raw_sentences):
        sentence_words = word_tokenize(sentence)
        score = 0.0
        word_count = 0
        
        for word in sentence_words:
            w_clean = clean_word(word)
            if w_clean in normalized_frequencies:
                score += normalized_frequencies[w_clean]
                word_count += 1
                
        # Length normalization: avoid disproportionately penalizing short or favoring overly long sentences
        if word_count > 0:
            sentence_scores[idx] = score / (1.0 + (word_count ** 0.5))
        else:
            sentence_scores[idx] = 0.0
            
    # 6. Select top N sentences with highest scores
    top_sentence_indices = heapq.nlargest(num_to_extract, sentence_scores, key=sentence_scores.get)
    
    # Re-sort indices chronologically to preserve natural document flow
    top_sentence_indices.sort()
    
    selected_set = set(top_sentence_indices)
    summary_sentences = [raw_sentences[i] for i in top_sentence_indices]
    summary_text = " ".join(summary_sentences)
    
    return {
        "summary_text": summary_text,
        "original_sentences": raw_sentences,
        "selected_indices": selected_set,
        "sentence_scores": sentence_scores,
        "top_keywords": [kw[0].capitalize() for kw in top_keywords],
        "total_sentences": total_sentences,
        "summary_sentence_count": len(summary_sentences)
    }


def count_words(text: str) -> int:
    """Return the total word count in a text string."""
    return len(re.findall(r'\b\w+\b', text))


# -----------------------------------------------------------------------------
# Sidebar: Settings & Controls
# -----------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Sample Text Loader
    st.subheader("📋 Sample Texts")
    sample_choice = st.selectbox(
        "Load an example article:",
        ["None (Custom Input)"] + list(SAMPLE_TEXTS.keys()),
        index=0,
        help="Select a sample paragraph to test summarization immediately."
    )
    
    st.divider()
    
    # Summary Length Controls
    st.subheader("📏 Summary Length")
    length_mode = st.radio(
        "Length Selection Mode:",
        ["Sentence Count", "Percentage of Original"],
        index=0,
        help="Choose whether to select the exact sentence count or a percentage of the original text."
    )
    
    if length_mode == "Sentence Count":
        target_sentences = st.slider(
            "Number of Sentences:",
            min_value=1,
            max_value=15,
            value=3,
            step=1,
            help="Specify how many key sentences to extract."
        )
        reduction_percentage = None
    else:
        reduction_percentage = st.slider(
            "Summary Size (% of text):",
            min_value=10,
            max_value=90,
            value=40,
            step=5,
            help="Specify the target percentage of sentences to retain."
        )
        target_sentences = None
        
    st.divider()
    
    # How It Works Info Box
    with st.expander("ℹ️ How Extractive Summarization Works", expanded=False):
        st.markdown("""
        **Extractive Summarization** works by identifying and extracting the most important sentences directly from the original text without modifying them.
        
        **NLTK Pipeline:**
        1. **Sentence Tokenization**: Breaks article into individual sentences.
        2. **Word Cleaning**: Removes common stopwords (the, is, at) and punctuation.
        3. **Frequency Scoring**: Scores words based on how often they appear.
        4. **Sentence Ranking**: Sums normalized word weights for each sentence.
        5. **Re-ordering**: Selects the highest-scoring sentences and orders them chronologically for natural readability.
        """)
        
    st.caption("Powered by Python, NLTK & Streamlit • Offline & Free")


# -----------------------------------------------------------------------------
# Main Application Content
# -----------------------------------------------------------------------------
st.markdown('<div class="main-header">📝 Automatic Text Summarizer</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Summarize articles, research paragraphs, or documents into concise key points using NLTK Extractive Summarization.</div>', unsafe_allow_html=True)

# Determine default input value based on sample text selection
default_input = ""
if sample_choice != "None (Custom Input)":
    default_input = SAMPLE_TEXTS[sample_choice]

# File Upload Option (Optional helper for users)
with st.expander("📂 Or upload a .txt file", expanded=False):
    uploaded_file = st.file_uploader("Choose a text file", type=["txt"])
    if uploaded_file is not None:
        try:
            default_input = uploaded_file.read().decode("utf-8")
            st.success(f"Loaded file: {uploaded_file.name}")
        except Exception as err:
            st.error(f"Error reading file: {err}")

# Text Area Input
user_input = st.text_area(
    "Enter or paste your text below:",
    value=default_input,
    height=220,
    placeholder="Paste a long paragraph, article, or document here to generate an extractive summary..."
)

col_btn1, col_btn2, _ = st.columns([1.5, 1.5, 5])
with col_btn1:
    summarize_clicked = st.button("✨ Generate Summary", type="primary", use_container_width=True)
with col_btn2:
    clear_clicked = st.button("🗑️ Clear Text", use_container_width=True)

if clear_clicked:
    st.rerun()

st.divider()

# -----------------------------------------------------------------------------
# Processing & Summarization Logic
# -----------------------------------------------------------------------------
trimmed_text = user_input.strip() if user_input else ""

if not trimmed_text:
    st.info("💡 **Getting Started:** Enter a paragraph above, select a sample from the sidebar, or upload a text file, then click **'Generate Summary'**.")
else:
    # Check sentence count in input
    prelim_sentences = sent_tokenize(trimmed_text)
    total_input_sentences = len(prelim_sentences)
    orig_words = count_words(trimmed_text)
    
    # Calculate target sentence count
    if length_mode == "Percentage of Original":
        calc_target = max(1, round(total_input_sentences * (reduction_percentage / 100.0)))
    else:
        calc_target = target_sentences

    # Handle very short text warning
    if total_input_sentences <= 1:
        st.warning(
            f"⚠️ **Short Text Notice:** Your text contains only **{total_input_sentences} sentence** ({orig_words} words). "
            "Extractive summarization is designed for multi-sentence paragraphs. The text is shown below as-is."
        )
    elif calc_target >= total_input_sentences:
        st.info(
            f"ℹ️ **Note:** The requested summary length ({calc_target} sentences) is greater than or equal to the total sentences in the text ({total_input_sentences} sentences). All sentences are retained."
        )

    # Perform summarization
    result = extractive_summarize(trimmed_text, target_sentences=calc_target)
    
    if result:
        summary_text = result["summary_text"]
        summary_words = count_words(summary_text)
        summary_sentences_count = result["summary_sentence_count"]
        
        # Calculate reduction percentage
        if orig_words > 0:
            reduction_pct = max(0, round((1.0 - (summary_words / orig_words)) * 100, 1))
        else:
            reduction_pct = 0.0

        # Estimated reading time (average 200 words per minute)
        orig_time_sec = max(1, round((orig_words / 200) * 60))
        summ_time_sec = max(1, round((summary_words / 200) * 60))
        time_saved_sec = max(0, orig_time_sec - summ_time_sec)

        # ---------------------------------------------------------------------
        # Statistics & Metrics Cards
        # ---------------------------------------------------------------------
        st.subheader("📊 Summary Statistics")
        m_col1, m_col2, m_col3, m_col4 = st.columns(4)
        
        with m_col1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{orig_words}</div>
                <div class="metric-label">Original Words ({total_input_sentences} Sentences)</div>
            </div>
            """, unsafe_allow_html=True)
            
        with m_col2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{summary_words}</div>
                <div class="metric-label">Summary Words ({summary_sentences_count} Sentences)</div>
            </div>
            """, unsafe_allow_html=True)
            
        with m_col3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value" style="color: #059669;">{reduction_pct}%</div>
                <div class="metric-label">Text Reduction</div>
            </div>
            """, unsafe_allow_html=True)
            
        with m_col4:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value" style="color: #7C3AED;">~{time_saved_sec}s</div>
                <div class="metric-label">Est. Reading Time Saved</div>
            </div>
            """, unsafe_allow_html=True)

        st.write("")

        # Display Top Keywords identified in text
        if result["top_keywords"]:
            st.markdown("**🏷️ Top Identified Keywords / Topics:**")
            badges_html = "".join([f'<span class="badge">📌 {kw}</span>' for kw in result["top_keywords"]])
            st.markdown(badges_html, unsafe_allow_html=True)

        st.write("")

        # ---------------------------------------------------------------------
        # Side-by-Side and Detailed View Tabs
        # ---------------------------------------------------------------------
        tab1, tab2, tab3 = st.tabs(["📄 Side-by-Side Comparison", "🔍 Highlighted Sentences in Context", "📊 Sentence Scoring Breakdown"])
        
        with tab1:
            col_orig, col_summ = st.columns(2)
            with col_orig:
                st.markdown("### 📖 Original Text")
                st.markdown(f'<div class="text-card">{trimmed_text}</div>', unsafe_allow_html=True)
                
            with col_summ:
                st.markdown("### 🎯 Summarized Text")
                st.markdown(f'<div class="summary-card">{summary_text}</div>', unsafe_allow_html=True)
                
                # Download summary button
                st.download_button(
                    label="💾 Download Summary (.txt)",
                    data=summary_text,
                    file_name="summary.txt",
                    mime="text/plain",
                    use_container_width=True
                )
                
        with tab2:
            st.markdown("### 🔍 Extracted Sentences Highlighted in Original Context")
            st.caption("Highlighted sentences in yellow were selected as the most informative by the NLTK scoring engine.")
            
            highlighted_html_parts = []
            for idx, sent in enumerate(result["original_sentences"]):
                if idx in result["selected_indices"]:
                    highlighted_html_parts.append(f'<span class="highlighted-text">{sent}</span>')
                else:
                    highlighted_html_parts.append(f'<span>{sent}</span>')
                    
            highlighted_doc = " ".join(highlighted_html_parts)
            st.markdown(f'<div class="text-card">{highlighted_doc}</div>', unsafe_allow_html=True)
            
        with tab3:
            st.markdown("### 📊 Sentence Importance Scores")
            st.caption("Each sentence is ranked based on the normalized frequencies of the key words it contains.")
            
            score_data = []
            for idx, sent in enumerate(result["original_sentences"]):
                is_selected = "✅ Included in Summary" if idx in result["selected_indices"] else "⚪ Excluded"
                score = result["sentence_scores"].get(idx, 0.0)
                score_data.append({
                    "Index": idx + 1,
                    "Sentence": sent,
                    "Score": round(score, 4),
                    "Status": is_selected
                })
                
            st.dataframe(
                score_data,
                use_container_width=True,
                column_config={
                    "Index": st.column_config.NumberColumn(width="small"),
                    "Sentence": st.column_config.TextColumn(width="large"),
                    "Score": st.column_config.NumberColumn(format="%.4f", width="small"),
                    "Status": st.column_config.TextColumn(width="medium")
                }
            )
