# 📝 Automatic Text Summarizer

An offline, free, and lightweight **Automatic Text Summarizer** web application built with **Python**, **Streamlit**, and **NLTK** (Natural Language Toolkit). It condenses long articles, research paragraphs, or documents into concise, informative summaries using **Extractive Summarization**.

---

## 🌟 Key Features

- **Extractive Summarization with NLTK**: Scores and ranks sentences based on normalized word frequency distributions without altering the original sentence structure.
- **Configurable Summary Length**:
  - **Sentence Count Mode**: Choose the exact number of key sentences to retain (e.g., 1 to 15 sentences).
  - **Percentage Mode**: Choose the percentage of the original text to keep (e.g., 20% to 90%).
- **Chronological Flow Preservation**: Extracted sentences are re-ordered according to their original sequence in the text for natural readability.
- **Side-by-Side Comparison**: View the original text and generated summary side-by-side.
- **Context Highlighting**: View the original text with extracted summary sentences highlighted in context.
- **Sentence Scoring Breakdown**: Inspect the individual importance score assigned to each sentence in an interactive table.
- **Word & Sentence Statistics**:
  - Original vs. Summary word count and sentence count.
  - Text reduction / compression percentage.
  - Estimated reading time saved.
- **Top Keywords / Topics**: Automatically extracts and displays the top key topic keywords.
- **1-Click Sample Articles**: Preloaded samples for AI, Renewable Energy, and Space Exploration for rapid testing.
- **File Upload & Export**:
  - Upload `.txt` files directly.
  - Download generated summaries as `.txt` files.
- **100% Free & Local**: No external paid APIs or API keys required. Runs entirely on your local machine.

---

## 🧠 How the Extractive Summarization Algorithm Works

Extractive summarization identifies and selects the most salient sentences directly from the original document.

1. **Sentence & Word Tokenization**: The text is segmented into individual sentences (`sent_tokenize`) and words (`word_tokenize`).
2. **Stopwords & Punctuation Filtering**: Common non-informative words (e.g., *the*, *is*, *at*, *in*) and punctuation marks are removed using NLTK's English stopword corpus.
3. **Word Frequency Calculation**: The occurrence of each informative word is calculated.
4. **Frequency Normalization**: Frequencies are divided by the maximum word frequency so that weights range from `0.0` to `1.0`.
5. **Sentence Scoring**: Each sentence receives a score based on the sum of normalized weights of its constituent words, adjusted for sentence length.
6. **Top-N Selection & Ordering**: Sentences with the highest scores are selected and sorted back into their original document order to preserve the author's logical narrative flow.

---

## 📁 Project Structure

```
Automatic Text Summarizer/
├── app.py              # Main Streamlit web application & summarization logic
├── requirements.txt    # Python dependencies (Streamlit, NLTK)
└── README.md           # Documentation and setup guide
```

---

## 🚀 Installation & Running Locally (Windows & VS Code)

### Prerequisites
- **Python 3.8+** installed on your system.
- **Visual Studio Code (VS Code)** (recommended) or any terminal.

---

### Step 1: Open Project in VS Code
1. Open **VS Code**.
2. Go to **File > Open Folder...** and select the `Automatic Text Summarizer` folder.
3. Open a terminal in VS Code: Press <kbd>Ctrl</kbd> + <kbd>`</kbd> (or go to **Terminal > New Terminal**).

---

### Step 2: (Optional but Recommended) Create a Virtual Environment

In your terminal, run:
```powershell
py -m venv venv
```

Activate the virtual environment:
```powershell
.\venv\Scripts\Activate.ps1
```
*(If PowerShell restricts script execution, run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` first).*

---

### Step 3: Install Required Dependencies

Install Streamlit and NLTK via `requirements.txt`:
```powershell
py -m pip install -r requirements.txt
```

---

### Step 4: Launch the Streamlit App

Run the following command in the terminal:
```powershell
py -m streamlit run app.py
```
*Or if `streamlit` is directly in your PATH:*
```powershell
streamlit run app.py
```

---

### Step 5: Open in Browser
Once launched, Streamlit will automatically open your default browser at:
```
http://localhost:8501
```

---

## 🛠️ Usage Instructions

1. **Enter Text**:
   - Type or paste your long text into the main text area, **OR**
   - Select one of the preloaded sample articles from the sidebar, **OR**
   - Upload a `.txt` file using the file uploader.
2. **Adjust Summary Length**:
   - In the sidebar, choose between **Sentence Count** or **Percentage of Original**.
   - Use the slider to set your desired summary length.
3. **Generate Summary**:
   - Click the **"✨ Generate Summary"** button.
4. **Review Results**:
   - Check the **Statistics & Metrics** (Word count, sentence count, reduction percentage, time saved).
   - Switch between **Side-by-Side Comparison**, **Highlighted Sentences in Context**, and **Sentence Scoring Breakdown**.
5. **Download Summary**:
   - Click **"💾 Download Summary (.txt)"** to save the output locally.

---

## 📦 Requirements

- Python `3.8+`
- `streamlit >= 1.30.0`
- `nltk >= 3.8.0`

---

## 📄 License
This project is open-source and free to use for educational and personal projects.
