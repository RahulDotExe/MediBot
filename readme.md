
# 🏥 MedAssistant: AI-Powered Medical Chatbot using RAG

MedAssistant is an AI-powered, real-time medical information assistant that leverages Retrieval-Augmented Generation (RAG) to provide accurate, context-aware responses using only trusted medical resources. This project addresses the critical issue of misinformation and delays in accessing verified medical content by offering a reliable, privacy-focused chatbot solution.

## 🚀 Project Highlights

- ✅ Uses only **verified offline source**: *Gale Encyclopedia of Medicine (PDF)*.
- 🧠 Powered by **state-of-the-art language models**:
    - `gemini-2.5-pro-exp-03-25` (Google Gemini)
    - `deepseek/deepseek-v3-base:free` (HuggingFace)
    - `sentence-transformers/all-MiniLM-L6-v2` (for embedding & similarity search)
- 📦 Deployed via **Streamlit** for a simple, responsive web interface.
- 🔎 Vector similarity search with **FAISS** for fast retrieval of context.
- 🔐 Prioritizes **user privacy**, **speed**, and **accuracy**.

---

## 🧠 Motivation

Many people struggle to find verified and accurate medical information online, leading to delays, confusion, or even harm. MedAssistant offers a solution by combining AI with a trusted medical knowledge base in a user-friendly chatbot interface.

---

## 🛠️ Tech Stack

| Component             | Technology Used                            |
|----------------------|---------------------------------------------|
| Frontend             | Streamlit                                   |
| Backend              | Python, FAISS, Langchain                    |
| Embedding Model      | sentence-transformers/all-MiniLM-L6-v2     |
| Language Models      | Gemini 2.5 Pro, DeepSeek v3 Base           |
| Data Source          | Gale Encyclopedia of Medicine (PDF)         |
| RAG Framework        | Custom RAG with chunked document retrieval  |

---



## ⚙️ How to Run the Project Locally

1. **Clone the repository**
     ```bash
     git clone https://github.com/your-username/MediBot.git
     cd MediBot
     ```

2. **Install dependencies**
     Make sure Python 3.9+ is installed.
     ```bash
     pip install -r requirements.txt
     ```

3. **Add the PDF**
     Place `gale_medical_encyclopedia.pdf` inside the `data/` directory.

4. **Run the Streamlit app**
     ```bash
     streamlit run medibot.py
     ```

---



## 📌 Limitations

- Only as reliable as the source PDF.
- Does not give real-time advice or diagnosis.
- Models depend on token limits (especially Gemini-based).

---

## 💡 Future Enhancements

- Add multi-PDF or full offline database support.
- Integrate audio/text-to-speech support for accessibility.
- Allow model switching or user-controlled temperature settings.
- Add user feedback and correction system.

---

## 🤝 Feedback & Contributions

Have suggestions to improve this chatbot?  
Feel free to open issues or suggest modifications. I’d love to hear from you and make MedAssistant even more helpful!

## 📬 Contact


- 🔗 [LinkedIn](https://www.linkedin.com/in/rahulsays)
- 🐦 [Twitter](https://twitter.com/chillrahull)



## Disclaimer: 
MediBot is for educational use only and does not provide professional medical advice. Always consult a healthcare professional for medical concerns.