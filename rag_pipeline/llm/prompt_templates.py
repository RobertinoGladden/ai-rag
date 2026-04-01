from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder

# === RAG QA Prompt ===
RAG_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """Kamu adalah AI assistant yang menjawab pertanyaan berdasarkan konteks dokumen yang diberikan.

ATURAN:
- Jawab HANYA berdasarkan konteks yang disediakan
- Jika jawaban tidak ada di konteks, katakan "Informasi ini tidak tersedia dalam dokumen yang diberikan"
- Selalu sebutkan sumber dokumen jika relevan
- Jawab dalam bahasa yang sama dengan pertanyaan pengguna
- Berikan jawaban yang ringkas, akurat, dan terstruktur

KONTEKS DOKUMEN:
{context}
"""),
    MessagesPlaceholder(variable_name="chat_history", optional=True),
    ("human", "{question}"),
])

# === Standalone Question Prompt (untuk rephrase pertanyaan follow-up) ===
CONDENSE_QUESTION_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """Diberikan riwayat percakapan dan pertanyaan terbaru dari pengguna, 
reformulasikan pertanyaan menjadi pertanyaan mandiri yang bisa dipahami tanpa konteks percakapan sebelumnya.
Jangan jawab pertanyaannya, cukup reformulasikan jika perlu. Jika tidak perlu, kembalikan apa adanya."""),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{question}"),
])

# === Summary Prompt ===
SUMMARY_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """Buat ringkasan dari dokumen berikut. 
Sertakan: poin-poin utama, informasi kunci, dan kesimpulan.
Format: gunakan bullet points untuk keterbacaan yang baik."""),
    ("human", "Dokumen:\n{document}\n\nBuat ringkasan:"),
])
