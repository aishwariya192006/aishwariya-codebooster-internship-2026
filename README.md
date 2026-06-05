# Day 10 — Student Reference Guide
## GenAI + Data Engineering Integration + Capstone
### Codeboosters Tech — Data Engineering + GenAI Internship

---

> **This is your final internship reference document. Keep it. It is your cheat sheet for building AI Data systems.**

---

## Quick Revision — What You Learned in 10 Days

| Day | Topic | Core Skill |
|-----|-------|-----------|
| Day 1 | Data Engineering Basics | Pandas, CSV, JSON, GitHub |
| Day 2 | SQL + Visualization | SQLite, SQL queries, Matplotlib |
| Day 3 | ETL + APIs | Data cleaning, Requests, JSON parsing |
| Day 4 | Big Data | PySpark, Medallion Architecture |
| Day 5 | ML Basics | Regression, Feature Engineering, Sklearn |
| Day 6 | GenAI + LLMs | Groq API, Prompt Engineering |
| Day 7 | Embeddings | Sentence Transformers, ChromaDB |
| Day 8 | RAG Systems | Retrieval-Augmented Generation |
| Day 9 | AI Agents | Tool calling, Multi-step workflows |
| Day 10 | **Integration + Capstone** | **All of the above combined** |

---

## Section 1 — Responsible AI Quick Reference

### What is AI Hallucination?

**Simple definition:** AI generates text that sounds confident but is factually incorrect.

**Why it happens:** LLMs predict the next word using statistical patterns. They do not look up verified facts. When uncertain, they generate plausible-sounding content that may be false.

**How to prevent it:**
- Use RAG — give AI real documents to read before answering
- Add guardrail instructions to your system prompt
- Tell the AI: "If you do not know, say so clearly"
- Validate AI answers against trusted data sources

**Your guardrail template:**
```python
system_prompt = """
You are a [role].
Only answer based on the data or documents provided to you.
If you are not certain, say: I do not have enough information to answer accurately.
Never invent statistics or facts.
"""
```

---

### What is AI Bias?

**Definition:** AI gives systematically unfair results based on gender, race, region, religion, or other protected attributes.

**Root cause:** Training data reflects historical human biases.

**Real example:** Amazon's 2018 hiring AI rejected female resumes because it was trained on historically male-dominated hiring data.

**How to detect it:** Test your AI output across different demographic groups. If results differ unfairly, your data or model is biased.

**How to reduce it:** Use diverse, representative training data. Audit model outputs regularly. Document your data sources.

---

### What is Data Privacy in AI?

**Definition:** AI systems must not expose, misuse, or retain personal information without proper consent.

**The Samsung incident (2023):** Employees pasted proprietary code into ChatGPT. That code became part of training data. Samsung banned ChatGPT company-wide as a result.

**Rules to follow:**
- Never paste real personal data (names, phone numbers, health records) into commercial AI APIs
- Anonymize data before sending to any external AI service
- Use access controls — not everyone should see all data
- Know your country's data laws (India: DPDP Act 2023)

---

## Section 2 — GenAI + Data Engineering Integration

### The End-to-End AI Data System Architecture

```
RAW DATA SOURCES
(CSV, APIs, Databases)
       |
       v
DATA ENGINEERING LAYER
(Pandas ETL, Cleaning, SQL, PySpark)
       |
       v
DATA STORAGE LAYER
(SQLite tables + ChromaDB vectors)
       |
       v
AI REASONING LAYER
(Groq LLM + RAG + Embeddings)
       |
       v
USER-FACING APPLICATION
(Chatbot, Report, Search System)
```

**Key insight:** Remove any one layer and the system fails.
- No Data Engineering = AI gets dirty, unreliable data
- No Storage = No persistence between requests
- No AI Layer = Just a traditional data analysis tool
- No Interface = No one can use it

---

### Your 10-Day Skills Map to This Architecture

| Architecture Layer | Your Skills (Days) |
|-------------------|-------------------|
| Raw Data Sources | CSV/JSON loading, API calls (D1, D3) |
| Data Engineering | Pandas ETL, data cleaning (D1, D3) |
| Structured Storage | SQLite + SQL queries (D2) |
| Vector Storage | ChromaDB embeddings (D7) |
| AI Reasoning | Groq API + RAG + Agents (D6, D8, D9) |
| Application | Python output + prompt results (D6-D10) |

---

## Section 3 — Syntax Cheat Sheet

### Pandas — Most Used Commands

```python
# Load data
df = pd.read_csv("student_performance.csv")

# Basic exploration
df.head(5)              # First 5 rows
df.shape                # (rows, columns)
df.columns              # Column names
df.dtypes               # Data types
df.describe()           # Statistical summary

# Data quality
df.isnull().sum()       # Missing values per column
df.duplicated().sum()   # Count duplicate rows
df.drop_duplicates()    # Remove duplicates

# Filtering
df[df['gpa'] > 8.0]                        # Filter by condition
df[(df['branch'] == 'CSE') & (df['gpa'] > 7)] # Multiple conditions

# Aggregation
df['gpa'].mean()                           # Average GPA
df.groupby('branch')['gpa'].mean()         # Average GPA per branch
df['passed'].value_counts()                # Count per category

# Cleaning
df['col'].fillna(df['col'].median())       # Fill missing with median
df['col'].clip(lower=0, upper=100)         # Cap to valid range
df['col'].str.strip().str.title()          # Clean text formatting

# Export
df.to_csv("output.csv", index=False)       # Save to CSV
```

---

### SQLite — Most Used Commands

```python
import sqlite3
import pandas as pd

# Setup
conn = sqlite3.connect(':memory:')         # In-memory database
df.to_sql('students', conn, if_exists='replace', index=False)

# Run queries
result_df = pd.read_sql("SELECT * FROM students", conn)

# Common SQL patterns
"""
-- Average by group
SELECT branch, ROUND(AVG(gpa), 2) AS avg_gpa
FROM students
GROUP BY branch
ORDER BY avg_gpa DESC

-- Filter rows
SELECT name, gpa FROM students
WHERE gpa > 8.0 AND passed = 'Yes'

-- Top N results
SELECT name, gpa FROM students
ORDER BY gpa DESC
LIMIT 5

-- Count per category
SELECT passed, COUNT(*) AS total
FROM students
GROUP BY passed
"""
```

---

### Groq API — Most Used Patterns

```python
from groq import Groq

# Setup
client = Groq(api_key="gsk_your_key_here")
MODEL = "llama-3.1-8b-instant"

# Basic call
response = client.chat.completions.create(
    model=MODEL,
    max_tokens=500,
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user",   "content": "Your question here"}
    ]
)
answer = response.choices[0].message.content

# With data context (the core pattern)
data_summary = f"Key metrics: {some_variable}"
user_prompt  = f"Data:\n{data_summary}\n\nQuestion: What are the key insights?"

# Multi-turn conversation
messages = [
    {"role": "system",    "content": system_prompt},
    {"role": "user",      "content": first_question},
    {"role": "assistant", "content": first_answer},      # Previous AI reply
    {"role": "user",      "content": follow_up_question} # New question
]
```

---

### ChromaDB + RAG — Most Used Patterns

```python
import chromadb
from sentence_transformers import SentenceTransformer

# Setup
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
chroma_client   = chromadb.EphemeralClient()
collection      = chroma_client.get_or_create_collection(name="my_kb")

# Add documents
embedding = embedding_model.encode("Document text").tolist()
collection.add(
    documents=["Full document text"],
    embeddings=[embedding],
    metadatas=[{"topic": "SQL", "subject": "Database"}],
    ids=["doc_001"]
)

# Search (semantic query)
question_embedding = embedding_model.encode("My question").tolist()
results = collection.query(
    query_embeddings=[question_embedding],
    n_results=3                          # Return top 3 most relevant docs
)

retrieved_docs = results['documents'][0]  # List of matching document texts
retrieved_meta = results['metadatas'][0]  # List of matching metadata dicts

# Build RAG prompt
context = "\n\n".join(retrieved_docs)
rag_prompt = f"Context:\n{context}\n\nQuestion: {my_question}"
```

---

## Section 4 — Capstone Quick Reference

### Capstone 01 — AI Data Analyst: Summary Steps

```
1. pd.read_csv("student_performance.csv")
2. sqlite3.connect(':memory:') → df.to_sql() → pd.read_sql(query)
3. Build data_summary string using f-strings
4. client.chat.completions.create(model, messages=[system, user+data])
5. Print AI response → save to .txt file
```

**Key prompt pattern:**
```python
system = "You are an academic advisor. Only use the data provided."
user   = f"Data:\n{data_summary}\n\nProvide 3 insights and 1 recommendation."
```

---

### Capstone 02 — College Knowledge Assistant: Summary Steps

```
1. pd.read_csv("college_notes.csv")
2. chromadb.EphemeralClient() → get_or_create_collection()
3. Loop through notes → encode each → collection.add()
4. student_question → encode → collection.query(n_results=3)
5. Build RAG prompt with retrieved docs → Groq API call → print answer
```

**Key prompt pattern:**
```python
system = "You are a professor. Answer only from the provided notes."
user   = f"Notes:\n{context}\n\nStudent Question: {question}"
```

---

### Capstone 03 — AI Data Cleaner: Summary Steps

```
1. pd.read_csv("student_performance.csv")
2. df.isnull().sum() + df.duplicated().sum() + range checks → quality report
3. Build quality_report string → Groq API: "What cleaning steps do you recommend?"
4. Apply: df.drop_duplicates() + df.fillna() + df.clip() + df.str.strip()
5. df.to_csv("cleaned.csv", index=False)
```

**Key prompt pattern:**
```python
system = "You are a Data Quality Engineer. Recommend specific Pandas cleaning steps."
user   = f"Quality Report:\n{quality_report}\n\nWhat are the top 3 cleaning actions?"
```

---

## Section 5 — Common Errors and Fixes

| Error | Cause | Fix |
|-------|-------|-----|
| `AuthenticationError: Invalid API Key` | Wrong or missing Groq key | Check key starts with `gsk_`. No extra spaces. |
| `FileNotFoundError: student_performance.csv` | File not uploaded to Colab | Upload via sidebar folder icon. Use `/content/student_performance.csv` |
| `ModuleNotFoundError: No module named chromadb` | Package not installed | Run `!pip install chromadb sentence-transformers` then restart runtime |
| `Collection already exists` | Running cell twice without reset | Use `get_or_create_collection()` instead of `create_collection()` |
| `RateLimitError` | Too many API calls | Wait 30 seconds. Reduce `max_tokens`. Free tier: ~14,400 tokens/min |
| `None` or empty AI response | Bad messages format | Check `messages` list has at least one `{"role": "user"}` item |
| `SettingWithCopyWarning` | Unsafe Pandas assignment | Use `df.loc[mask, 'col'] = value` or `df = df.copy()` first |
| Colab disconnects mid-install | Free tier timeout | Re-run pip install. Keep tab active. Use `!pip install -q` for faster installs |

---

## Section 6 — Career Roadmap

### Three Career Paths After This Internship

#### Path 1 — Data Engineer
**Timeline:** 0-6 months to first role  
**Key skills to add:** Apache Airflow, dbt, Cloud platforms (AWS/GCP), Docker  
**Starting salary (India):** Rs 6-12 LPA  
**Job titles:** Junior Data Engineer, ETL Developer, Data Pipeline Engineer

#### Path 2 — AI/GenAI Application Developer
**Timeline:** 3-12 months to first role  
**Key skills to add:** LangChain, FastAPI, LlamaIndex, API integration patterns  
**Starting salary (India):** Rs 8-20 LPA  
**Job titles:** AI Developer, LLM Engineer, GenAI Application Engineer

#### Path 3 — ML Engineer
**Timeline:** 12-24 months to first role (more learning required)  
**Key skills to add:** Deep learning (PyTorch), model training, MLOps, cloud ML platforms  
**Starting salary (India):** Rs 12-30 LPA  
**Job titles:** ML Engineer, AI Engineer, Applied Scientist

---

### Your 7-Day Action Plan (Start Today)

- [ ] **Day 1:** Create GitHub account. Push all 10 internship notebooks. Name the repo: `codeboosters-internship`
- [ ] **Day 2:** Write a LinkedIn post describing your capstone project. Tag your college and trainer.
- [ ] **Day 3:** Enroll in one free course — IBM Data Engineering (Coursera) or DeepLearning.AI Prompt Engineering
- [ ] **Day 4:** Create a Kaggle account. Explore beginner competitions.
- [ ] **Day 5:** Extend your capstone with one new feature (new dataset, new question, or better UI)
- [ ] **Day 6:** Connect on LinkedIn with every classmate from this internship
- [ ] **Day 7:** Apply to one data engineering or AI internship/job posting

---

### Recommended Learning Resources

| Resource | Topic | Cost |
|----------|-------|------|
| Coursera — IBM Data Engineering Certificate | Full data engineering path | Free audit |
| DeepLearning.AI — Prompt Engineering (Andrew Ng) | Advanced prompting | Free |
| Fast.ai — Practical Deep Learning | ML and deep learning | Free |
| Google ML Crash Course | ML fundamentals | Free |
| LangChain Documentation | Building AI apps | Free |
| Kaggle | Practice with real data | Free |
| Hugging Face | Open-source AI models | Free |

---

## Section 7 — Internship Completion Checklist

Before you leave today, verify:

- [ ] Capstone notebook runs from top to bottom without errors
- [ ] At least one successful Groq API call completed
- [ ] Data loaded from CSV and analyzed with Pandas or SQL
- [ ] AI response is displayed and readable
- [ ] Output saved to a file (`.txt` or `.csv`)
- [ ] Code has markdown cells explaining each major section
- [ ] Responsible AI guardrail added to at least one prompt
- [ ] Capstone showcased to the class
- [ ] All 10 day notebooks saved locally or to Google Drive
- [ ] Career roadmap action plan noted

---

## Section 8 — Practice Questions and Answers

### Beginner Questions

**Q1: What does AI hallucination mean? Give an example.**

*Answer:* AI hallucination is when an LLM generates confident, fluent text that is factually incorrect. Example: asking an AI "Who won the 2025 ICC World Cup?" and it confidently names a team that never participated. The AI does not look up facts — it predicts the next word based on patterns, which can produce false information.

**Q2: Name three ways GenAI and Data Engineering work together.**

*Answer:* (1) Data Engineering cleans and structures data that LLMs use as context in RAG systems. (2) Data pipelines fetch real-time information that keeps AI answers accurate and up-to-date. (3) Vector databases (ChromaDB) are maintained by data engineering infrastructure so AI can perform semantic search over large document collections.

**Q3: What is RAG and how does it reduce hallucination?**

*Answer:* RAG stands for Retrieval-Augmented Generation. Before the LLM answers a question, RAG first searches a database for relevant documents and injects those documents into the prompt. Because the AI now has actual source material to read, it is less likely to invent information. Instead of guessing, it reads and summarizes verified content.

---

### Intermediate Questions

**Q4: Explain the four layers of an end-to-end AI data system.**

*Answer:* Layer 1 — Data Ingestion: raw data is loaded from CSVs, APIs, or databases using Pandas or ETL pipelines. Layer 2 — Data Processing: data is cleaned, structured, and stored in SQLite or ChromaDB. Layer 3 — AI Reasoning: the LLM receives structured data or retrieved documents and generates intelligent answers via the Groq API. Layer 4 — Delivery: results are displayed to the user as reports, chatbot responses, or saved files.

**Q5: What is AI bias and how would you detect it in a student grade prediction model?**

*Answer:* AI bias occurs when a model gives systematically unfair predictions based on protected attributes like gender or branch. To detect it: train the model, then evaluate its predictions separately for male vs. female students. If the model predicts lower grades for female students despite equal marks, that is bias. Also check predictions by branch and socioeconomic indicators. Tools like scikit-learn's fairness metrics can quantify this.

---

### Coding Questions

**Q6: Write Python code to load student_performance.csv, calculate average GPA by branch, and send the result to Groq AI for analysis.**

```python
import pandas as pd
import sqlite3
from groq import Groq

df = pd.read_csv("student_performance.csv")
conn = sqlite3.connect(':memory:')
df.to_sql('students', conn, if_exists='replace', index=False)

result = pd.read_sql(
    "SELECT branch, ROUND(AVG(gpa),2) AS avg_gpa FROM students GROUP BY branch ORDER BY avg_gpa DESC",
    conn
)

summary = result.to_string(index=False)
client = Groq(api_key="gsk_your_key")
response = client.chat.completions.create(
    model="llama-3.1-8b-instant",
    max_tokens=300,
    messages=[
        {"role": "system", "content": "You are a data analyst. Only use the data provided."},
        {"role": "user",   "content": f"Branch GPA data:\n{summary}\n\nWhich branch needs improvement?"}
    ]
)
print(response.choices[0].message.content)
```

**Q7: Write a ChromaDB query that retrieves the top 2 notes most similar to "What is recursion?"**

```python
from sentence_transformers import SentenceTransformer
import chromadb

embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
chroma_client   = chromadb.EphemeralClient()
collection      = chroma_client.get_or_create_collection("college_kb")

# Assuming notes were already added to the collection

question = "What is recursion?"
question_embedding = embedding_model.encode(question).tolist()

results = collection.query(
    query_embeddings=[question_embedding],
    n_results=2
)

for i, doc in enumerate(results['documents'][0]):
    print(f"Result {i+1}: {doc[:200]}")
```

**Q8: Write a Groq API call with a system prompt that prevents hallucination.**

```python
from groq import Groq

client = Groq(api_key="gsk_your_key")

response = client.chat.completions.create(
    model="llama-3.1-8b-instant",
    max_tokens=400,
    messages=[
        {
            "role": "system",
            "content": """You are a helpful data quality advisor.
Never make up answers. If you are unsure, say:
I do not have enough information to answer that accurately.
Only use facts from the data provided by the user."""
        },
        {
            "role": "user",
            "content": "What cleaning steps should I apply to a dataset with 5 missing GPA values?"
        }
    ]
)
print(response.choices[0].message.content)
```

---

## The Responsible AI Engineer's Pledge

*Read this. Remember it. Build with it.*

1. I will design AI systems that are accurate, fair, and transparent
2. I will use real, verified data — not hallucinated facts
3. I will protect user privacy and never misuse personal data
4. I will test my AI systems for bias before deploying them
5. I will continue learning and improving as an AI engineer
6. I will build technology that helps — not harms — people

---

*Codeboosters Tech — Data Engineering + GenAI Internship — Day 10 Complete*

*You are now a Data + AI Engineer. Go build something great.*
