import os
import streamlit as st

from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

st.set_page_config(
    page_title="LinkedIn Content Generator",
    page_icon="💼",
    layout="centered"
)

st.title("LinkedIn Content Generator")
st.write("Generate engaging LinkedIn posts from your ideas and achievements.")

# Sidebar
with st.sidebar:
    st.header("Settings")
    groq_api_key = st.text_input(
        "Groq API Key",
        type="password"
    )

# Main Inputs
instruction = st.text_area(
    "Enter your instruction",
    placeholder="Example: I completed a GenAI project using LangChain and Groq APIs..."
)

token_size = st.number_input(
    "Approximate Post Length (tokens)",
    min_value=50,
    max_value=1000,
    value=200,
    step=50
)

emoji = st.radio(
    "Include Emojis?",
    ["yes", "no"]
)

temperature = st.slider(
    "Creativity Level",
    min_value=0.0,
    max_value=1.0,
    value=0.7,
    step=0.1
)

if st.button("Generate LinkedIn Post"):

    if not groq_api_key:
        st.error("Please enter your Groq API Key.")
        st.stop()

    os.environ["GROQ_API_KEY"] = groq_api_key

    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=temperature
    )

    prompt = PromptTemplate(
        input_variables=[
            "instruction",
            "emoji",
            "token_size"
        ],
        template="""
You are a professional LinkedIn content writer and personal branding expert.

Your task is to generate a high-quality LinkedIn post from the user's instruction.

Follow these rules:

1. Understand the user's instruction, achievement, experience, opinion, project, event, learning, or announcement.
2. Create an attention-grabbing opening hook.
3. Write in a professional yet natural and human tone.
4. Keep the post approximately within the token_size specified.
5. Highlight key insights, lessons learned, challenges overcome, or impact created.
6. Maintain a positive and engaging style.
7. Avoid generic AI phrases, clichés, and excessive corporate jargon.
8. Include relevant emojis only if emoji = yes.
9. End with a question or call-to-action to encourage engagement.
10. Add 3–8 relevant hashtags.
11. Do not explain your reasoning.
12. Return only the final LinkedIn post.

Instruction:
{instruction}

Emoji:
{emoji}

Token Size:
{token_size}

Answer:
"""
    )

    parser = StrOutputParser()
    chain = prompt | llm | parser

    with st.spinner("Generating LinkedIn post..."):
        response = chain.invoke({
            "instruction": instruction,
            "emoji": emoji,
            "token_size": token_size
        })

    st.success("Post Generated Successfully!")
    st.markdown("### Generated LinkedIn Post")
    st.write(response)

    st.download_button(
        label=" Download Post",
        data=response,
        file_name="linkedin_post.txt",
        mime="text/plain"
    )