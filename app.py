import streamlit as st
import validators
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import YoutubeLoader, UnstructuredURLLoader

st.title("YouTube and Website Summarization")
st.subheader("Summarize URL")

with st.sidebar:
    api_key = st.text_input("Groq API Key", value ="",type = "password")
    st.write("Made by **Ankit**")
    st.sidebar.write("""
        **Why We Need Your API Key:**
        
        To ensure your privacy and security, we require you to provide your Groq API Key. This key is used exclusively for accessing the summarization capabilities of our application and is not stored or shared with any third parties. Your API Key allows us to leverage powerful language models to generate summaries without compromising your personal information.

        **Please enter your Groq API Key to get started.** 
        If you don't know where to obtain your API key, you can follow this link: 
        [Get your Groq API Key here](https://console.groq.com/keys).
    """)
    st.write("""
    #### Important Note on YouTube Summarization:
    
    For YouTube videos that do not have a transcript or closed captions available, our tool attempts to summarize the content by extracting audio information. While this process can yield useful summaries, the accuracy and comprehensiveness of the results may vary. 
    Therefore, for the best summarization experience, it is recommended to use videos that include transcripts or closed captions.
""")
gen_url = st.text_input("URL", label_visibility="collapsed" )

if not api_key.strip():
    st.error("API Key is required!")

prompt_template = """
Provide summary of the following content in 100 to 300 words:
content: {text}"""
prompt = PromptTemplate(template = prompt_template, input_variables=['text'])
output_displayed =False
if st.button("Summarize"):
    if not api_key.strip() or not gen_url.strip():
        st.error("Please provide the required informations")
    elif not validators.url(gen_url):
        st.error("Invalid URL")
    else:
        try:
            with st.spinner("Summarizing... Please wait. This may take a moment."):
                llm = ChatGroq(model = "gemma2-9b-it",groq_api_key=api_key)
                if "youtube.com" in gen_url or "youtu.be" in gen_url:
                    loader = YoutubeLoader.from_youtube_url(gen_url, add_video_info=True)
                    data = loader.load()
                    if not data:
                        st.error("No transcript available for this video.")
                    else:
                        # Summarization step for YouTube
                        chain = load_summarize_chain(llm, chain_type="stuff", prompt=prompt)
                        output = chain.run(data)
                        st.success(output)
                        output_displayed = True
                else:
                    loader = UnstructuredURLLoader(urls=[gen_url], ssl_verify=False,
                        headers={"User-Agent": "Mozilla/5.0"})
                    data = loader.load()
                    chain = load_summarize_chain(llm, chain_type="stuff", prompt=prompt)
                    output = chain.run(data)
                    st.success(output)
                    output_displayed = True
        except Exception as e:
            st.error(f"Exception: {e}")
if not output_displayed:
    st.write("""
        Welcome to the **YouTube and Website Summarization Tool**! This web application is designed to help you quickly and efficiently summarize content from YouTube videos and various websites.

        #### Key Features:
        - **YouTube Video Summarization:** Simply input the URL of any YouTube video, and our tool will extract key information and provide a concise summary, saving you time on content consumption.
        - **Website Content Summarization:** Enter any valid URL, and the tool will fetch the content, allowing you to receive a summary that highlights the main points, essential for research or quick information gathering.
        - **User-Friendly Interface:** With a clean and intuitive design, anyone can easily navigate through the app, making summarization accessible for all.

        #### How It Works:
        1. **Input Your Groq API Key:** To get started, enter your Groq API Key in the sidebar. This key allows us to leverage powerful language models for summarization.
        2. **Provide a URL:** Paste the URL of the YouTube video or website you want to summarize.
        3. **Receive Your Summary:** Click on the "Summarize" button, and within moments, you’ll receive a well-structured summary of the content.

        #### Why Use This Tool?
        In today’s fast-paced digital world, finding time to watch lengthy videos or read through extensive articles can be challenging. This summarization tool empowers you to grasp essential information quickly, helping you stay informed and make better decisions.

        #### Feedback and Support:
        We value your feedback! If you encounter any issues or have suggestions for improvement, please reach out through the provided contact options.
    """)
