import torch
import numpy as np
from sentence_transformers import util
from langchain.text_splitter import CharacterTextSplitter
from bs4 import BeautifulSoup
import requests
from PyPDF2 import PdfReader
import streamlit as st
import os

hf_token = st.secrets['hf_token']

LLM_api_key= st.secrets['LLM_api_key']

st.title('Ask your PDF 💬')

pdf=st.file_uploader(label='Please upload your pdf file',type=['pdf'])


if pdf:
    pdf=PdfReader(pdf)
    text=''
    for i in range(len(pdf.pages)):
        text+=pdf.pages[i].extract_text()
    
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        #chunk_overlap=200,
        length_function=len
      )
    chunks = text_splitter.split_text(text)
    
    input=st.text_input(label='Ask your question here 😀')
    
    submit=st.button('Submit')

    if input or submit:    
        api_url = f"https://api-inference.huggingface.co/pipeline/feature-extraction/sentence-transformers/all-MiniLM-L6-v2"
    
        headers = {"Authorization": f"Bearer {hf_token}"}

        def query(texts):
            response = requests.post(api_url, headers=headers, json={"inputs": texts, "options":{"wait_for_model":True}})
            return response.json()

        question = query([input])
        
        query_embeddings = torch.FloatTensor(question)

        output=query(chunks)

        output=torch.from_numpy(np.array(output)).to(torch.float)

        result=util.semantic_search(query_embeddings, output,top_k=2)

        final=[chunks[result[0][i]['corpus_id']] for i in range(len(result[0]))]

        url = "https://api.ai21.com/studio/v1/answer"

        payload = {
            "context":' '.join(final),
            "question":input
        }
        headers = {
            "accept": "application/json",
            "content-type": "application/json",
            "Authorization": f"Bearer {LLM_api_key}"
        }

        response = requests.post(url, json=payload, headers=headers)

        if response.json()['answerInContext']:
            st.write(response.json()['answer'])
            st.header(':blue[Context]')
            st.write(final[0])
        else:
            st.write('The answer is not in the document ⚠️, please reforumulate your question')

        

        
        
