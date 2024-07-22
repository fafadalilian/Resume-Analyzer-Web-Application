# -*- coding: utf-8 -*-
"""
Created on Tue Jul 16 21:44:55 2024

@author: Fatemeh Dalilian
"""
#from dotenv import load_dotenv
#import os
import streamlit as st
# from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
#load_dotenv()

google_api_key = st.secrets["GOOGLE_API_KEY"]

#google_api_key = os.getenv("GOOGLE_API_KEY")

# st.set_page_config(page_title="Detailed Resume Analysis", page_icon=":robot:")
st.header("Detailed Resume Analysis:")

# def get_api_key():
#     input_text = st.text_input(label="Google API Key ",  placeholder="Ex: sk-2twmA8tfCb8un4...", key="google_api_key_input")
#     return input_text

# google_api_key = get_api_key()




def get_resume():
    resume_text = st.text_area(label="resume Input", label_visibility='collapsed', placeholder="Your resume...", key="resume_input")
    return resume_text

def get_job():
    job_text = st.text_area(label="job Input", label_visibility='collapsed', placeholder="Your job desciption...", key="job_input")
    return job_text



llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro",
                             temperature=0.3)


col1, col2 = st.columns(2)
with col1:
  resume_input = get_resume()

with col2:
  job_input = get_job()



job_prompt = """
give me a list of all keywords from the following job description in order of importance (do not provide more information):

Job Description:
{text}


Keywords in order of importance:
- List each keyword
Years of industry experience:
- Mention the years of industry experience (if needed) otherwise say Not specified

"""

resume_prompt = """
Extract the main keywords from the following resume and format them in a clear, structured manner :

Resume:
{text}

## Keywords:

- List each keyword
 """

match_prompt = """
Given the list of qualifications and keywords extracted from the job description, and the qualifications listed on the resume, provide the following (do not make up things or recommendations if the resume is good):

1. A list of qualifications from the resume that match the job description keywords and qualifications.
2. A list of qualifications from the resume that do not match the job description keywords and qualifications.
3. A list of qualifications needed in the job description keywords but do not match the resume qualifications.


## Job Description Keywords and Qualifications:
{keywords_and_qualifications}

## Resume Qualifications:
{resume_qualifications}

## Output:
### Matching Qualifications:
- List each matching qualification

### Non-Matching Qualifications:
- List each non-matching qualification


### Qualifications to be added:
- List each qualification needed in the job description not mentioned in the resume

### Recommendations:
- List each recommendation
at the end put
###
"""

def analyse_resume(resume):
  resume_keyword_prompt = PromptTemplate(template=resume_prompt, input_variables=["text"])
  resume_keyword_chain = LLMChain(llm=llm, prompt=resume_keyword_prompt)
  resume_qualifications = resume_keyword_chain.run({"text": resume})
  return resume_qualifications

def analyse_job(job_description):
  # Create the prompt template
  job_keyword_prompt = PromptTemplate(template=job_prompt, input_variables=["text"])
  # Create a chain to extract keywords using the LLM
  job_keyword_chain = LLMChain(llm=llm, prompt = job_keyword_prompt)
  # Extract keywords from the job description
  keywords_and_qualifications = job_keyword_chain.run({"text": job_description})
  return keywords_and_qualifications

def match_analysis(job_text, resume_text):
  resume_qualifications = analyse_resume(resume_text)
  keywords_and_qualifications = analyse_job(job_text)
  match_prompt_template = PromptTemplate(template=match_prompt, input_variables=["keywords_and_qualifications", "resume_qualifications"])
  match_chain = LLMChain(llm=llm, prompt=match_prompt_template)
  match_result = match_chain.run({"keywords_and_qualifications": keywords_and_qualifications, "resume_qualifications": resume_qualifications})
  return match_result


# Create a placeholder for the results at the top of the page
#result_placeholder = st.empty()

# job_text = get_job()
# if len(job_input.split(" ")) > 700:
#     st.write("Please enter a shorter job description. The maximum length is 700 words.")
#     st.stop()
# resume_text 


# if len(email_input.split(" ")) > 700:
#     st.write("Please enter a shorter email. The maximum length is 700 words.")
#     st.stop()

def analyze():
    # Show "OK Wait!" message
    wait_placeholder = st.empty()
    wait_placeholder.markdown("### OK Wait!")
    
    # Perform the analysis
    res = match_analysis(job_input, resume_input)
    
    # Clear the "OK Wait!" message and show the result
    wait_placeholder.empty()
    result_placeholder.write(res)

# Placeholder for the results at the bottom of the page
result_placeholder = st.empty()

if job_input and resume_input:
    if st.button("*analyze*", type='secondary', help="Click to analyze your resume."):
        analyze()