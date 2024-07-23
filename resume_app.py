# -*- coding: utf-8 -*-
"""
Created on Tue Jul 16 21:44:55 2024

@author: Fatemeh Dalilian
"""

import streamlit as st
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI


google_api_key = st.secrets["GOOGLE_API_KEY"]




llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro",
                             temperature=0.3)




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
List all the keywords implied from the following resume and format them in a structured manner :

Resume:
{text}

## Keywords:

- List each keyword (where this keyword is extracted and implied from)
 """

match_prompt = """
Given the list of qualifications and keywords extracted from the job description, and the qualifications as well as implied qualification listed on the resume, provide the following (do not make up things or recommendations if the resume is good):

1. A list of qualifications from the resume that match the job description keywords and qualifications.
2. A list of qualifications from the resume that do not match the job description keywords and qualifications.
3. A list of qualifications needed in the job description keywords but do not match the resume qualifications.


## Job Description Keywords and Qualifications:
{keywords_and_qualifications}

## Resume Qualifications:
{resume_qualifications}

## Output:
### Matching Qualifications:
- List each matching qualification (where this is extracted or implied from)

### Qualifications you might wanna add:
- List each qualification needed in the job description not mentioned in the resume. (where this keyword is extracted or implied from)

### Non-Matching Qualifications:
- List each non-matching qualification (if it is beneficial to have on the resume)

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




# Main function to run the Streamlit app
def main():
    st.title("Resume Analyzer Web Application")
    
    col1, col2 = st.columns(2)
    with col1:
      resume_input = st.text_area("Resume", label_visibility='collapsed', placeholder="Enter the resume content here...", key="resume_input")

    with col2:
      job_input = st.text_area("Job Description", label_visibility='collapsed', placeholder="Enter the job description here...", key="job_input")


    if st.button("Analyze Resume"):
        with st.spinner("Analyzing..."):
            results = match_analysis(job_input, resume_input)
            st.write(results)


if __name__ == "__main__":
    main()
