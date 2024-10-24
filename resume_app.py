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
ASK:
Provide a list of keywords from the following job description, ranked by importance.

CONTEXT:
Job Description:
{text}

CONSTRAINTS:

List only the keywords in order of importance.
For years of industry experience, state the number if mentioned; otherwise, say "Not specified."
OUTPUT FORMAT:

Keywords in order of importance:
List each keyword.
Years of industry experience:
Mention the years of experience or say "Not specified."

"""

resume_prompt = """

Here’s the refined version of your prompt:

ASK:
List all the keywords implied from the following resume and present them in a structured format.

CONTEXT:
Resume:
{text}

OUTPUT FORMAT:

Keywords:
List each keyword, specifying the section of the resume from which the keyword is extracted or implied.
 """

match_prompt = """
ASK:
Based on the provided list of qualifications and keywords extracted from the job description, and the qualifications listed or implied in the resume, provide the following:

A list of qualifications from the resume that match the job description keywords and qualifications.
A list of qualifications from the resume that do not match the job description keywords and qualifications.
A list of qualifications required by the job description that are not present on the resume.
CONSTRAINTS:

Do not make up recommendations if the resume is complete.
Do not suggest changes if the resume is a good fit.
INPUT:

Job Description Keywords and Qualifications:
{keywords_and_qualifications}

Resume Qualifications:
{resume_qualifications}

OUTPUT:

Matching Qualifications:
List each matching qualification (state where it is extracted or implied from).
Qualifications to Consider Adding:
List each qualification needed in the job description that is not present on the resume (state where the keyword is extracted or implied from).
Qualifications on the Resume Not Required by the Job:
List each non-matching qualification, mentioning if it's beneficial to have on the resume.
Recommendations:
Provide any recommendations only if necessary (optional).

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
