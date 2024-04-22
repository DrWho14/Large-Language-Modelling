import streamlit as st
import requests
import re
import evaluate
import numpy as np
import fitz

from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.document_loaders import PyMuPDFLoader
from langchain.output_parsers import PydanticOutputParser


from newspaper import Article
from pydantic import BaseModel, Field
from typing import Optional, List

# Functions
def web_article_scraper(article_url):
    title = None
    text = None
    headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/89.0.4389.82 Safari/537.36'
    }
    session = requests.Session()
    try:
      response = session.get(article_url, headers=headers, timeout=10)
    
      if response.status_code == 200:
          article = Article(article_url)
          article.download()
          article.parse()
            
          title = article.title
          text = article.text
      else:
          st.error(f"Failed to fetch article at {article_url}")
    except Exception as e:
      st.error(f"Error occurred while fetching article at {article_url}: {e}")

    return title, text

# def pdf_content_scraper(pdf_location=):
#   pdf_loader = PyMuPDFLoader(pdf_location)
#   pdf_pages = pdf_loader.load_and_split()
#   return pdf_pages

# def concat_pdf_content(pdf_pages):
#   doc_content = ""
#   for page in pdf_pages:
#     doc_content += page.page_content

#   return doc_content
    
def generate_qa_prompt_template():

    prompt_template = """ You are an assitant that generates questions and answers from corpus
    Here's the content you want to generate questions and answers from.
    ==================
    {corpus}
    ==================
    {format_instructions}
    Generate {qa_count} question and answer pairs from the corpus.
    """
    
    return prompt_template

def generate_QA(article_url, api_key, temperature=0.0, count=10):
    response = None
    try:
        chat = ChatOpenAI(model_name="gpt-3.5-turbo-16k", openai_api_key=api_key ,temperature=temperature)
        title, content = web_article_scraper(article_url)
        if content != None:
            qa_prompt_template = generate_qa_prompt_template()
            parser = PydanticOutputParser(pydantic_object=Questionnaire)
            #
            qa_prompt = PromptTemplate(template=qa_prompt_template, input_variables=["corpus", "qa_count"] , partial_variables={"format_instructions": parser.get_format_instructions()})
            llm_qa = LLMChain(llm=chat, prompt = qa_prompt, output_parser=parser) #
            response = llm_qa.predict(corpus = content, qa_count=count)
    except Exception as e:
        st.error(f"Error occured while generating Q&As. Retry Generate... : \n{e}", icon='❌')
    return response

def generate_QA_pdf(content, api_key, temperature=0.0, count=10):
    response = None
    try:
        chat = ChatOpenAI(model_name="gpt-3.5-turbo-16k", openai_api_key=api_key ,temperature=temperature)
        if content != None:
            qa_prompt_template = generate_qa_prompt_template()
            parser = PydanticOutputParser(pydantic_object=Questionnaire)
            #
            qa_prompt = PromptTemplate(template=qa_prompt_template, input_variables=["corpus", "qa_count"] , partial_variables={"format_instructions": parser.get_format_instructions()})
            llm_qa = LLMChain(llm=chat, prompt = qa_prompt, output_parser=parser) #
            response = llm_qa.predict(corpus = content, qa_count=count)
    except Exception as e:
        st.error(f"Error occured while generating Q&As. Retry Generate... : \n{e}", icon='❌')
    return response    

class Questionnaire(BaseModel):
    questions: Optional[List[str]] = Field(description="List of generated questions")
    answers: Optional[List[str]] = Field(description="List of generated answers")

if 'questions' not in st.session_state:
    st.session_state.questions = {}
if 'answers' not in st.session_state:
    st.session_state.answers = {}
if 'displayedAnswers' not in st.session_state:
    st.session_state.displayedAnswers = {}
if 'warnings' not in st.session_state:
    st.session_state.warnings = {}
if 'userAnswers' not in st.session_state:
    st.session_state.userAnswers = {}
if 'questionnaire' not in st.session_state:
    st.session_state.questionnaire = 0
if 'rougeScores' not in st.session_state:
    st.session_state.rougeScores = {}

if 'pdf_questions' not in st.session_state:
    st.session_state.pdf_questions = {}
if 'pdf_answers' not in st.session_state:
    st.session_state.pdf_answers = {}
if 'pdf_displayedAnswers' not in st.session_state:
    st.session_state.pdf_displayedAnswers = {}
if 'pdf_warnings' not in st.session_state:
    st.session_state.pdf_warnings = {}
if 'pdf_userAnswers' not in st.session_state:
    st.session_state.pdf_userAnswers = {}
if 'pdf_questionnaire' not in st.session_state:
    st.session_state.pdf_questionnaire = 0
if 'pdf_rougeScores' not in st.session_state:
    st.session_state.pdf_rougeScores = {}

rouge = evaluate.load('rouge')


def submit_answer(ans_id):
    set_user_answer(ans_id)
    user_answer = st.session_state.userAnswers[ans_id]
    # Always clear the warning and set it up again
    st.session_state.warnings[ans_id] = ""
    if user_answer != "":
        reference_answer = st.session_state.answers[ans_id]
        st.session_state.displayedAnswers[ans_id] = reference_answer
        # Calculate rouge score and set it in session state
        rouge_scores = rouge.compute(predictions=[user_answer],references=[reference_answer])
        st.session_state.rougeScores[ans_id] = rouge_scores
    else:
        st.session_state.warnings[ans_id] = f"Please Answer Question {ans_id} before submitting!"

def submit_answer_pdf(ans_id):
    set_user_answer_pdf(ans_id)
    user_answer = st.session_state.pdf_userAnswers[ans_id]
    # Always clear the warning and set it up again
    st.session_state.pdf_warnings[ans_id] = ""
    if user_answer != "":
        reference_answer = st.session_state.pdf_answers[ans_id]
        st.session_state.pdf_displayedAnswers[ans_id] = reference_answer
        # Calculate rouge score and set it in session state
        rouge_scores = rouge.compute(predictions=[user_answer],references=[reference_answer])
        st.session_state.pdf_rougeScores[ans_id] = rouge_scores
    else:
        st.session_state.pdf_warnings[ans_id] = f"Please Answer Question {ans_id} before submitting!"
        
def set_user_answer(ans_id):
    st.session_state.userAnswers[ans_id] = st.session_state[f"UA{ans_id}"]

def set_user_answer_pdf(ans_id):
    st.session_state.pdf_userAnswers[ans_id] = st.session_state[f"pdf-UA{ans_id}"]

def clear_questionnaire():
    # Clear previous questionnaire
    questions = st.session_state.questions
    if questions:
        for qId, question in questions.items():
            if f"UA{qId}" in st.session_state:
                st.session_state[f"UA{qId}"]= ""

def clear_questionnaire_pdf():
    # Clear previous pdf questionnaire
    questions = st.session_state.pdf_questions
    if questions:
        for qId, question in questions.items():
            if f"pdf-UA{qId}" in st.session_state:
                st.session_state[f"pdf-UA{qId}"]= ""
    
# Side bar widget
with st.sidebar:
    st.markdown("**<small>Version 2.0</small>**", unsafe_allow_html=True)
    openai_api_key = st.text_input("**OpenAI API Key**", key="qa_api_key", type="password")
    st.title("App Settings")
    max_number_of_qa = st.slider(
        "**Max number of questions & answers to be generated**",
        min_value=1,
        max_value=20,
        value=10,
        step=1
    )
    temperature = st.slider(
        "**Temperature used to control how creative the QA generation should be.**",
        min_value=0.0,
        max_value=2.0,
        value=0.9,
        step=0.1
    )
    # Uncomment and use if measurement is by these rouge measures
    # rouge1_threshold = st.slider(
    #     "**Rouge 1 threshold for accepting an user's answer for a question.**",
    #     min_value=0.0,
    #     max_value=1.0,
    #     value=0.5,
    #     step=0.01        
    # )
    # rouge2_threshold = st.slider(
    #     "**Rouge 2 threshold for accepting an user's answer for a question.**",
    #     min_value=0.0,
    #     max_value=1.0,
    #     value=0.5,
    #     step=0.01        
    # )
    rougel_threshold = st.slider(
        "**Rouge L threshold for accepting an user's answer for a question.Uses Longest common subsequence based scoring.**",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.01        
    )
    # Uncomment for using rouge Lsum
    # rougelsum_threshold = st.slider(
    #     "**Rouge LSum threshold for accepting an user's answer for a question.**",
    #     min_value=0.0,
    #     max_value=1.0,
    #     value=0.5,
    #     step=0.01        
    # )

# Title
st.title(":genie: QAGenie: A Questions and Answers Generator")
st.markdown("*<small>A Question Answer Generative tool for scalable skills assessment.</small>*", unsafe_allow_html=True)
description = st.expander(label="Description")
with description:
    st.markdown("""
It can empower organizations to establish mutually accepted skill assessments and overcome friction in this process. The benefits are profound, including reduced employee churn, accurate benchmarking of talent and skills, and the ability to quantify and measure skill levels objectively. 

With QAGenie, organizations can now possess a quantitative and measurable skills map of their workforce, enabling them to proactively undertake skill improvement measures and gauge progress over time.

""", unsafe_allow_html=True)

how_it_works = st.expander(label="How it works")
with how_it_works:
    st.markdown(
        """
The application follows a sequence of steps to generate Questions & Answers:
- **User Input**: The application starts by collecting content either by a website URL or uploaded pdf(or text) file.
- **QA Generator**: The user-provided content is then fed to a Large Language Model (ChatOpenAI) via LangChain LLMChain. The LLMChain interprets and generates the Q&As from the content.
- **User Answers**: Users can write their answer in the provided text area and hit submit.
- **Answer Submission**: Once a user submits an answer, the application displays the answer generated by the LLM.
- **Answer Scoring**: we use Rouge scoring to compare user's answer to LLM generated answer. Specifically we use RougeL score, and if it is more than the threshold set in App Settings it will mark it as a pass (:white_check_mark:), else it is marked as fail(:x:).
- **App Settings**: Here we can set how many question answer pairs we need to generate, the temparature setting lets the LLM know how deterministic or creative the Q&As should be.
""", unsafe_allow_html=True
    )
    #st.image(image='QAGenie-Workflow.png')

tab_url, tab_pdf = st.tabs([":desktop_computer: Website", ":page_facing_up: PDF"])

with tab_url:
    st.header("From Website URLs")
    # Generate Questions & Answers Form
    with st.form('generate_form'):
        article_url = st.text_input('Enter Web Url here to scrape content and generate Q&A', value='', placeholder='https://blog.langchain.dev/agents-round/')
        submitted = st.form_submit_button('Generate', on_click=clear_questionnaire)
        if not openai_api_key.startswith('sk-'):
            st.warning('Please enter your OpenAI API key!', icon='⚠')
        if submitted and openai_api_key.startswith('sk-'):
            # Validate URL
            url_pattern_protocol = "^https?:\\/\\/(?:www\\.)?[-a-zA-Z0-9@:%._\\+~#=]{1,256}\\.[a-zA-Z0-9()]{1,6}\\b(?:[-a-zA-Z0-9()@:%_\\+.~#?&\\/=]*)$"
            #TODO : Accept without protocol
            url_pattern_no_protocol ="^[-a-zA-Z0-9@:%._\\+~#=]{1,256}\\.[a-zA-Z0-9()]{1,6}\\b(?:[-a-zA-Z0-9()@:%_\\+.~#?&\\/=]*)$"
            if article_url!="":
                if re.match(url_pattern_protocol,article_url)!=None:
                    
                    with st.spinner(text='Generating Q&As'):
                        llm_output = generate_QA(article_url, openai_api_key, temperature, max_number_of_qa)
                        if llm_output !=None:
                            questions = llm_output.questions
                            answers = llm_output.answers
                            counter = 1
                            for question, answer in zip(questions, answers):
                                st.session_state.questions[counter] = question
                                st.session_state.answers[counter] = answer
                                st.session_state.displayedAnswers[counter] = ""
                                st.session_state.userAnswers[counter] = ""
                                st.session_state.warnings[counter] = ""
                                st.session_state.rougeScores[counter] = None
                                counter +=1
                            
                            st.success('Done')
                        else:
                            st.warning('No response received!!', icon='⚠')
                else:
                    st.warning('Please enter a valid web URL with http(s) protocol.', icon='⚠')
            else:
                st.warning('Please enter an URL', icon='⚠')
            
    questionnaire = st.empty()
    with questionnaire.container():
        questions = st.session_state.questions
        if questions:
            for qId, question in questions.items():
                form = st.form(f"qa-form{qId}")
                with form:
                    form.info(f"Q{qId}: {question}")
                    user_answer = form.text_area("Your Answer", key = f"UA{qId}", max_chars=300)
                    disable_submit = True if st.session_state.displayedAnswers[qId] != "" else False
                    form.form_submit_button("Submit", disabled= disable_submit, on_click=submit_answer, args=[qId]) #
                    answer_placeholder = form.empty()
                    with answer_placeholder.container():
                        # Warnings element
                        warning_placeholder = st.empty()
                        if st.session_state.warnings[qId] == "":
                            warning_placeholder.empty() # if there are no warnings then clear the warning
                        else:
                            warning_placeholder.warning(st.session_state.warnings[qId], icon='⚠')
                            
                        # Score element
                        scores_palceholder = st.empty()
                        if st.session_state.rougeScores[qId] != None:
                            rouge_scores = st.session_state.rougeScores[qId]
                            rouge1 = np.round(rouge_scores['rouge1'], 2)
                            rouge2 = np.round(rouge_scores['rouge2'],2)
                            rougeL = np.round(rouge_scores['rougeL'],2)
                            rougeLsum = np.round(rouge_scores['rougeLsum'],2)
                            if rougeL >= rougel_threshold:
                                st.caption(':white_check_mark: :green[ Pass: Correct Answer!!]')
                            else:
                                st.caption(':x: :red[ Fail: Incorrect Answer!!]')
                            
                            scores_palceholder.info(f"Rouge1 : {rouge1}, Rouge2 : {rouge2}, RougeL : {rougeL}, RougeLSum : {rougeLsum} ")
                        else:
                            scores_palceholder.empty()
                        
                        form.info(st.session_state.displayedAnswers[qId])

with tab_pdf:
    st.header("From PDFs")
    with st.form('pdf_generate_form', clear_on_submit=True):
        uploaded_file = st.file_uploader("Upload a PDF document", type=("pdf"))
        submitted = st.form_submit_button('Generate', on_click=clear_questionnaire_pdf)
        # Generate Questions & Answers from uploaded PDFs
        if not openai_api_key.startswith('sk-'):
            st.warning('Please enter your OpenAI API key!', icon='⚠')
        if uploaded_file and submitted and openai_api_key.startswith('sk-'):
            with st.spinner(text='Generating Q&As'):
                with fitz.open(stream=uploaded_file.read(), filetype="pdf") as doc:
                    pdf_content = ""
                    for page in doc:
                        pdf_content += page.get_text()
                llm_output = generate_QA_pdf(pdf_content, openai_api_key, temperature, max_number_of_qa)
                if llm_output !=None:
                    questions = llm_output.questions
                    answers = llm_output.answers
                    counter = 1
                    for question, answer in zip(questions, answers):
                        st.session_state.pdf_questions[f"{counter}"] = question
                        st.session_state.pdf_answers[f"{counter}"] = answer
                        st.session_state.pdf_displayedAnswers[f"{counter}"] = ""
                        st.session_state.pdf_userAnswers[f"{counter}"] = ""
                        st.session_state.pdf_warnings[f"{counter}"] = ""
                        st.session_state.pdf_rougeScores[f"{counter}"] = None
                        counter +=1
                    
                    st.success('Done')
                else:
                    st.warning('No response received!!', icon='⚠')

    pdf_questionnaire = st.empty()
    with pdf_questionnaire.container():
        questions = st.session_state.pdf_questions
        if questions:
            for qId, question in questions.items():
                form = st.form(f"pdf-qa-form{qId}")
                with form:
                    form.info(f"Q{qId}: {question}")
                    user_answer = form.text_area("Your Answer", key = f"pdf-UA{qId}", max_chars=300)
                    disable_submit = True if st.session_state.pdf_displayedAnswers[qId] != "" else False
                    form.form_submit_button("Submit", disabled= disable_submit, on_click=submit_answer_pdf, args=[qId]) #
                    answer_placeholder = form.empty()
                    with answer_placeholder.container():
                        # Warnings element
                        warning_placeholder = st.empty()
                        if st.session_state.pdf_warnings[qId] == "":
                            warning_placeholder.empty() # if there are no warnings then clear the warning
                        else:
                            warning_placeholder.warning(st.session_state.pdf-warnings[qId], icon='⚠')
                            
                        # Score element
                        scores_palceholder = st.empty()
                        if st.session_state.pdf_rougeScores[qId] != None:
                            rouge_scores = st.session_state.pdf_rougeScores[qId]
                            rouge1 = np.round(rouge_scores['rouge1'], 2)
                            rouge2 = np.round(rouge_scores['rouge2'],2)
                            rougeL = np.round(rouge_scores['rougeL'],2)
                            rougeLsum = np.round(rouge_scores['rougeLsum'],2)
                            if rougeL >= rougel_threshold:
                                st.caption(':white_check_mark: :green[ Pass: Correct Answer!!]')
                            else:
                                st.caption(':x: :red[ Fail: Incorrect Answer!!]')
                            
                            scores_palceholder.info(f"Rouge1 : {rouge1}, Rouge2 : {rouge2}, RougeL : {rougeL}, RougeLSum : {rougeLsum} ")
                        else:
                            scores_palceholder.empty()
                        
                        form.info(st.session_state.pdf_displayedAnswers[qId]) 
            
        
    
