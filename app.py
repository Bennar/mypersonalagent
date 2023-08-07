# -*- coding: utf-8 -*-

from langchain import LLMMathChain
from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType
from langchain.chat_models import ChatOpenAI
from langchain.utilities.zapier import ZapierNLAWrapper
from langchain.agents.agent_toolkits import ZapierToolkit
import streamlit as st
from streamlit_chat import message
from langchain.utilities import GoogleSerperAPIWrapper

llm = ChatOpenAI(temperature=0, model="gpt-4-0613", openai_api_key = OPENAI_API_KEY)

search = GoogleSerperAPIWrapper(serper_api_key=SERPER_API_KEY)
llm_math_chain = LLMMathChain.from_llm(llm=llm, verbose=True)

zapier = ZapierNLAWrapper(zapier_nla_api_key = ZAPIER_NLA_API_KEY)
zap_toolkit = ZapierToolkit.from_zapier_nla_wrapper(zapier)
zap_toolkit.get_tools()[0].name = 'Gmail_Send_Email'
zap_toolkit.get_tools()[1].name = 'Gmail_Reply_to_Email'
zap_toolkit.get_tools()[2].name = 'Gmail_Find_Email'
zap_toolkit.get_tools()[0].description = 'Zapier NLA: natural language tool to send an email via Gmail. Params include: Body, To, Cc, Subject. Requires params in instructions. Stops execution if missing info or null response.'
zap_toolkit.get_tools()[1].description = 'Zapier NLA: natural language tool for replying to an email in Gmail. Params: To, Thread, Body. Requires params in instructions. Stops execution if missing info or null response.'
zap_toolkit.get_tools()[2].description = 'Zapier NLA: natural language tool to find an email in Gmail. Search_String is the required parameter. Instruction must include it. Stops execution if missing info or null response.'

tools = [
    Tool(
        name="Search",
        func=search.run,
        description="useful for when you need to answer questions about current events. You should ask targeted questions",
    ),
    Tool(
        name="Calculator",
        func=llm_math_chain.run,
        description="useful for when you need to answer questions about math",
    )
    ]


# Initialize markdown area
markdown_area = st.empty()

# Define tools
def display_markdown_text(markdown_text):
    markdown_area.markdown(markdown_text)

markdown_tool = Tool(
    name="MarkdownDisplay",
    func=display_markdown_text,
    description="useful for displaying markdown text",
)

tools.append(markdown_tool)

from langchain.prompts import MessagesPlaceholder
from langchain.memory import ConversationBufferMemory

agent_kwargs = {
    "extra_prompt_messages": [MessagesPlaceholder(variable_name="memory")],
}

#memory = ConversationBufferMemory(memory_key="memory", return_messages=True)

if "memory" not in st.session_state:
    st.session_state['memory'] = ConversationBufferMemory(
        memory_key="memory",
        return_messages=True
        )

agent = initialize_agent(
    tools + zap_toolkit.get_tools(),
    llm,
    agent=AgentType.OPENAI_MULTI_FUNCTIONS,
    verbose=True,
    agent_kwargs=agent_kwargs,
    memory=st.session_state['memory'],  
)


st.header("AI Assistent")

if 'generated' not in st.session_state:
    st.session_state['generated'] = []

if 'past' not in st.session_state:
    st.session_state['past'] = []

import PyPDF2

def upload_pdf_and_get_text():
    uploaded_file = st.file_uploader("Upload a PDF", type="pdf")
    process_button = st.button("Process PDF")
    if process_button:
        if uploaded_file is not None:
            # Process the PDF only if it's the first time (i.e., not already in st.session_state)
            if 'processed_pdf_text' not in st.session_state:
                pdf_reader = PyPDF2.PdfFileReader(uploaded_file)
                text = ""
                for page_num in range(pdf_reader.getNumPages()):
                    page = pdf_reader.getPage(page_num)
                    text += page.extractText()

                # Storing the processed pdf text in the session state
                st.session_state['processed_pdf_text'] = text

def get_text():
    input_text = st.text_input(" ","Hello, how are you?", key="input")
    send_button = st.button("Send")
    return input_text, send_button 

# Call the function 
upload_pdf_and_get_text()

# If the processed pdf text is in the session state (i.e., a file was uploaded and processed)
if 'processed_pdf_text' in st.session_state:
    pdf_message = "I just received this document, can you summarize it\n" + st.session_state['processed_pdf_text']
    output = agent.run(pdf_message)
    st.session_state.past.append("Here is a document i want you to have a look at. Can you summarize its content?")
    st.session_state.generated.append(output)
    # clear the processed pdf text as it's already handled
    del st.session_state['processed_pdf_text']
    

# Add this condition to avoid processing the text input when a PDF is uploaded
if 'processed_pdf_text' not in st.session_state:
    user_input, send_button = get_text()
    if send_button: 
        if user_input:
            output = agent.run(user_input)
            st.session_state.past.append(user_input)
            st.session_state.generated.append(output)


if st.session_state['generated']:
    for i in range(len(st.session_state['generated'])-1, -1, -1):
        message(st.session_state['generated'][i], key=str(i), avatar_style='bottts', seed = 5)
        message(st.session_state['past'][i], is_user=True, avatar_style='miniavs', seed = 8, key=str(i) + '_user')
