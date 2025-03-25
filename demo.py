import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.chains import create_extraction_chain
from langchain_openai import ChatOpenAI
import os
import csv
from langchain.agents.agent_types import AgentType
from langchain_experimental.agents.agent_toolkits import create_csv_agent
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain import VectorDBQA
from langchain.llms import OpenAI

# Set OpenAI API key
os.environ["OPENAI_API_KEY"] = "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"

# Schema
schema = {
    "properties": {
        "Parties Involved": {"type": "string"},
        "Scope of work or services": {"type": "string"},
        "Start Date": {"type": "string"},
        "Termination Date/End Date/ Finishing Date/ Exit Date/ Expiry Date": {"type": "string"},
        "Confidentiality": {"type": "string"},
        "Warranties and disclaimers": {"type": "string"},
        "Termination clauses": {"type": "string"},
        "Dispute resolution": {"type": "string"},
        "Notice Period": {"type": "string"},
        "Force Majeure": {"type": "string"},
        "Penalty":{"type":"string"},
        "Payment structure":{"type":"string"}
    },
}

# Streamlit app
def main():
    st.title("Contract Analysis")

    # Tabs
    tabs = ["Upload Contract", "Analysis","Chat with your contracts"]
    selected_tab = st.sidebar.radio("Select Tab", tabs)

    if selected_tab == "Upload Contract":
        upload_pdf_tab()
    elif selected_tab == "Analysis":
        chat_csv_agent_tab()
    elif selected_tab == "Chat with your contracts":
        sub_tabs = ["CreditcardscomInc", "CybergyHoldingsInc", "SteelVaultCorp", "UsioInc"]
        selected_sub_tab = st.radio("Select Contract", sub_tabs)
        if selected_sub_tab == "CreditcardscomInc":
            if  os.path.exists("CreditcardscomInc_20070810_S-1_EX-10.33_362297_EX-10.33_Affiliate Agreement"):
                embeddings = HuggingFaceEmbeddings()
                new_db1 = FAISS.load_local("CreditcardscomInc_20070810_S-1_EX-10.33_362297_EX-10.33_Affiliate Agreement", embeddings)
                qa1 = VectorDBQA.from_chain_type(llm=OpenAI(), chain_type="stuff", vectorstore=new_db1, return_source_documents=True,)
                que1 = st.text_input("Enter a question")
                if st.button("Send"):
                    if que1:
                        result = qa1({'query': que1})
                        answer = result['result']
                        st.write(answer)
            else:
                st.write("Embeddings for this file does not exist.To proceed with Q&A first upload a file and create embeddings")
        elif selected_sub_tab == "CybergyHoldingsInc":
            if  os.path.exists("CybergyHoldingsInc_20140520_10-Q_EX-10.27_8605784_EX-10.27_Affiliate Agreement"):
                embeddings = HuggingFaceEmbeddings()
                new_db2 = FAISS.load_local("CybergyHoldingsInc_20140520_10-Q_EX-10.27_8605784_EX-10.27_Affiliate Agreement", embeddings)
                qa2 = VectorDBQA.from_chain_type(llm=OpenAI(), chain_type="stuff", vectorstore=new_db2, return_source_documents=True,)
                que2 = st.text_input("Enter a question")
                if st.button("Send"):
                    if que2:
                        result = qa2({'query': que2})
                        answer = result['result']
                        st.write(answer)
            else:
                st.write("Embeddings for this file does not exist.To proceed with Q&A first upload a file and create embeddings")
        elif selected_sub_tab == "SteelVaultCorp":
            if  os.path.exists("SteelVaultCorp_20081224_10-K_EX-10.16_3074935_EX-10.16_Affiliate Agreement"):
                embeddings = HuggingFaceEmbeddings()
                new_db3 = FAISS.load_local("SteelVaultCorp_20081224_10-K_EX-10.16_3074935_EX-10.16_Affiliate Agreement", embeddings)
                qa3 = VectorDBQA.from_chain_type(llm=OpenAI(), chain_type="stuff", vectorstore=new_db3, return_source_documents=True,)
                que3 = st.text_input("Enter a question")
                if st.button("Send"):
                    if que3:
                        result = qa3({'query': que3})
                        answer = result['result']
                        st.write(answer)
            else:
                st.write("Embeddings for this file does not exist.To proceed with Q&A first upload a file and create embeddings")
        elif selected_sub_tab == "UsioInc":
            if  os.path.exists("UsioInc_20040428_SB-2_EX-10.11_1723988_EX-10.11_Affiliate Agreement 2"):
                embeddings = HuggingFaceEmbeddings()
                new_db4 = FAISS.load_local("UsioInc_20040428_SB-2_EX-10.11_1723988_EX-10.11_Affiliate Agreement 2", embeddings)
                qa4 = VectorDBQA.from_chain_type(llm=OpenAI(), chain_type="stuff", vectorstore=new_db4, return_source_documents=True,)
                que4 = st.text_input("Enter a question")
                if st.button("Send"):
                    if que4:
                        result = qa4({'query': que4})
                        answer = result['result']
                        st.write(answer)
            else:
                st.write("Embeddings for this file does not exist.To proceed with Q&A first upload a file and create embeddings")

def upload_pdf_tab():
    # File upload
    uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

    if uploaded_file:
        file_path = save_uploaded_file(uploaded_file)

        # Display file details
        st.subheader("File details:")
        st.write(uploaded_file.name)
        man = uploaded_file.name
        name_only = man.replace(".pdf", "")



        # Process PDF and extract information
        loader = PyPDFLoader(file_path)
        pages = loader.load_and_split()
        # st.write(pages)
        llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-16k")
        chain = create_extraction_chain(schema, llm)
        chunk_details = chain.run(pages)

        # Display extracted information
        st.subheader("Extracted Information:")
        st.write(chunk_details)

        # Save to CSV
        save_csv(chunk_details)

        if not os.path.exists(name_only):
            
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
            docs = text_splitter.split_documents(pages)

            embeddings = HuggingFaceEmbeddings()

            db = FAISS.from_documents(docs, embeddings)

            db.save_local(name_only)

            st.success(f"Pickle file saved successfully at {name_only}")
        else:
            st.write("Pickle file exists.So, not proceeding with embeddings.")



def chat_csv_agent_tab():
    # Load CSV file
    csv_file_path = 'godrej.csv'
    if os.path.exists(csv_file_path):
        st.subheader("Chat with CSV Agent")
        
        # Create CSV agent
        agent = create_csv_agent(
            ChatOpenAI(temperature=0, model="gpt-3.5-turbo-0613"),
            csv_file_path,
            verbose=True,
            agent_type=AgentType.OPENAI_FUNCTIONS,

        )

        # Chat input
        que = st.text_input("Enter the question")
        if st.button("Send"):
            if que:
                # Run agent only when the user inputs a question
                answer = agent.run(que)
                st.write(answer)

    else:
        st.write("No csv file is present, please proceed with file uploading.")


def save_uploaded_file(uploaded_file):
    # Save the uploaded file to a temporary location
    temp_location = f"/home/manikanta/mani/Manikantaworkspace_office/godrejclient/godrej{uploaded_file.name}"
    with open(temp_location, "wb") as temp_file:
        temp_file.write(uploaded_file.read())
    return temp_location

def save_csv(chunk_details):
    # Specify the CSV file path
    csv_file_path = 'godrej.csv'


    file_exists = os.path.exists(csv_file_path)

    if not file_exists:
        # Open the CSV file in write mode
        with open(csv_file_path, 'a', newline='') as csv_file:
            # Create a CSV writer object
            csv_writer = csv.DictWriter(csv_file, fieldnames=chunk_details.keys())

            # Write the header
            csv_writer.writeheader()

            # Write the data
            csv_writer.writerow(chunk_details)

            st.success(f"CSV file saved successfully at {csv_file_path}")

    else:
        with open(csv_file_path, 'a', newline='') as csv_file:

        # Create a CSV writer object
            csv_writer = csv.DictWriter(csv_file, fieldnames=chunk_details.keys())
            # Write the data
            csv_writer.writerow(chunk_details)

            st.success(f"CSV file saved successfully at {csv_file_path}")




if __name__ == "__main__":
    main()
