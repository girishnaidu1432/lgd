import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.chains import create_extraction_chain
import openai
import os
import csv
import tempfile
import json

# ✅ Azure OpenAI Configuration
openai.api_key = "14560021aaf84772835d76246b53397a"
openai.api_base = "https://amrxgenai.openai.azure.com/"
openai.api_type = 'azure'
openai.api_version = '2024-02-15-preview'
deployment_name = 'gpt'

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
        "Penalty": {"type": "string"},
        "Payment structure": {"type": "string"}
    },
}

# Streamlit app
def main():
    st.title("📑 Contract Analysis with Azure OpenAI")

    # Tabs
    tabs = ["Upload Contract", "Analysis", "Chat with Extracted Data"]
    selected_tab = st.sidebar.radio("Select Tab", tabs)

    if selected_tab == "Upload Contract":
        upload_pdf_tab()
    elif selected_tab == "Analysis":
        chat_csv_agent_tab()
    elif selected_tab == "Chat with Extracted Data":
        chat_with_data_tab()


# ✅ Upload PDF and extract data
def upload_pdf_tab():
    uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

    if uploaded_file:
        # Save the uploaded PDF file locally
        file_path = save_uploaded_file(uploaded_file)
        
        st.subheader("File details:")
        st.write(uploaded_file.name)

        # Extract text from PDF
        loader = PyPDFLoader(file_path)
        
        try:
            pages = loader.load_and_split()
            
            # Azure OpenAI Extraction
            extracted_data = extract_data_with_azure(pages)

            # Store extracted data in session state
            st.session_state["extracted_data"] = extracted_data

            # Display extracted information
            st.subheader("Extracted Information:")
            st.write(extracted_data)

            # Save to CSV
            save_csv(extracted_data)
            st.success("Data saved and stored in session state!")
        
        except Exception as e:
            st.error(f"Failed to process the PDF: {e}")


# ✅ Extract data using Azure OpenAI
def extract_data_with_azure(pages):
    """Extracts structured data from PDF pages using Azure OpenAI."""
    
    prompt = f"Extract the following fields: {json.dumps(schema, indent=2)}\n\n"
    prompt += "\n".join([page.page_content for page in pages])

    # Azure OpenAI request
    response = openai.ChatCompletion.create(
        engine=deployment_name,
        messages=[
            {"role": "system", "content": "You are an AI that extracts contract information."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3,
        max_tokens=2000
    )

    result = response['choices'][0]['message']['content']
    
    try:
        return json.loads(result)
    except json.JSONDecodeError:
        return {"error": "Failed to parse extracted data"}


# ✅ Chat with extracted data tab
def chat_with_data_tab():
    if "extracted_data" in st.session_state:
        st.subheader("💬 Chatbot with Extracted Data")
        
        # Chat input
        user_query = st.text_input("Ask about the results")

        if st.button("Ask"):
            data = st.session_state["extracted_data"]
            
            # Prepare prompt with history
            if "history" not in st.session_state:
                st.session_state.history = []

            st.session_state.history.append({"role": "user", "content": user_query})
            
            prompt = json.dumps(data, indent=2) + "\n" + user_query

            # Azure OpenAI ChatCompletion
            response = openai.ChatCompletion.create(
                engine=deployment_name,
                messages=[
                    {"role": "system", "content": prompt}
                ],
                temperature=0.5,
                max_tokens=500
            )

            st.session_state.history.append({"role": "assistant", "content": response['choices'][0]['message']['content']})
            
            # Display response
            st.write(response['choices'][0]['message']['content'])

    else:
        st.write("No extracted data available. Upload a contract first.")


# ✅ Chat with CSV agent tab
def chat_csv_agent_tab():
    csv_file_path = 'godrej.csv'

    if os.path.exists(csv_file_path):
        st.subheader("📊 CSV Data Analysis")

        # Read CSV into memory
        with open(csv_file_path, 'r') as csv_file:
            reader = csv.DictReader(csv_file)
            rows = list(reader)

        que = st.text_input("Search CSV Data")

        if st.button("Search CSV"):
            if que:
                results = [row for row in rows if any(que.lower() in str(value).lower() for value in row.values())]

                if results:
                    st.write("Matching Rows:")
                    for result in results:
                        st.write(result)
                else:
                    st.write("No matches found.")
    else:
        st.write("No CSV file found. Please upload a contract first.")


# ✅ Save uploaded PDF to a temporary file and return the path
def save_uploaded_file(uploaded_file):
    """Saves the uploaded PDF to a temporary file and returns the file path."""
    
    # Create a temporary file to store the uploaded PDF
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(uploaded_file.getbuffer())
        temp_path = temp_file.name
    
    return temp_path


# ✅ Save extracted data to CSV
def save_csv(data):
    csv_file_path = 'godrej.csv'
    
    file_exists = os.path.exists(csv_file_path)

    with open(csv_file_path, 'a', newline='') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=schema['properties'].keys())

        # Write header if the file is new
        if not file_exists:
            writer.writeheader()

        writer.writerow(data)
        st.success(f"Data saved to CSV: {csv_file_path}")


# ✅ Main execution
if __name__ == "__main__":
    main()
