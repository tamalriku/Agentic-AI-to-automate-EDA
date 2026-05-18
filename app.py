import os
import sys
import shutil
import subprocess
import pandas as pd
import streamlit as st
from typing import TypedDict
from langchain_openai import OpenAI
from langgraph.graph import StateGraph, END

# Streamlit Page Config
st.set_page_config(page_title="Agentic EDA", page_icon="🤖", layout="wide", initial_sidebar_state="collapsed")

# Custom CSS for Premium Aesthetics
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    .stApp {
        background-color: #0d1117;
        color: #e6edf3;
    }
    
    /* Headers */
    h1, h2, h3 {
        color: #58a6ff !important;
        font-weight: 600 !important;
    }
    
    /* File Uploader Container */
    [data-testid="stFileUploadDropzone"] {
        background-color: #161b22;
        border: 2px dashed #30363d;
        border-radius: 12px;
        padding: 2rem;
        transition: all 0.3s ease;
    }
    
    [data-testid="stFileUploadDropzone"]:hover {
        border-color: #58a6ff;
        background-color: #1c2128;
    }
    
    /* Buttons */
    .stButton>button {
        background: linear-gradient(135deg, #238636 0%, #2ea043 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.75rem 1.5rem;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 12px rgba(46, 160, 67, 0.3);
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 16px rgba(46, 160, 67, 0.4);
        background: linear-gradient(135deg, #2ea043 0%, #3fb950 100%);
    }
    
    /* Expanders */
    .streamlit-expanderHeader {
        background-color: #161b22;
        border-radius: 8px;
        border: 1px solid #30363d;
    }
</style>
""", unsafe_allow_html=True)

# Initialize the model running locally via LM Studio
@st.cache_resource
def get_llm():
    return OpenAI(
        base_url="http://localhost:1234/v1", 
        api_key="lm-studio",                 
        temperature=0.2,
        max_tokens=2048
    )

# Define State
class AgentState(TypedDict):
    schema_info: str
    file_path: str
    file_name: str
    code: str
    output: str
    error: str
    report: str
    iterations: int

# --- Nodes ---
def coding_agent(state: AgentState):
    schema_info = state["schema_info"]
    file_name = state["file_name"]
    error = state.get("error", "")
    llm = get_llm()
    
    if error:
        prompt = f"""
        You are an expert Python data scientist.
        You previously wrote Pandas code for this dataset, but it failed with this error:
        {error}

        Here is the dataset metadata again:
        {schema_info}

        Please provide the corrected Python code. The code should:
        1. Read '{file_name}'
        2. Generate summary statistics
        3. Print out useful insights (e.g. using print statements)
        Do NOT try to show plots, just text output.
        Only return the executable Python code block.
        """
    else:
        prompt = f"""
        You are an expert Python data scientist.
        Here is the dataset metadata:
        {schema_info}

        Write a Python script using pandas to:
        1. Read '{file_name}'
        2. Print the shape and columns
        3. Print summary statistics using describe()
        4. Check for any missing values and print the result
        
        Do NOT generate plots, only print text summaries.
        Only return the executable Python code block.
        """
        
    generated_code = llm.invoke(prompt)
    return {"code": generated_code.replace('\r', '')}

def execution_agent(state: AgentState):
    iterations = state.get("iterations", 0) + 1
    code = state["code"]
    file_path = state["file_path"]
    file_name = state["file_name"]
    
    if "```python" in code:
        code_block = code.split("```python")[1].split("```")[0].strip()
    elif "```" in code:
        code_block = code.split("```")[1].split("```")[0].strip()
    else:
        code_block = code.strip()
        
    sandbox_dir = "sandbox"
    os.makedirs(sandbox_dir, exist_ok=True)
    
    if os.path.exists(file_path):
        shutil.copy(file_path, os.path.join(sandbox_dir, file_name))
        
    script_path = os.path.join(sandbox_dir, "execute.py")
    with open(script_path, "w", encoding="utf-8") as f:
        f.write(code_block)
        
    safe_env = {
        "PATH": os.environ.get("PATH", ""),
        "SYSTEMROOT": os.environ.get("SYSTEMROOT", "")
    }
    
    try:
        result = subprocess.run(
            [sys.executable, "execute.py"],
            cwd=sandbox_dir,
            env=safe_env,
            capture_output=True,
            text=True,
            timeout=30
        )
        if result.returncode == 0:
            return {"output": result.stdout, "error": "", "iterations": iterations}
        else:
            return {"output": "", "error": result.stderr, "iterations": iterations}
    except subprocess.TimeoutExpired:
        return {"output": "", "error": "Execution timed out after 30 seconds.", "iterations": iterations}
    except Exception as e:
        return {"output": "", "error": str(e), "iterations": iterations}

def interpreting_agent(state: AgentState):
    output = state.get("output", "")
    error = state.get("error", "")
    llm = get_llm()
    
    if error:
        prompt = f"The execution failed after maximum retries. The last error was:\n{error}\nPlease write a short markdown report explaining the failure."
    else:
        prompt = f"""
        You are a senior data scientist.
        Here is the output from our Python data analysis script:
        {output}

        Please write a clear, concise Markdown report summarizing the key findings, data quality issues, and potential insights.
        """
        
    report = llm.invoke(prompt)
    return {"report": report.replace('\r', '')}

def should_continue(state: AgentState):
    if state.get("error") and state.get("iterations", 0) < 3:
        return "coding_agent"
    return "interpreting_agent"

@st.cache_resource
def build_graph():
    workflow = StateGraph(AgentState)
    workflow.add_node("coding_agent", coding_agent)
    workflow.add_node("execution_agent", execution_agent)
    workflow.add_node("interpreting_agent", interpreting_agent)
    workflow.set_entry_point("coding_agent")
    workflow.add_edge("coding_agent", "execution_agent")
    workflow.add_conditional_edges(
        "execution_agent",
        should_continue,
        {
            "coding_agent": "coding_agent",
            "interpreting_agent": "interpreting_agent"
        }
    )
    workflow.add_edge("interpreting_agent", END)
    return workflow.compile()

# --- Streamlit UI ---
st.title("🤖 Agentic AI Pipeline for Automating EDA")
st.markdown("Upload a CSV dataset and our multi-agent system will write code, safely run it, and generate a statistical report!")

uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file is not None:
    st.success(f"Uploaded `{uploaded_file.name}`")
    
    # Save file to a temporary directory
    upload_dir = "uploaded_data"
    os.makedirs(upload_dir, exist_ok=True)
    file_path = os.path.join(upload_dir, uploaded_file.name)
    
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
        
    # Show preview
    try:
        df = pd.read_csv(file_path, index_col=False)
        st.subheader("Data Preview")
        st.dataframe(df.head())
        
        # Run Button
        if st.button("Run Agentic EDA", type="primary"):
            with st.spinner("Generating Report using Multi-Agent Workflow..."):
                schema_info_df = pd.DataFrame({
                    "Column Name": df.columns,
                    "Data Type": df.dtypes.values,
                    "Missing Values": df.isnull().sum().values
                })
                
                analysis_summary = f"""
                Dataset Shape: {df.shape}
                
                Schema Information:
                {schema_info_df.to_string()}
                """
                
                initial_state = {
                    "schema_info": analysis_summary,
                    "file_path": file_path,
                    "file_name": uploaded_file.name,
                    "iterations": 0
                }
                
                app = build_graph()
                final_state = app.invoke(initial_state)
                
                st.subheader("Final Data Analysis Report")
                st.markdown(final_state.get("report", "No report generated."))
                
                with st.expander("Show Generated Python Code"):
                    st.code(final_state.get("code", "No code generated."), language="python")
                    
                with st.expander("Show Raw Execution Output"):
                    if final_state.get("error"):
                        st.error(final_state.get("error"))
                    else:
                        st.text(final_state.get("output"))
                        
    except Exception as e:
        st.error(f"Error reading CSV: {e}")
