import os
import logging
import argparse
import json
from PyPDF2 import PdfReader
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import AgentExecutor, create_react_agent
from langchain.agents.format_scratchpad import format_log_to_str
from langchain.agents.output_parsers import ReActSingleInputOutputParser
from langchain.tools.render import render_text_description
from langchain.tools import BaseTool
from langchain.prompts import PromptTemplate

# Logging settings
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load .env file
load_dotenv()

# Get Gemini API key
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

class PaperValidatorTool(BaseTool):
    name = "paper_validator"
    description = "Validates if a given PDF is a computational science article and checks its development information"
    
    def _run(self, pdf_path: str) -> str:
        try:
            # Clean and validate the file path
            pdf_path = pdf_path.split('\n')[0].strip()  # Take only the first line and clean it
            
            if not os.path.exists(pdf_path):
                raise FileNotFoundError(f"PDF file not found: {pdf_path}")
            
            if not pdf_path.lower().endswith('.pdf'):
                raise ValueError("File is not a PDF")
            
            # Read PDF content
            reader = PdfReader(pdf_path)
            text_content = ""
            for page in reader.pages:
                text_content += page.extract_text()

            if not text_content.strip():
                raise ValueError("PDF content is empty")

            # Create LLM instance for validation
            llm = ChatGoogleGenerativeAI(
                model="gemini-2.0-pro-exp-02-05",
                google_api_key=GOOGLE_API_KEY,
                convert_system_message_to_human=True,
                temperature=0.1
            )
            
            # Prepare the prompt
            validation_prompt = """You are a scientific paper analyzer. Your task is to analyze the given paper content and provide a structured evaluation.

IMPORTANT: You must respond with ONLY a valid JSON object. No other text, no explanations, no markdown formatting.

Here's the paper content to analyze:
---
{text_content}
---

Respond with this exact JSON structure, replacing the placeholders with appropriate values:
{{
    "is_computational_science": <true/false>,
    "has_development_info": <true/false>,
    "paper_type": "<type of the paper>",
    "development_details": {{
        "methodology_explained": <true/false>,
        "implementation_details": <true/false>,
        "tools_frameworks_mentioned": [<list of tools/frameworks>],
        "missing_critical_info": [<list of missing information>]
    }},
    "summary": "<brief summary of the paper>",
    "recommendations": [<list of recommendations>]
}}

Remember:
1. Respond ONLY with the JSON object
2. Use true/false for boolean values (not strings)
3. Keep string values concise
4. Ensure valid JSON format
5. No text outside the JSON structure"""

            # Get LLM response and clean it
            response = llm.invoke(validation_prompt.format(text_content=text_content[:15000]))
            cleaned_response = response.content.strip()
            
            # Try to find JSON content if there's any extra text
            try:
                # First try direct parsing
                return json.dumps(json.loads(cleaned_response), indent=2)
            except json.JSONDecodeError:
                # Try to find JSON-like content
                import re
                json_match = re.search(r'({[\s\S]*})', cleaned_response)
                if json_match:
                    try:
                        json_content = json_match.group(1)
                        return json.dumps(json.loads(json_content), indent=2)
                    except json.JSONDecodeError:
                        raise ValueError("Could not parse JSON from LLM response")
                else:
                    raise ValueError("No valid JSON found in LLM response")
            
        except Exception as e:
            logger.error(f"Error in PaperValidatorTool: {str(e)}")
            error_response = {
                "error": str(e),
                "is_computational_science": False,
                "has_development_info": False,
                "paper_type": "unknown",
                "development_details": {
                    "methodology_explained": False,
                    "implementation_details": False,
                    "tools_frameworks_mentioned": [],
                    "missing_critical_info": ["Failed to process paper"]
                },
                "summary": "Error processing paper",
                "recommendations": ["Please check if the PDF file is valid and try again"]
            }
            return json.dumps(error_response, indent=2)
    
    async def _arun(self, query: str) -> str:
        raise NotImplementedError("Async version not implemented")

class HelloTool(BaseTool):
    name = "hello_tool"
    description = "Returns a simple hello message"
    
    def _run(self, query: str) -> str:
        return "Hello! I'm a response from a tool!"
    
    async def _arun(self, query: str) -> str:
        raise NotImplementedError("async version not implemented yet")

def create_agent(name: str):
    # Create LLM model
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-pro-exp-02-05",
        google_api_key=GOOGLE_API_KEY,
        convert_system_message_to_human=True
    )
    
    # Define tools based on agent name
    if name == "PaperReader":
        tools = [PaperValidatorTool()]
        template = """You are a Paper Validation Assistant specialized in computational science papers.
        Your task is to analyze PDF papers and determine if they are computational science articles with sufficient development information.
        
        Available tools:
        {tool_names}
        
        Tools and their descriptions:
        {tools}
        
        User Input: {input}
        {agent_scratchpad}
        
        Follow these steps exactly:
        1. Extract the PDF file path from the user input
        2. Use the paper_validator tool with ONLY the file path
        3. Analyze the JSON response
        4. Provide a clear summary
        
        Use this format EXACTLY:
        Action: paper_validator
        Action Input: [PDF_FILE_PATH]
        
        Observation: [TOOL_RESPONSE]
        
        Thought: Analyzing the validation results...
        Final Answer: [Clear summary of the findings]
        """
    else:
        tools = [HelloTool()]
        template = """You are an AI assistant. Use exactly one tool to respond to the user's request.

Available tools:
{tool_names}

Tools and their descriptions:
{tools}

User Input: {input}
{agent_scratchpad}

Use this format:
Action: tool_name
Action Input: input to tool

Observation: tool response

Thought: Based on the tool response, I will provide a final answer.
Final Answer: [Your response here]
"""
    
    prompt = PromptTemplate(
        template=template,
        input_variables=["tools", "tool_names", "input", "agent_scratchpad"]
    )
    
    # Create agent
    agent = create_react_agent(llm, tools, prompt)
    
    # Create Agent Executor with max iterations
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        max_iterations=1,
        handle_parsing_errors=True
    )
    
    return agent_executor

def validate_pdf_file(file_path: str) -> bool:
    if not os.path.exists(file_path):
        return False
    if not file_path.lower().endswith('.pdf'):
        return False
    return True

def main():
    # Create argument parser
    parser = argparse.ArgumentParser(description='PDF file processing tool')
    parser.add_argument('pdf_file', nargs='?', help='Path to the PDF file to process')
    args = parser.parse_args()

    logger.info("Starting application...")

    # PDF file validation
    if not args.pdf_file:
        logger.error("Error: No PDF file specified!")
        print("Usage: python main.py <pdf_file>")
        return

    # Clean the file path by removing any whitespace or newline characters
    pdf_file_path = args.pdf_file.strip()

    if not validate_pdf_file(pdf_file_path):
        logger.error(f"Error: Invalid PDF file or file not found: {pdf_file_path}")
        return

    logger.info(f"Processing PDF file: {pdf_file_path}")
    
    # Create agent and process the PDF
    try:
        agent = create_agent("PaperReader")
        logger.info("Paper validation agent created")
        
        # Send PDF processing request to agent
        logger.info("Starting paper validation...")
        response = agent.invoke({"input": pdf_file_path})
        
        # Print the response in a formatted way
        print("\nPaper Validation Results:")
        print("-" * 50)
        print(response['output'])
        print("-" * 50)
        
    except Exception as e:
        logger.error(f"An error occurred during paper validation: {str(e)}")
        print(f"\nError: Failed to validate paper - {str(e)}")

if __name__ == "__main__":
    main()
