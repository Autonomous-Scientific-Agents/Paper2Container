import os
import logging
import argparse
import json
import string
import random
from pathlib import Path
from PyPDF2 import PdfReader
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import AgentExecutor, create_react_agent
from langchain.agents.format_scratchpad import format_log_to_str
from langchain.agents.output_parsers import ReActSingleInputOutputParser
from langchain.tools.render import render_text_description
from langchain.tools import BaseTool
from langchain.prompts import PromptTemplate
import re

# Logging settings
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load .env file
load_dotenv()

# Get Gemini API key
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

class WorkspaceCreatorTool(BaseTool):
    name = "workspace_creator"
    description = "Creates a workspace with Dockerfile and docker-compose.yml based on development details"
    
    def _run(self, development_details: str) -> str:
        try:
            # Parse development details
            details = json.loads(development_details)
            
            # Create random workspace name
            workspace_name = ''.join(random.choices(string.ascii_letters + string.digits, k=8))
            workspace_path = os.path.join('workspaces', workspace_name)
            
            # Create workspace directory
            os.makedirs(workspace_path, exist_ok=True)
            
            # Create LLM instance for file generation
            llm = ChatGoogleGenerativeAI(
                model="gemini-pro",
                google_api_key=GOOGLE_API_KEY,
                convert_system_message_to_human=True,
                temperature=0.1
            )
            
            # Generate Dockerfile
            dockerfile_prompt = """Create a Dockerfile based on the following development details:
            {details}
            
            The Dockerfile should:
            1. Use appropriate base image
            2. Install all necessary tools and frameworks
            3. Set up the development environment
            4. Follow best practices
            
            Respond with ONLY the Dockerfile content, no explanations."""
            
            dockerfile_response = llm.invoke(dockerfile_prompt.format(details=json.dumps(details, indent=2)))
            
            # Generate docker-compose.yml
            compose_prompt = """Create a docker-compose.yml file based on the following development details:
            {details}
            
            The docker-compose.yml should:
            1. Define necessary services
            2. Set up appropriate volumes
            3. Configure networking
            4. Include any required environment variables
            
            Respond with ONLY the docker-compose.yml content, no explanations."""
            
            compose_response = llm.invoke(compose_prompt.format(details=json.dumps(details, indent=2)))
            
            # Write files
            with open(os.path.join(workspace_path, 'Dockerfile'), 'w') as f:
                f.write(dockerfile_response.content)
            
            with open(os.path.join(workspace_path, 'docker-compose.yml'), 'w') as f:
                f.write(compose_response.content)
            
            return json.dumps({
                "status": "success",
                "workspace_path": workspace_path,
                "files_created": ["Dockerfile", "docker-compose.yml"]
            }, indent=2)
            
        except Exception as e:
            logger.error(f"Error in WorkspaceCreatorTool: {str(e)}")
            return json.dumps({
                "status": "error",
                "error": str(e)
            }, indent=2)
    
    async def _arun(self, query: str) -> str:
        raise NotImplementedError("Async version not implemented")

class PaperValidatorTool(BaseTool):
    name = "paper_validator"
    description = "Validates if a given PDF is a computational science article and checks its development information"
    
    def _run(self, pdf_path: str) -> str:
        try:
            # Clean and validate the file path
            if isinstance(pdf_path, list):
                pdf_path = pdf_path[0]  # Take first element if it's a list
            pdf_path = str(pdf_path).strip()  # Convert to string and clean
            pdf_path = pdf_path.replace('"', '').replace("'", "")  # Remove quotes
            
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

            # Create LLM instance for validation with increased timeout
            # Add gRPC environment variable at the top after imports
            os.environ['GRPC_ENABLE_FORK_SUPPORT'] = '1'
            os.environ['GRPC_POLL_STRATEGY'] = 'epoll1'
            
            # Update timeout and add retry logic in PaperValidatorTool._run
            llm = ChatGoogleGenerativeAI(
                model="gemini-pro",
                google_api_key=GOOGLE_API_KEY,
                convert_system_message_to_human=True,
                temperature=0.1,
                timeout=180,  # Increased timeout to 3 minutes
                max_retries=3  # Add retries for resilience
            )
            
            # Log the content length for debugging
            logger.info(f"Processing PDF content of length: {len(text_content)}")
            
            # Prepare the prompt
            validation_prompt = """Analyze this scientific paper content and provide a structured evaluation.
            
IMPORTANT: Respond with ONLY a valid JSON object. No other text or explanations.

Paper content:
{text_content}

Required JSON structure:
{{
    "is_computational_science": <true/false>,
    "has_development_info": <true/false>,
    "paper_type": "<type>",
    "development_details": {{
        "methodology_explained": <true/false>,
        "implementation_details": <true/false>,
        "tools_frameworks_mentioned": [<list>],
        "missing_critical_info": [<list>]
    }},
    "summary": "<brief>",
    "recommendations": [<list>]
}}"""

            # Get LLM response
            response = llm.invoke(validation_prompt.format(text_content=text_content[:15000]))
            
            if not response or not response.content:
                raise ValueError("Empty response from LLM")
                
            # Log raw response for debugging
            logger.debug(f"Raw LLM response: {response.content}")
            
            # Clean and parse response
            # Improve JSON response handling
            try:
                # Direct JSON parsing with better error handling
                cleaned_response = response.content.strip()
                
                # First try: direct parsing after basic cleaning
                try:
                    # Remove any leading/trailing whitespace and common formatting characters
                    cleaned_response = re.sub(r'^[\s\n\r]*|[\s\n\r]*$', '', cleaned_response)
                    cleaned_response = re.sub(r'^["\'`]|["\'`]$', '', cleaned_response)
                    json_obj = json.loads(cleaned_response)
                except json.JSONDecodeError:
                    # Second try: remove markdown and code block markers
                    cleaned_response = re.sub(r'```(?:json)?\s*', '', cleaned_response, flags=re.DOTALL)
                    cleaned_response = re.sub(r'`.*?`', '', cleaned_response)
                    cleaned_response = re.sub(r'[\n\r\t]+', ' ', cleaned_response)
                    
                    try:
                        json_obj = json.loads(cleaned_response)
                    except json.JSONDecodeError:
                        # Third try: extract JSON using more aggressive pattern matching
                        json_patterns = [
                            r'\{[^{}]*\}',  # Simple JSON object
                            r'\{(?:[^{}]|\{[^{}]*\})*\}',  # Nested JSON object
                            r'\{(?:[^{}]|\{(?:[^{}]|\{[^{}]*\})*\})*\}'  # Deeply nested JSON
                        ]
                        
                        for pattern in json_patterns:
                            matches = re.finditer(pattern, cleaned_response)
                            for match in matches:
                                try:
                                    potential_json = match.group(0)
                                    json_obj = json.loads(potential_json)
                                    if all(key in json_obj for key in ['is_computational_science', 'has_development_info', 'paper_type']):
                                        return json.dumps(json_obj, indent=2)
                                except:
                                    continue
                        
                        raise ValueError("Could not extract valid JSON from response")
                
                return json.dumps(json_obj, indent=2)
                
            except Exception as e:
                logger.error(f"Error in JSON parsing: {str(e)}")
                raise ValueError(f"Failed to parse response: {str(e)}")
            except json.JSONDecodeError as je:
                logger.warning(f"Initial JSON parsing failed: {str(je)}")
                
                # Try to extract JSON using regex
                json_patterns = [
                    r'({[\s\S]*?})\s*$',  # End of string
                    r'({[\s\S]*})',        # Any JSON-like structure
                    r'{[^}]*}'             # Simple JSON object
                ]
                
                for pattern in json_patterns:
                    try:
                        match = re.search(pattern, cleaned_response)
                        if match:
                            json_str = match.group(1)
                            json_obj = json.loads(json_str)
                            return json.dumps(json_obj, indent=2)
                    except:
                        continue
                
                # If no valid JSON found, raise error
                raise ValueError(f"Could not extract valid JSON from response: {cleaned_response[:200]}...")
            
        except Exception as e:
            logger.error(f"Error in PaperValidatorTool: {str(e)}")
            logger.error(f"Full error details:", exc_info=True)
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
                "summary": f"Error processing paper: {str(e)}",
                "recommendations": ["Please check if the PDF file is valid and try again"]
            }
            return json.dumps(error_response, indent=2)
    
    async def _arun(self, query: str) -> str:
        raise NotImplementedError("Async version not implemented")

def create_agent(name: str):
    # Create LLM model
    llm = ChatGoogleGenerativeAI(
        model="gemini-pro",
        google_api_key=GOOGLE_API_KEY,
        convert_system_message_to_human=True,
        temperature=0.1
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
        3. If you receive a valid JSON response, return it immediately
        4. Do not make additional validation calls if you already have a valid response
        
        Use this format EXACTLY:
        Action: paper_validator
        Action Input: [PDF_FILE_PATH]
        
        Observation: [TOOL_RESPONSE]
        
        Thought: I have received a valid response and will return it.
        Final Answer: [TOOL_RESPONSE]
        """
    elif name == "WorkspaceCreator":
        tools = [WorkspaceCreatorTool()]
        template = """You are a Workspace Creator Assistant specialized in setting up development environments.
        Your task is to create a workspace with Dockerfile and docker-compose.yml based on the development details.
        
        Available tools:
        {tool_names}
        
        Tools and their descriptions:
        {tools}
        
        User Input: {input}
        {agent_scratchpad}
        
        Follow these steps exactly:
        1. Parse the development details from the input JSON
        2. Use the workspace_creator tool with the development details
        3. Return the workspace creation results immediately after receiving a response
        4. Do not make additional workspace creation calls
        
        Use this format EXACTLY:
        Action: workspace_creator
        Action Input: [DEVELOPMENT_DETAILS_JSON]
        
        Observation: [TOOL_RESPONSE]
        
        Thought: Workspace creation is complete. I will return the results.
        Final Answer: [TOOL_RESPONSE]
        """
    else:
        logger.warning(f"Unknown agent type: {name}")
        return None
    
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
        max_iterations=2,  # Reduced from 5 to 2 since we only need one validation
        max_execution_time=300,  # Reduced timeout to 5 minutes
        handle_parsing_errors=True,
        early_stopping_method="force",  # Changed from generate to force for compatibility
        return_intermediate_steps=True
    )
    
    return agent_executor

def validate_pdf_file(file_path: str) -> bool:
    if not os.path.exists(file_path):
        return False
    if not file_path.lower().endswith('.pdf'):
        return False
    return True

def extract_json_from_text(text: str) -> str:
    """Extract and clean JSON from text."""
    # Remove markdown code blocks and backticks
    text = re.sub(r'```(?:json)?\s*', '', text)
    text = text.replace('`', '')
    
    # Find all potential JSON objects
    json_candidates = []
    start_indices = [m.start() for m in re.finditer(r'{', text)]
    
    for start_idx in start_indices:
        stack = []
        for i, char in enumerate(text[start_idx:], start=start_idx):
            if char == '{':
                stack.append(char)
            elif char == '}':
                stack.pop()
                if not stack:  # Found complete JSON object
                    json_str = text[start_idx:i+1]
                    try:
                        # Clean the JSON string
                        json_str = re.sub(r'[\n\r\t]+', ' ', json_str)
                        json_str = re.sub(r'\s+', ' ', json_str)
                        # Try to parse it
                        json.loads(json_str)
                        json_candidates.append((len(json_str), json_str))
                    except json.JSONDecodeError:
                        continue
    
    if not json_candidates:
        raise ValueError("No valid JSON object found in text")
    
    # Return the longest valid JSON string
    return max(json_candidates, key=lambda x: x[0])[1]

def main():
    # Create argument parser
    parser = argparse.ArgumentParser(description='PDF file processing tool')
    parser.add_argument('pdf_file', nargs='?', help='Path to the PDF file to process')
    args = parser.parse_args()

    logger.info("Starting application...")

    # Create workspaces directory if it doesn't exist
    os.makedirs('workspaces', exist_ok=True)

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
    
    try:
        # First, analyze the paper with PaperReader
        paper_reader = create_agent("PaperReader")
        logger.info("Paper validation agent created")
        
        logger.info("Starting paper validation...")
        paper_response = paper_reader.invoke({"input": pdf_file_path})
        
        if not paper_response:
            raise ValueError("No response received from PaperReader")
            
        # Extract output from paper_response
        output = paper_response.get('output', '')
        
        # Try to find JSON in the output
        validation_result = None
        try:
            # Find the first occurrence of a complete JSON object
            json_pattern = r'({[\s\S]*?})\s*(?:>|\n|$)'
            match = re.search(json_pattern, output)
            if match:
                validation_result = json.loads(match.group(1))
                logger.info("Successfully parsed JSON from output")
        except Exception as e:
            logger.error(f"Failed to parse JSON: {str(e)}")
            print(f"\nError: Failed to parse validation response")
            print(f"Error details: {str(e)}")
            return
            
        if not validation_result:
            logger.error("No valid JSON object found in response")
            print("\nError: Invalid validation response format")
            return
        
        # Try to extract output from paper_response using a more direct approach
        validation_result = None
        
        # First try: Look for a complete JSON object
        try:
            # Find the first occurrence of a complete JSON object
            json_pattern = r'({[\s\S]*?})\s*(?:>|\n|$)'
            match = re.search(json_pattern, output)
            if match:
                potential_json = match.group(1)
                validation_result = json.loads(potential_json)
                logger.info("Successfully parsed JSON directly from output")
        except (json.JSONDecodeError, AttributeError) as e:
            logger.warning(f"First JSON parsing attempt failed: {str(e)}")
        
        # If direct parsing failed, try the existing extraction method
        if not validation_result:
            try:
                validation_result = json.loads(extract_json_from_text(output))
                logger.info("Successfully parsed JSON using extraction method")
            except Exception as e:
                logger.warning(f"JSON extraction failed: {str(e)}")
        
        if not validation_result:
            logger.error("Failed to parse JSON: No valid JSON object found")
            print("\nError: Failed to parse JSON response")
            print("Error details: No valid JSON object found")
            print("\nRaw output:")
            print("-" * 50)
            print(output)
            print("-" * 50)
            return
            
        # Print validation results
        print("\nPaper Validation Results:")
        print("-" * 50)
        print(json.dumps(validation_result, indent=2))
        print("-" * 50)
        
        # Check if we should create workspace
        if validation_result.get("has_development_info", False) and validation_result.get("development_details"):
            logger.info("Paper has sufficient development information. Creating workspace...")
            
            # Extract development details
            dev_details = validation_result.get("development_details")
            
            # Create workspace with WorkspaceCreator
            workspace_creator = create_agent("WorkspaceCreator")
            
            # Prepare development details as a clean JSON string
            development_details = {
                "tools": dev_details.get("tools_frameworks_mentioned", []),
                "implementation_details": dev_details.get("implementation_details", False),
                "methodology_explained": dev_details.get("methodology_explained", False),
                "paper_type": validation_result.get("paper_type", "Unknown"),
                "summary": validation_result.get("summary", ""),
                "missing_critical_info": dev_details.get("missing_critical_info", [])
            }
            
            # Convert to a clean JSON string
            dev_details_str = json.dumps(development_details, ensure_ascii=False, indent=None)
            
            logger.info(f"Sending development details to workspace creator: {dev_details_str}")
            
            # Invoke workspace creator with the clean JSON string
            workspace_response = workspace_creator.invoke({"input": dev_details_str})
            
            # Process workspace response
            if isinstance(workspace_response, dict) and 'output' in workspace_response:
                output = workspace_response['output']
                
                # Try to find JSON in the output using the same pattern as before
                try:
                    json_pattern = r'({[\s\S]*?})\s*(?:>|\n|$)'
                    match = re.search(json_pattern, output)
                    if match:
                        workspace_result = json.loads(match.group(1))
                        logger.info("Successfully parsed workspace creation result")
                    else:
                        workspace_result = {"status": "error", "message": "Could not extract JSON from response"}
                except Exception as e:
                    workspace_result = {"status": "error", "message": f"Error parsing workspace response: {str(e)}"}
            else:
                workspace_result = {"status": "error", "message": "Invalid workspace response format"}
            
            # Print workspace results
            print("\nWorkspace Creation Results:")
            print("-" * 50)
            print(json.dumps(workspace_result, indent=2))
            print("-" * 50)
        else:
            logger.info("Paper does not have sufficient development information")
            print("\nNote: Workspace not created due to insufficient development information")
            print("Development Info:", json.dumps(validation_result.get("development_details", {}), indent=2))
        
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}", exc_info=True)
        print(f"\nError: {str(e)}")

if __name__ == "__main__":
    main()
