import os
import logging
import argparse
import json
import string
import random
import yaml
import subprocess
import shutil
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
from typing import Dict, List, Optional, Tuple, Any
import re
import time
from langchain_community.llms import Ollama
from langchain.schema.language_model import BaseLanguageModel

# Logging settings
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load .env file
load_dotenv()

# Get Gemini API key
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

class WorkspaceDebugger(BaseTool):
    name = "workspace_debugger"
    description = "Debugs and fixes Docker configuration files in a workspace"
    max_retries = 10
    interactive = True
    llm: Optional[BaseLanguageModel] = None
    
    def __init__(self, interactive: bool = True):
        super().__init__()
        self.interactive = interactive
        try:
            # Initialize Ollama LLM
            self.llm = Ollama(
                model="llama3.2",
                base_url="http://localhost:11434",
                temperature=0.1
            )
        except Exception as e:
            logger.error(f"Error initializing LLM: {str(e)}")
            raise

    def _run(self, workspace_path: str) -> str:
        try:
            # Clean and validate workspace path
            workspace_path = str(workspace_path).strip()
            if not os.path.exists(workspace_path):
                raise FileNotFoundError(f"Workspace directory not found: {workspace_path}")
            
            # Initialize debug session
            debug_session = {
                "attempts": 0,
                "fixed_issues": [],
                "current_errors": [],
                "status": "in_progress"
            }
            
            while True:
                if debug_session["attempts"] >= self.max_retries:
                    if not self.interactive:
                        debug_session["status"] = "max_retries_reached"
                        break
                    
                    user_input = input("\nReached maximum retry attempts. Would you like to continue debugging? (y/n): ")
                    if user_input.lower() != 'y':
                        debug_session["status"] = "user_stopped"
                        break
                    
                    debug_session["attempts"] = 0
                    logger.info("Resetting debug attempts counter...")
                
                debug_session["attempts"] += 1
                logger.info(f"Debug attempt {debug_session['attempts']} of {self.max_retries}")
                
                # Attempt to build Docker configuration
                logger.info(f"Starting build attempt {debug_session['attempts']}...")
                build_result = self._attempt_build(workspace_path)
                
                # Log build result
                if build_result["success"]:
                    logger.info(f"Build attempt {debug_session['attempts']} succeeded")
                else:
                    logger.warning(f"Build attempt {debug_session['attempts']} failed with errors: {json.dumps(build_result['errors'], indent=2)}")

                
                if build_result["success"]:
                    debug_session["status"] = "success"
                    break
                
                # If build failed, analyze and fix errors
                debug_session["current_errors"] = build_result["errors"]
                fix_result = self._fix_configuration(workspace_path, build_result["errors"])
                
                if fix_result["fixed"]:
                    debug_session["fixed_issues"].append({
                        "attempt": debug_session["attempts"],
                        "errors": build_result["errors"],
                        "fixes": fix_result["changes"]
                    })
                else:
                    debug_session["status"] = "failed"
                    break
            
            # Prepare final report
            report = {
                "status": debug_session["status"],
                "attempts": debug_session["attempts"],
                "fixed_issues": debug_session["fixed_issues"],
                "remaining_errors": debug_session["current_errors"] if debug_session["status"] == "failed" else []
            }
            
            return json.dumps(report, indent=2)
            
        except Exception as e:
            logger.error(f"Error in WorkspaceDebugger: {str(e)}")
            return json.dumps({
                "status": "error",
                "error": str(e)
            }, indent=2)
    
    def _attempt_build(self, workspace_path: str) -> Dict:
        """Attempts to build Docker configuration and returns build results."""
        try:
            # Log the build attempt
            logger.info("Starting docker-compose build...")
            logger.info(f"Running command: docker-compose build --no-cache in {workspace_path}")
            
            # First, try docker-compose build with detailed output
            compose_result = subprocess.run(
                ['docker-compose', 'build', '--no-cache', '--progress=plain'],
                cwd=workspace_path,
                capture_output=True,
                text=True,
                env={**os.environ, 'DOCKER_BUILDKIT': '1', 'COMPOSE_DOCKER_CLI_BUILD': '1'}
            )
            
            # Combine stdout and stderr for comprehensive error analysis
            full_output = compose_result.stdout + "\n" + compose_result.stderr
            
            # Log the command output
            logger.info("Build command output:")
            logger.info(full_output)
            
            if compose_result.returncode == 0:
                # Even with success return code, check for warning patterns
                if any(pattern in full_output.lower() for pattern in ['warning:', 'error:', 'failed']):
                    return {
                        "success": False,
                        "errors": [{
                            "type": "docker_compose_build_warning",
                            "message": full_output,
                            "exit_code": compose_result.returncode
                        }]
                    }
                return {"success": True, "errors": []}
            
            # Parse the error output to identify specific issues
            error_lines = full_output.split('\n')
            parsed_errors = []
            current_error = ""
            
            for line in error_lines:
                if any(pattern in line.lower() for pattern in ['error:', 'failed:', 'fatal:', 'exception']):
                    if current_error:
                        parsed_errors.append(current_error)
                    current_error = line
                elif line.strip() and current_error:
                    current_error += "\n" + line
            
            if current_error:
                parsed_errors.append(current_error)
            
            return {
                "success": False,
                "errors": [{
                    "type": "docker_compose_build_error",
                    "message": error if error else full_output,
                    "exit_code": compose_result.returncode,
                    "full_output": full_output
                } for error in (parsed_errors if parsed_errors else [full_output])]
            }
        except Exception as e:
            return {
                "success": False,
                "errors": [{
                    "type": "build_error",
                    "message": str(e)
                }]
            }
    
    def _fix_configuration(self, workspace_path: str, errors: List[Dict]) -> Dict:
        """Analyzes errors and attempts to fix Docker configuration files."""
        try:
            # Read current configuration files
            dockerfile_path = os.path.join(workspace_path, 'Dockerfile')
            compose_path = os.path.join(workspace_path, 'docker-compose.yml')
            
            dockerfile_content = ""
            compose_content = ""
            
            if os.path.exists(dockerfile_path):
                with open(dockerfile_path, 'r') as f:
                    dockerfile_content = f.read()
            
            if os.path.exists(compose_path):
                with open(compose_path, 'r') as f:
                    compose_content = f.read()
            
            # Prepare prompt for LLM with structured format
            fix_prompt = f"""You are a Docker configuration debugging expert. Analyze the provided errors and configuration files, then suggest fixes.

Context:
- Dockerfile content: {dockerfile_content if dockerfile_content else "Not present"}
- docker-compose.yml content: {compose_content if compose_content else "Not present"}
- Errors encountered: {json.dumps(errors, indent=2)}

Instructions:
1. Analyze the errors and identify their root causes
2. Determine which files need modifications
3. Provide specific fixes while maintaining the original configuration structure
4. Ensure all changes are compatible with Docker best practices

IMPORTANT: You MUST respond with ONLY a valid JSON object in the following format:

{{
    "analysis": {{
        "error_type": "<error classification>",
        "root_cause": "<identified root cause>",
        "affected_files": ["<list of affected files>"]
    }},
    "fixes": {{
        "dockerfile": {{
            "needs_update": <boolean>,
            "content": "<complete updated content or null if no changes>"
        }},
        "docker_compose": {{
            "needs_update": <boolean>,
            "content": "<complete updated content or null if no changes>"
        }}
    }},
    "explanation": "<brief explanation of the fixes applied>",
    "recommendations": [
        "<list of additional recommendations for preventing similar issues>"
    ]
}}

NO additional text, explanations, or markdown formatting should be included."""
            
            # Get fix suggestions with retries
            max_retries = 3
            retry_delay = 2
            
            for attempt in range(max_retries):
                try:
                    response = self.llm.invoke(fix_prompt)
                    if not response or not response.content:
                        logger.warning(f"Empty response from LLM on attempt {attempt + 1}")
                        if attempt < max_retries - 1:
                            time.sleep(retry_delay * (attempt + 1))
                            continue
                        return {"fixed": False, "changes": "Empty response from LLM after retries"}
                    
                    # Clean and parse response
                    cleaned_response = response.content.strip()
                    cleaned_response = re.sub(r'```(?:json)?\s*', '', cleaned_response)
                    cleaned_response = cleaned_response.replace('```', '').strip()
                    
                    fixes = json.loads(cleaned_response)
                    changes_made = False
                    
                    # Apply fixes based on the structured response
                    if fixes["fixes"]["dockerfile"]["needs_update"] and fixes["fixes"]["dockerfile"]["content"]:
                        with open(dockerfile_path, 'w') as f:
                            f.write(fixes["fixes"]["dockerfile"]["content"])
                        changes_made = True
                    
                    if fixes["fixes"]["docker_compose"]["needs_update"] and fixes["fixes"]["docker_compose"]["content"]:
                        with open(compose_path, 'w') as f:
                            f.write(fixes["fixes"]["docker_compose"]["content"])
                        changes_made = True
                    
                    return {
                        "fixed": changes_made,
                        "changes": {
                            "analysis": fixes["analysis"],
                            "explanation": fixes["explanation"],
                            "recommendations": fixes["recommendations"]
                        }
                    }
                    
                except json.JSONDecodeError as je:
                    logger.error(f"Failed to parse JSON on attempt {attempt + 1}: {str(je)}")
                    if attempt < max_retries - 1:
                        time.sleep(retry_delay * (attempt + 1))
                        continue
                    return {"fixed": False, "changes": f"Invalid JSON response: {str(je)}"}
                
                except Exception as e:
                    logger.error(f"Error in LLM processing on attempt {attempt + 1}: {str(e)}")
                    if attempt < max_retries - 1:
                        time.sleep(retry_delay * (attempt + 1))
                        continue
                    return {"fixed": False, "changes": f"LLM processing error: {str(e)}"}
            
            return {"fixed": False, "changes": "Failed after all retries"}
                
        except Exception as e:
            logger.error(f"Error fixing configuration: {str(e)}")
            return {"fixed": False, "changes": str(e)}
    
    async def _arun(self, query: str) -> str:
        raise NotImplementedError("Async version not implemented")


class WorkspaceCreatorTool(BaseTool):
    name = "workspace_creator"
    description = "Creates a workspace with Dockerfile and docker-compose.yml based on development details"
    
    def _run(self, development_details: str) -> str:
        try:
            # Parse development details and clean up any potential trailing data
            details = json.loads(development_details.split('\n')[0].strip())
            
            # Use a fixed workspace name instead of random
            workspace_path = os.path.join('workspaces', 'current')
            
            # Clean up existing workspace if it exists
            if os.path.exists(workspace_path):
                shutil.rmtree(workspace_path)
            
            # Create workspace directory
            os.makedirs(workspace_path, exist_ok=True)
            
            # Create LLM instance for file generation
            llm = ChatGoogleGenerativeAI(
                model="gemini-2.0-pro-exp-02-05",
                google_api_key=GOOGLE_API_KEY,
                convert_system_message_to_human=True,
                temperature=0.1
            )
            
            # Generate Dockerfile.R with focus on tools and frameworks
            dockerfile_prompt = """Create a Dockerfile.R based on the following development details, with special focus on the tools and frameworks mentioned:
            {details}
            
            The Dockerfile.R should:
            1. Use rocker/r-ver:4.3.1 as the base image
            2. Install all necessary system dependencies for R packages
            3. Install the following R packages with specific versions:
               - pso (latest version)
               - xgboost (latest version)
               - any additional packages needed for the tools mentioned
            4. Set up proper working directory and file structure
            5. Follow Docker best practices and optimize the build
            
            Respond with ONLY the Dockerfile.R content, no explanations or markdown formatting."""
            
            dockerfile_response = llm.invoke(dockerfile_prompt.format(details=json.dumps(details, indent=2)))
            
            # Clean up Dockerfile content
            dockerfile_content = self._clean_content(dockerfile_response.content)
            
            # Generate docker-compose.yml with focus on tools and frameworks
            compose_prompt = """Create a docker-compose.yml file based on the following development details, ensuring all tools and frameworks are properly integrated:
            {details}
            
            The docker-compose.yml should:
            1. Define a service named 'r-environment' that uses Dockerfile.R
            2. Set up the following volume mounts:
               - ./src:/app/src (for source code)
               - ./data:/app/data (for data files)
               - ./results:/app/results (for output)
            3. Configure environment variables:
               - R_LIBS_USER=/usr/local/lib/R/site-library
            4. Set up networking with 'patternchrome_network'
            5. Set working directory to /app
            6. Add a command to run the main R script
            
            Respond with ONLY the docker-compose.yml content, no explanations or markdown formatting."""
            
            compose_response = llm.invoke(compose_prompt.format(details=json.dumps(details, indent=2)))
            
            # Clean up docker-compose.yml content
            compose_content = self._clean_content(compose_response.content)
            
            # Validate YAML content
            try:
                yaml.safe_load(compose_content)
            except yaml.YAMLError as e:
                raise ValueError(f"Invalid docker-compose.yml format: {str(e)}")
            
            # Write files
            with open(os.path.join(workspace_path, 'Dockerfile.R'), 'w') as f:
                f.write(dockerfile_content)
            
            with open(os.path.join(workspace_path, 'docker-compose.yml'), 'w') as f:
                f.write(compose_content)
            
            # Create necessary directories
            os.makedirs(os.path.join(workspace_path, 'src'), exist_ok=True)
            os.makedirs(os.path.join(workspace_path, 'data'), exist_ok=True)
            os.makedirs(os.path.join(workspace_path, 'results'), exist_ok=True)
            
            # Create a basic R script
            with open(os.path.join(workspace_path, 'src', 'main.R'), 'w') as f:
                f.write("""# Main R script for PatternChrome
library(pso)
library(xgboost)

# Your code will go here
print("PatternChrome environment initialized successfully!")
""")
            
            # Run workspace debugger after creation
            debugger = WorkspaceDebugger(interactive=False)
            debug_result = json.loads(debugger._run(workspace_path))
            
            return json.dumps({
                "status": "success",
                "workspace_path": workspace_path,
                "files_created": ["Dockerfile.R", "docker-compose.yml", "src/main.R"],
                "debug_result": debug_result
            }, indent=2)
            
        except Exception as e:
            logger.error(f"Error in WorkspaceCreatorTool: {str(e)}")
            return json.dumps({
                "status": "error",
                "error": str(e)
            }, indent=2)
    
    def _clean_content(self, content: str) -> str:
        """Clean up the generated content by removing markdown artifacts and unnecessary formatting."""
        # Remove any leading/trailing whitespace
        content = content.strip()
        
        # Remove markdown code block markers and language identifiers
        content = re.sub(r'```(?:dockerfile|yaml|yml)?\s*', '', content)
        content = re.sub(r'```\s*$', '', content)
        
        # Remove any remaining backticks
        content = content.replace('`', '')
        
        # Remove any yaml: or dockerfile: prefix that might appear
        content = re.sub(r'^(?:yaml|dockerfile):\s*\n', '', content)
        
        # Remove version declaration from docker-compose.yml
        content = re.sub(r'^version:\s*["\']?[0-9\.]+["\']?\s*\n', '', content)
        
        # Ensure proper line endings
        content = content.replace('\r\n', '\n')
        
        # Remove any empty lines at the start or end
        content = content.strip()
        
        return content

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
                model="gemini-2.0-pro-exp-02-05",
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
            cleaned_response = response.content.strip()
            
            # First try: direct parsing
            try:
                json_obj = json.loads(cleaned_response)
                return json.dumps(json_obj, indent=2)
            except json.JSONDecodeError:
                # Second try: remove markdown and code block markers
                try:
                    cleaned_response = re.sub(r'```(?:json)?\s*', '', cleaned_response)
                    cleaned_response = re.sub(r'`.*?`', '', cleaned_response)
                    cleaned_response = re.sub(r'[\n\r\t]+', ' ', cleaned_response)
                    json_obj = json.loads(cleaned_response)
                    return json.dumps(json_obj, indent=2)
                except json.JSONDecodeError:
                    # Third try: extract JSON using regex
                    json_pattern = r'\{(?:[^{}]|\{[^{}]*\}[^{}]*)*\}|\{[^}]*}'
                    matches = re.finditer(json_pattern, cleaned_response)
                    
                    for match in matches:
                        try:
                            json_obj = json.loads(match.group(0))
                            if all(key in json_obj for key in ['is_computational_science', 'has_development_info', 'paper_type']):
                                return json.dumps(json_obj, indent=2)
                        except json.JSONDecodeError:
                            continue
                    
                    raise ValueError("Could not extract valid JSON from response")
                
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

class WorkspaceSyntaxValidator(BaseTool):
    name = "workspace_syntax_validator"
    description = "Validates the syntax of Dockerfile and docker-compose.yml files without using LLM"
    
    def _clean_yaml_content(self, content: str) -> str:
        # Remove markdown artifacts and clean YAML content
        content = re.sub(r'```(?:yaml)?\s*', '', content)
        content = re.sub(r'`.*?`', '', content)
        return content.strip()
    
    def _run(self, workspace_path: str) -> str:
        try:
            # Clean and validate the workspace path
            workspace_path = str(workspace_path).strip()
            if not os.path.exists(workspace_path):
                raise FileNotFoundError(f"Workspace directory not found: {workspace_path}")
            
            validation_results = {
                "dockerfile": {"exists": False, "valid": False, "errors": []},
                "docker_compose": {"exists": False, "valid": False, "errors": []}
            }
            
            # Check Dockerfile
            dockerfile_path = os.path.join(workspace_path, 'Dockerfile')
            if os.path.exists(dockerfile_path):
                validation_results["dockerfile"]["exists"] = True
                try:
                    # Use docker CLI to validate Dockerfile syntax
                    result = subprocess.run(
                        ['docker', 'build', '--no-cache', '--quiet', '-f', dockerfile_path, '-t', 'syntax_check', '.'],
                        cwd=workspace_path,
                        capture_output=True,
                        text=True
                    )
                    if result.returncode == 0:
                        validation_results["dockerfile"]["valid"] = True
                    else:
                        validation_results["dockerfile"]["errors"].append(result.stderr)
                except Exception as e:
                    validation_results["dockerfile"]["errors"].append(str(e))
            
            # Check docker-compose.yml
            compose_path = os.path.join(workspace_path, 'docker-compose.yml')
            if os.path.exists(compose_path):
                validation_results["docker_compose"]["exists"] = True
                try:
                    # Read and clean YAML content
                    with open(compose_path, 'r') as f:
                        content = f.read()
                    
                    cleaned_content = self._clean_yaml_content(content)
                    yaml_data = yaml.safe_load(cleaned_content)
                    
                    # Check for deprecated version key
                    if 'version' in yaml_data:
                        validation_results["docker_compose"]["errors"].append("The 'version' key is deprecated in recent Docker Compose versions and should be removed")
                    
                    # Then use docker-compose CLI to validate configuration
                    result = subprocess.run(
                        ['docker-compose', 'config', '-q'],
                        cwd=workspace_path,
                        capture_output=True,
                        text=True
                    )
                    if result.returncode == 0:
                        validation_results["docker_compose"]["valid"] = True
                    else:
                        validation_results["docker_compose"]["errors"].append(result.stderr)
                except yaml.YAMLError as e:
                    validation_results["docker_compose"]["errors"].append(f"YAML syntax error: {str(e)}")
                except Exception as e:
                    validation_results["docker_compose"]["errors"].append(str(e))
            
            return json.dumps(validation_results, indent=2)
            
        except Exception as e:
            logger.error(f"Error in WorkspaceSyntaxValidator: {str(e)}")
            return json.dumps({
                "error": str(e),
                "dockerfile": {"exists": False, "valid": False, "errors": [str(e)]},
                "docker_compose": {"exists": False, "valid": False, "errors": [str(e)]}
            }, indent=2)
    
    async def _arun(self, query: str) -> str:
        raise NotImplementedError("Async version not implemented")

def create_agent(name: str):
    if name == "WorkspaceDebugger":
        # Use Ollama LLM for WorkspaceDebugger
        llm = Ollama(
            model="llama3.2",
            base_url="http://localhost:11434",
            temperature=0.1
        )
    else:
        # Use Gemini for other agents
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-pro-exp-02-05",
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
        Thought: I will create a workspace based on the development details.
        Action: workspace_creator
        Action Input: [DEVELOPMENT_DETAILS_JSON]
        
        Observation: [TOOL_RESPONSE]
        
        Thought: Workspace creation is complete. I will return the results.
        Final Answer: [TOOL_RESPONSE]
        """
    elif name == "WorkspaceSyntaxValidator":
        tools = [WorkspaceSyntaxValidator()]
        template = """You are a Workspace Syntax Validator specialized in checking Docker configuration files.
        Your task is to validate the syntax of Dockerfile and docker-compose.yml files in a workspace.
        
        Available tools:
        {tool_names}
        
        Tools and their descriptions:
        {tools}
        
        User Input: {input}
        {agent_scratchpad}
        
        Follow these steps exactly:
        1. Extract the workspace path from the input
        2. Use the workspace_syntax_validator tool with the workspace path
        3. Return the validation results immediately after receiving a response
        4. Do not make additional validation calls
        
        Use this format EXACTLY:
        Action: workspace_syntax_validator
        Action Input: [WORKSPACE_PATH]
        
        Observation: [TOOL_RESPONSE]
        
        Thought: Syntax validation is complete. I will return the results.
        Final Answer: [TOOL_RESPONSE]
        """
    elif name == "WorkspaceDebugger":
        tools = [WorkspaceDebugger(interactive=False)]
        template = """You are a Workspace Debugger Assistant specialized in fixing Docker configuration issues.
        Your task is to debug and fix Docker configuration files in a workspace.
        
        Available tools:
        {tool_names}
        
        Tools and their descriptions:
        {tools}
        
        User Input: {input}
        {agent_scratchpad}
        
        Follow these steps exactly:
        1. Extract the workspace path from the input
        2. Use the workspace_debugger tool with the workspace path
        3. Return the debugging results immediately after receiving a response
        4. Do not make additional debug calls
        
        Use this format EXACTLY:
        Action: workspace_debugger
        Action Input: [WORKSPACE_PATH]
        
        Observation: [TOOL_RESPONSE]
        
        Thought: Debugging is complete. I will return the results.
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
            
        # Process paper_response
        if not paper_response or 'output' not in paper_response:
            raise ValueError("Invalid or empty response from PaperReader")
            
        output = paper_response['output']
        logger.debug(f"Raw output from PaperReader: {output}")
        
        # Enhanced JSON extraction approach
        validation_result = None
        
        # Try multiple JSON extraction methods
        extraction_methods = [
            # Method 1: Direct JSON parsing
            lambda text: json.loads(text) if isinstance(text, str) else None,
            
            # Method 2: Extract JSON from markdown code blocks
            lambda text: json.loads(re.sub(r'```(?:json)?\s*(.+?)\s*```', r'\1', text, flags=re.DOTALL)) if '```' in text else None,
            
            # Method 3: Find the last complete JSON object
            lambda text: next((json.loads(match.group(0)) 
                for match in reversed(list(re.finditer(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', text))) 
                if all(key in json.loads(match.group(0)) 
                    for key in ['is_computational_science', 'has_development_info', 'paper_type'])), None),
            
            # Method 4: Use extract_json_from_text function as fallback
            lambda text: json.loads(extract_json_from_text(text))
        ]
        
        # Try each extraction method
        for extract_method in extraction_methods:
            try:
                result = extract_method(output)
                if result and isinstance(result, dict) and all(key in result for key in ['is_computational_science', 'has_development_info', 'paper_type']):
                    validation_result = result
                    logger.info("Successfully extracted and validated JSON response")
                    break
            except Exception as e:
                logger.debug(f"Extraction method failed: {str(e)}")
                continue
        
        if not validation_result:
            logger.error("Failed to extract valid JSON from response")
            print("\nError: Could not parse validation response")
            print(f"Raw output:\n{output}")
            return
        
        # Process the validation result
        if validation_result['is_computational_science'] and validation_result['has_development_info']:
            logger.info("Paper validated successfully. Creating workspace...")
            
            # Create workspace with development details
            workspace_creator = create_agent("WorkspaceCreator")
            workspace_response = workspace_creator.invoke({"input": json.dumps(validation_result['development_details'])})
            
            if workspace_response and 'output' in workspace_response:
                print("\nWorkspace creation results:")
                print(workspace_response['output'])
            else:
                print("\nError: Failed to create workspace")
        else:
            print("\nPaper validation results:")
            print(json.dumps(validation_result, indent=2))
            
        # Enhanced JSON extraction approach
        validation_result = None
        
        # First try: Extract the last complete JSON object from the output
        try:
            # Look for JSON objects in the response
            json_matches = list(re.finditer(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', output))
            if json_matches:
                # Get the last match as it's likely the final response
                last_json = json_matches[-1].group(0)
                potential_result = json.loads(last_json)
                # Verify it has the expected structure
                if all(key in potential_result for key in ['is_computational_science', 'has_development_info', 'paper_type']):
                    validation_result = potential_result
                    logger.info("Successfully parsed JSON from output using pattern matching")
        except Exception as e:
            logger.warning(f"Pattern matching JSON extraction failed: {str(e)}")
        
        # Second try: Use the extract_json_from_text function
        if not validation_result:
            try:
                extracted_json = extract_json_from_text(output)
                potential_result = json.loads(extracted_json)
                if all(key in potential_result for key in ['is_computational_science', 'has_development_info', 'paper_type']):
                    validation_result = potential_result
                    logger.info("Successfully parsed JSON using extraction method")
            except Exception as e:
                logger.warning(f"JSON extraction method failed: {str(e)}")
        
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
