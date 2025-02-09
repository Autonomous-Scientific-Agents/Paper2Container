import os
import json
import logging
import subprocess
from typing import Dict, List, Optional, Tuple
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.tools import BaseTool

logger = logging.getLogger(__name__)

class WorkspaceDebugger(BaseTool):
    name = "workspace_debugger"
    description = "Debugs and fixes Docker configuration files in a workspace"
    max_retries = 10
    interactive = True  # Add default class attribute
    
    def __init__(self, google_api_key: str, interactive: bool = True):
        super().__init__()
        self.interactive = interactive
        self.google_api_key = google_api_key
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-pro-exp-02-05",
            google_api_key=google_api_key,
            convert_system_message_to_human=True,
            temperature=0.1
        )
    
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
            
            # Prepare prompt for LLM
            fix_prompt = f"""Analyze these Docker configuration errors and suggest fixes:
            
Errors:
{json.dumps(errors, indent=2)}

Current Dockerfile:
{dockerfile_content}

Current docker-compose.yml:
{compose_content}

Provide ONLY a JSON response with fixes in this format:
{{
    "dockerfile_changes": {{
        "content": "updated content" or null if no changes
    }},
    "compose_changes": {{
        "content": "updated content" or null if no changes
    }},
    "explanation": "brief explanation of fixes"
}}"""
            
            # Get fix suggestions from LLM with error handling
            try:
                response = self.llm.invoke(fix_prompt)
                if not response or not response.content:
                    logger.warning("Empty response from LLM")
                    return {"fixed": False, "changes": "Empty response from LLM"}
                
                # Clean and parse response
                cleaned_response = response.content.strip()
                fixes = json.loads(cleaned_response)
                
                changes_made = False
                
                # Apply fixes to Dockerfile
                if fixes.get("dockerfile_changes", {}).get("content"):
                    with open(dockerfile_path, 'w') as f:
                        f.write(fixes["dockerfile_changes"]["content"])
                    changes_made = True
                
                # Apply fixes to docker-compose.yml
                if fixes.get("compose_changes", {}).get("content"):
                    with open(compose_path, 'w') as f:
                        f.write(fixes["compose_changes"]["content"])
                    changes_made = True
                
                return {
                    "fixed": changes_made,
                    "changes": fixes.get("explanation", "No changes made")
                }
            except json.JSONDecodeError as je:
                logger.error(f"Failed to parse LLM response: {str(je)}")
                return {"fixed": False, "changes": f"Invalid JSON response: {str(je)}"}
            except Exception as e:
                logger.error(f"Error in LLM processing: {str(e)}")
                return {"fixed": False, "changes": f"LLM processing error: {str(e)}"}
                
        except Exception as e:
            logger.error(f"Error fixing configuration: {str(e)}")
            return {"fixed": False, "changes": str(e)}
    
    async def _arun(self, query: str) -> str:
        raise NotImplementedError("Async version not implemented")