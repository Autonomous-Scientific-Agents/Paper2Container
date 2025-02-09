import os
import logging
import argparse
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

# Create a simple tool
class HelloTool(BaseTool):
    name = "hello_tool"
    description = "Returns a simple hello message"
    
    def _run(self, query: str) -> str:
        return "Hello! I'm a response from a tool!"
    
    async def _arun(self, query: str) -> str:
        raise NotImplementedError("async version not implemented yet")

def create_agent(name: str):
    # Create LLM model
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-pro-exp-02-05", google_api_key=GOOGLE_API_KEY)
    
    # Define tools
    tools = [HelloTool()]
    
    # Prompt template
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
        max_iterations=1  # Limit to one tool use
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

    if not validate_pdf_file(args.pdf_file):
        logger.error(f"Error: Invalid PDF file or file not found: {args.pdf_file}")
        return

    logger.info(f"Processing PDF file: {args.pdf_file}")
    
    # Create agents and process the PDF
    try:
        agent1 = create_agent("Agent1")
        agent2 = create_agent("Agent2")
        
        logger.info("Agents created")
        
        # Send PDF processing request to agents
        logger.info("Sending request to Agent1...")
        response1 = agent1.invoke({"input": f"Process this PDF file: {args.pdf_file}"})
        print(f"Agent1 response: {response1['output']}")
        
        logger.info("Sending request to Agent2...")
        response2 = agent2.invoke({"input": f"Process this PDF file: {args.pdf_file}"})
        print(f"Agent2 response: {response2['output']}")
        
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()
