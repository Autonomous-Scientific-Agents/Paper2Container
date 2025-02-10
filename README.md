# Paper 2 Container

A tool that analyzes academic papers and automatically creates containerized environments based on their computational requirements. This project helps researchers reproduce computational experiments by extracting development information from scientific papers and generating appropriate container configurations.

## Features

- Analyzes PDF papers to identify computational requirements
- Validates if the paper contains sufficient development information
- Extracts tools, frameworks, and implementation details
- Generates containerized environments based on paper specifications
- Supports both Python and R environments
- Utilizes dual LLM support (Gemini Pro and Ollama's LLaMA 3.2) for enhanced analysis and generation

## Prerequisites

- Python 3.8 or higher
- pip package manager
- Google API key for Gemini Pro model (set in environment variables)
- Ollama with LLaMA 3.2 model installed locally

## Setup

1. Clone the repository
2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

3. Set up your environment variables:
   - Create a `.env` file in the project root
   - Add your Google API key: `GOOGLE_API_KEY=your_key_here`

## Usage

Run the tool by providing a PDF file path:

```bash
python main.py <path_to_paper.pdf>
```

The tool will:
1. Analyze the paper for computational content
2. Extract development information
3. Create a workspace with appropriate container configurations
4. Generate necessary environment files

## Output

The tool generates:
- Validation results of the paper analysis
- Docker and docker-compose configurations
- Environment setup files
- Implementation recommendations

## Notes

- Ensure the PDF file is readable and contains computational research content
- The tool works best with papers that include clear implementation details
- Generated container configurations may need manual adjustments based on specific requirements
