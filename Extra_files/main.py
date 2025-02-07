import os
from crewai import Agent, Task, Crew
import requests
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

# Importing crewAI tools
from crewai_tools import (
    DirectoryReadTool,
    FileReadTool,
    WebsiteSearchTool
)

# Load environment variables
load_dotenv()

# Initialize the LLM for advanced space-related insights
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash", 
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

# Fetch NASA API key from environment
nasa_api_key = os.getenv("NASA_API_KEY")

def get_nasa_apod_tool() -> str:
    """Fetch the Astronomy Picture of the Day (APOD) from NASA."""
    print("Fetching the latest Astronomy Picture of the Day from NASA APOD API...")
    
    url = f"https://api.nasa.gov/planetary/apod?api_key={nasa_api_key}"
    response = requests.get(url)
    
    if response.status_code == 200:
        data = response.json()
        return (f"Title: {data['title']}\n"
                f"Date: {data['date']}\n"
                f"Explanation: {data['explanation']}\n"
                f"Image URL: {data['url']}")
    else:
        return f"Error: Unable to fetch data, status code {response.status_code}"

# Instantiate tools for reading and web searches
docs_tool = DirectoryReadTool(directory='./space-articles')
file_tool = FileReadTool()
web_rag_tool = WebsiteSearchTool()

# Create agents
space_researcher = Agent(
    role='Space Research Analyst',
    goal='Provide up-to-date research on space phenomena and astronomy news',
    backstory='An expert in analyzing space data and trends.',
    tools=[web_rag_tool],
    llm=llm,
    verbose=True
)

space_writer = Agent(
    role='Space Content Writer',
    goal='Craft engaging articles about space exploration and astronomy',
    backstory='A writer passionate about making space science accessible to everyone.',
    tools=[docs_tool, file_tool],
    llm=llm,
    verbose=True
)

nasa_tool_agent = Agent(
    role='NASA Data Specialist',
    goal='Provide data and insights using NASA APIs',
    backstory='An expert in space data analysis.',
    tools=[get_nasa_apod_tool],
    llm=llm,
    verbose=True
)

# Define tasks
research_space = Task(
    description='Research the latest developments in space exploration and provide a detailed summary.',
    expected_output='A summary of the top 3 recent events in astronomy and space science with unique insights.',
    agent=space_researcher
)

write_space_article = Task(
    description='Write an engaging article about recent space discoveries, based on the researcher’s findings. Incorporate data from the latest space articles.',
    expected_output='A 4-paragraph article formatted in markdown, using simple language to make complex space concepts engaging.',
    agent=space_writer,
    output_file='space-articles/new_space_post.md'  # The final space article will be saved here
)

nasa_data_task = Task(
    description='Fetch the Astronomy Picture of the Day (APOD) data from NASA API and provide a summary.',
    expected_output='A detailed summary of the APOD including title, date, explanation, and image URL.',
    agent=nasa_tool_agent
)

# Assemble a crew with planning enabled
crew = Crew(
    agents=[space_researcher, space_writer, nasa_tool_agent],
    tasks=[research_space, write_space_article, nasa_data_task],
    verbose=True,
    planning=True
)

# Execute tasks
crew.Kickoff()
