from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, START
from langgraph.graph import MessagesState
from langgraph.prebuilt import ToolNode
from langgraph.prebuilt import tools_condition
from crewai import Agent, Task, Crew
from astropy.coordinates import SkyCoord
from astropy import units as u
from skyfield.api import Loader
import plotly.graph_objects as go
import pandas as pd
from langchain_community.utilities import WikipediaAPIWrapper
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.state import CompiledStateGraph # type
from langchain_core.messages import  HumanMessage, SystemMessage
from dotenv import load_dotenv
from pydantic import BaseModel
import requests
import os
from fastapi import FastAPI, HTTPException

load_dotenv()

app = FastAPI()


llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash", 
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

search = TavilySearchResults(tavily_api_key=os.getenv("TAVILY_API_KEY"))

# Fetch NASA API key from environment
# Define custom tools
def get_nasa_apod() -> dict:
    """Fetch NASA Astronomy Picture of the Day"""
    url = f"https://api.nasa.gov/planetary/apod?api_key={os.getenv('NASA_API_KEY')}"
    response = requests.get(url)
    return response.json() if response.status_code == 200 else {"error": "Failed to fetch APOD"}


# Load planetary data
load = Loader('data')
planets = load('de421.bsp')

def get_neo_data() -> dict:
    """Fetch Near-Earth Objects data from NASA"""
    url = f"https://api.nasa.gov/neo/rest/v1/feed?api_key={os.getenv('NASA_API_KEY')}"
    response = requests.get(url)
    return response.json() if response.status_code == 200 else {"error": "Failed to fetch NEO data"}

def calculate_sky_coordinates(ra: str, dec: str) -> dict:
    """Convert RA/Dec to celestial coordinates"""
    coord = SkyCoord(ra, dec, unit=(u.hourangle, u.deg))
    return {'ra_deg': coord.ra.deg, 'dec_deg': coord.dec.deg}

def generate_orbital_plot(body_name: str) -> go.Figure:
    """Generate 3D orbital plot for celestial bodies"""
    ts = load.timescale()
    t = ts.now()
    
    try:
        body = planets[body_name]
        astrometric = planets['earth'].at(t).observe(body)
        ra, dec, distance = astrometric.radec()
        
        fig = go.Figure(data=[go.Scatter3d(
            x=[0, distance.au],
            y=[0, 0],
            z=[0, 0],
            mode='lines+markers',
            marker=dict(size=5),
            line=dict(width=2)
        )])
        
        fig.update_layout(
            scene=dict(
                xaxis_title='X (AU)',
                yaxis_title='Y (AU)',
                zaxis_title='Z (AU)'
            ),
            title=f"{body_name} Orbital Position"
        )
        return fig
    except KeyError:
        return None


tools = [search, get_nasa_apod, get_neo_data, calculate_sky_coordinates, generate_orbital_plot]


llm_with_tools = llm.bind_tools(tools)

# Define agents with validated tools
research_agent = Agent(
    role="Senior Astronomy Researcher",
    goal="Provide accurate real-time astronomical data",
    backstory="Expert in space phenomena tracking with access to latest observational data",
    tools=[get_nasa_apod, get_neo_data],
    llm=llm,
    verbose=True
)

visualization_agent = Agent(
    role="3D Visualization Specialist",
    goal="Create interactive astronomical visualizations",
    backstory="Skilled in converting raw astronomical data into engaging 3D representations",
    tools=[calculate_sky_coordinates, generate_orbital_plot],
    llm=llm,
    verbose=True
)

# Define agents with validated tools
educational_agent = Agent(
    role="Astronomy Tutor",
    goal="Explain complex concepts in simple terms",
    backstory="Experienced educator with PhD in Astrophysics",
    tools=[search],
    llm=llm,
    verbose=True
)


# Define tasks
research_task = Task(
    description='Gather real-time data on near-Earth objects (NEOs) and recent astronomical events using validated tools.',
    expected_output='A comprehensive report detailing recent astronomical observations and notable near-Earth objects.',
    agent=research_agent
)

visualization_task = Task(
    description='Generate interactive 3D orbital plots for recently detected near-Earth objects based on research data.',
    expected_output='A set of engaging 3D visualizations showcasing orbital trajectories and positions of NEOs.',
    agent=visualization_agent,
    output_file='visualizations/neo_orbital_plot.zip'  # Save the visualization files here
)

education_task = Task(
    description='Explain the significance of newly detected near-Earth objects in simple, accessible terms for educational purposes.',
    expected_output='A clear and engaging educational document formatted in markdown, making complex concepts easy to understand.',
    agent=educational_agent,
    output_file='education/neo_summary.md'  # Save the educational content here
)

# Assemble a crew with planning enabled
crew = Crew(
    agents=[research_agent, visualization_agent, educational_agent],
    tasks=[research_task, visualization_task, education_task],
    verbose=True,
    planning=True  # Enable planning feature
)

result = crew.kickoff()

# System message
sys_msg = SystemMessage(content='''This is designed to provide real-time astronomical data, visualization, and educational content. Below are the key functions and tools integrated into the system and their specific purposes:

1. **Tavily Search (`search`) Integration:**  
   - This tool is responsible for providing search results from Tavily, enabling users to access up-to-date and relevant space-related information.

2. **NASA APOD Tool (`get_nasa_apod_tool`)**  
   - Purpose: Fetches the Astronomy Picture of the Day (APOD) from NASA's APOD API.  
   - Usage: Provides users with the title, date, explanation, and image URL of the latest astronomy image shared by NASA.

3. **Wikipedia API Wrapper (`wikipedia`)**  
   - Purpose: Retrieves detailed and authoritative information on astronomical concepts for educational purposes.  
   - Usage: Answers conceptual queries and provides additional context on celestial events and phenomena.

4. **NEO Data Fetcher (`get_neo_data`)**  
   - Purpose: Fetches real-time data on Near-Earth Objects (NEOs) using NASA's NEO API.  
   - Usage: Provides valuable information on potentially hazardous objects near Earth.

5. **Sky Coordinates Calculator (`calculate_sky_coordinates`)**  
   - Purpose: Converts Right Ascension (RA) and Declination (Dec) values to celestial coordinates in degrees.  
   - Usage: Useful for astronomers and researchers who need precise coordinate conversions.

6. **Orbital Plot Generator (`generate_orbital_plot`)**  
   - Purpose: Generates interactive 3D orbital plots of celestial bodies using Plotly.  
   - Usage: Helps visualize the position and orbital trajectories of planets and NEOs.

7. **State Graph with Memory (`react_graph_memory`)**  
   - Purpose: Maintains conversational state, allowing the system to remember previous messages and tool interactions.  
   - Usage: Ensures a seamless and dynamic conversation flow.

### Workflow:
- When a user asks a query, the system identifies the appropriate tools to answer it.
- If astronomical data or visualizations are required, it calls the corresponding NASA API or orbital plotting function.
- For educational content, Wikipedia and other research tools are used.
- The system remembers the context of the conversation using the `MemorySaver`.

### Guidelines:
- The system provides scientifically accurate information and interactive visualizations.  
- It ensures educational content is accessible and easy to understand.  
- For visualization requests, orbital plots should be rendered interactively.  
- Tool usage must align with user queries, invoking only the necessary functions.
''')

# Node
def assistant(state: MessagesState) -> MessagesState:
    return {"messages": [llm.invoke([sys_msg] + state["messages"][-10:])]}

# Build graph
builder: StateGraph = StateGraph(MessagesState)

# Define nodes: these do the work
builder.add_node("assistant", assistant)
builder.add_node("tools", ToolNode(tools))

# Define edges: these determine how the control flow moves
builder.add_edge(START, "assistant")
builder.add_conditional_edges(
    "assistant",
    # If the latest message (result) from assistant is a tool call -> tools_condition routes to tools
    # If the latest message (result) from assistant is a not a tool call -> tools_condition routes to END
    tools_condition,
)
builder.add_edge("tools", "assistant")
memory: MemorySaver = MemorySaver()
react_graph_memory: CompiledStateGraph = builder.compile(checkpointer=memory)

# Specify a thread
config1 = {"configurable": {"thread_id": "1"}}


messages = [HumanMessage(content="tell me about earth near objects?")]
messages = react_graph_memory.invoke({"messages": messages}, config1)
for m in messages['messages']:
    m.pretty_print()

# class UserInput(BaseModel):
#     input_text: str 

# # API endpoint
# @app.post("/generateanswer")
# async def generate_answer(user_input: UserInput):
#     try:
#         messages = [HumanMessage(content=user_input.input_text)]
#         response = react_graph_memory.invoke({"messages": messages}, config={"configurable": {"thread_id": "1"}})

#         # Extract the response from the graph output
#         if response and "messages" in response:
#             # Extract the last message (assistant's response)
#             assistant_response = response["messages"][-1].content
#             return {"response": assistant_response}
#         else:
#             return {"response": "No response generated."}

#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))
