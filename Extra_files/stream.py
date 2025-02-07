import os
import streamlit as st
from crewai import Agent, Task, Crew
from tavily import TavilyClient
from astropy.coordinates import SkyCoord
from astropy import units as u
from skyfield.api import Loader
import plotly.graph_objects as go
import pandas as pd
import requests
from langchain.tools import Tool
from langchain_community.utilities import WikipediaAPIWrapper
from dotenv import load_dotenv

load_dotenv()

# Initialize tools
tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
wikipedia = WikipediaAPIWrapper()

# Load planetary data
load = Loader('data')
planets = load('de421.bsp')

# Define custom tools
def get_nasa_apod() -> dict:
    """Fetch NASA Astronomy Picture of the Day"""
    url = f"https://api.nasa.gov/planetary/apod?api_key={os.getenv('NASA_API_KEY')}"
    response = requests.get(url)
    return response.json() if response.status_code == 200 else {"error": "Failed to fetch APOD"}

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

# Wrap tools with Tool.from_function for CrewAI compatibility
nasa_apod_tool = Tool.from_function(
    func=get_nasa_apod,
    name="nasa_apod",
    description="Fetch NASA Astronomy Picture of the Day"
)

neo_data_tool = Tool.from_function(
    func=get_neo_data,
    name="neo_data",
    description="Fetch Near-Earth Objects data from NASA"
)

sky_coord_tool = Tool.from_function(
    func=calculate_sky_coordinates,
    name="sky_coordinates",
    description="Convert RA/Dec to celestial coordinates"
)

orbital_plot_tool = Tool.from_function(
    func=generate_orbital_plot,
    name="orbital_plot",
    description="Generate 3D orbital plot for celestial bodies"
)

# Wrap Tavily search function with Tool.from_function
tavily_search_tool = Tool.from_function(
    func=tavily.search,
    name="tavily_search",
    description="Search for astronomical data using Tavily"
)

# Wrap Wikipedia search function with Tool.from_function
wikipedia_tool = Tool.from_function(
    func=wikipedia.run,
    name="wikipedia_search",
    description="Search for astronomy information on Wikipedia"
)

# Define agents with validated tools
# Define agents with validated tools
research_agent = Agent(
    role="Senior Astronomy Researcher",
    goal="Provide accurate real-time astronomical data",
    backstory="Expert in space phenomena tracking with access to latest observational data",
    tools=[nasa_apod_tool, neo_data_tool, tavily_search_tool],
    verbose=True
)

visualization_agent = Agent(
    role="3D Visualization Specialist",
    goal="Create interactive astronomical visualizations",
    backstory="Skilled in converting raw astronomical data into engaging 3D representations",
    tools=[sky_coord_tool, orbital_plot_tool],
    verbose=True
)

# Define agents with validated tools
educational_agent = Agent(
    role="Astronomy Tutor",
    goal="Explain complex concepts in simple terms",
    backstory="Experienced educator with PhD in Astrophysics",
    tools=[wikipedia_tool],
    verbose=True
)

# Streamlit UI
st.title("Astronomy AI Assistant")

tab1, tab2, tab3 = st.tabs(["Research", "Visualization", "Education"])

with tab1:
    st.header("Real-time Astronomy Research")
    research_query = st.text_input("Enter research topic:")
    
    if st.button("Research"):
        research_task = Task(
            description=f"Investigate {research_query} and provide detailed analysis",
            agent=research_agent,
            expected_output="Formal report with latest data and sources"
        )
        
        crew = Crew(
            agents=[research_agent],
            tasks=[research_task],
            verbose=True
        )
        
        result = crew.kickoff()
        st.subheader("Research Report")
        st.write(result)

with tab2:
    st.header("3D Astronomical Visualizations")
    visualization_option = st.selectbox("Select visualization type:", 
                                      ["Celestial Coordinates", "Orbital Mechanics"])
    
    if visualization_option == "Celestial Coordinates":
        ra = st.text_input("Right Ascension (HH:MM:SS):", "05:55:10")
        dec = st.text_input("Declination (DD:MM:SS):", "+07:24:25")
        
        if st.button("Plot Coordinates"):
            coord_task = Task(
                description=f"Convert RA/DEC {ra}/{dec} to 3D coordinates",
                agent=visualization_agent,
                expected_output="3D plot and coordinate data"
            )
            
            crew = Crew(
                agents=[visualization_agent],
                tasks=[coord_task],
                verbose=True
            )
            
            result = crew.kickoff(inputs={'ra': ra, 'dec': dec})
            st.plotly_chart(generate_orbital_plot('mars'))
            
    else:
        planet = st.selectbox("Select Planet:", ["mercury", "venus", "mars", "jupiter"])
        if st.button("Generate Orbital Plot"):
            fig = generate_orbital_plot(planet)
            if fig:
                st.plotly_chart(fig)
            else:
                st.error("Invalid celestial body selected")

with tab3:
    st.header("Astronomy Education")
    education_query = st.text_input("Ask an astronomy question:")
    
    if st.button("Explain"):
        education_task = Task(
            description=f"Explain {education_query} to a beginner student",
            agent=educational_agent,
            expected_output="Clear 3-paragraph explanation with examples"
        )
        
        crew = Crew(
            agents=[educational_agent],
            tasks=[education_task],
            verbose=True
        )
        
        result = crew.kickoff()
        st.subheader("Explanation")
        st.write(result)

st.sidebar.info("Note: Requires NASA API and Tavily API keys in environment variables")