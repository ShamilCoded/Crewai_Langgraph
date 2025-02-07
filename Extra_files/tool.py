from langchain_google_genai import GoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain_core.tools import tool
from langchain_community.tools.tavily_search import TavilySearchResults
import requests
from langchain_core.runnables import RunnableSequence
from astropy.coordinates import SkyCoord  # High-level coordinates

from crewai import Agent, Task, Crew
import pandas as pd
import requests
from langchain.tools import Tool
from langchain_core.tools import tool
# from langchain_community.utilities import WikipediaAPIWrapper

import os
from dotenv import load_dotenv
load_dotenv()

# Initialize tools
search = TavilySearchResults(tavily_api_key=os.getenv("TAVILY_API_KEY"))
# wikipedia = WikipediaAPIWrapper()

# Load planetary data
# load = Loader('data')
# planets = load('de421.bsp')

llm = GoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=os.getenv("GOOGLE_API_KEY"))

prompt_template = PromptTemplate(
    input_variables=["input"],
    template="""
You are a tool caller. You will call the appropriate tool based on the user's input. Here are the available tools:

1. **get_nasa_apod_tool**: This tool fetches the Astronomy astromony or space details from NASA. 
   - Call this tool when the user asks about NASA or Astronomy data.
   
2. **calculate_sky_coordinates**: This tool calculates celestial coordinates from RA (Right Ascension) and Dec (Declination).
   - Call this tool when the user provides Right Ascension (RA) and Declination (Dec) values, and you need to convert them to celestial coordinates.

3. **generate_orbital_plot**: This tool generates a 3D orbital plot for a given celestial body.
   - Call this tool when the user asks about the orbital position of a celestial body (e.g., "What is the orbit of Mars?").

Based on the user's input, select the appropriate tool and provide just the necessary arguments to it. User input: {input}
"""
)
# Fetch NASA API key from environment
nasa_api_key = os.getenv("NASA_API_KEY")

@tool
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

@tool
def calculate_sky_coordinates(ra: str, dec: str) -> dict:
    """Convert RA/Dec to celestial coordinates"""
    coord = SkyCoord(ra, dec, unit=(u.hourangle, u.deg))
    return {'ra_deg': coord.ra.deg, 'dec_deg': coord.dec.deg}

# @tool
# def generate_orbital_plot(body_name: str) -> go.Figure:
#     """Generate 3D orbital plot for celestial bodies"""
#     ts = load.timescale()
#     t = ts.now()
    
#     try:
#         body = planets[body_name]
#         astrometric = planets['earth'].at(t).observe(body)
#         ra, dec, distance = astrometric.radec()
        
#         fig = go.Figure(data=[go.Scatter3d(
#             x=[0, distance.au],
#             y=[0, 0],
#             z=[0, 0],
#             mode='lines+markers',
#             marker=dict(size=5),
#             line=dict(width=2)
#         )])
        
#         fig.update_layout(
#             scene=dict(
#                 xaxis_title='X (AU)',
#                 yaxis_title='Y (AU)',
#                 zaxis_title='Z (AU)'
#             ),
#             title=f"{body_name} Orbital Position"
#         )
#         return fig
#     except KeyError:
#         return None


chain = RunnableSequence(
    prompt_template, 
    llm,
    search, 
    get_nasa_apod_tool,
    )

output = chain.invoke("list the upcoming events in nasa but not the picture of the day")
print("output",output)
    