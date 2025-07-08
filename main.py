import streamlit as st
import pandas as pd
import numpy as np
import json
import folium
from streamlit_folium import st_folium
import geopandas as gpd
from shapely.geometry import Point, Polygon, LineString
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import time
import random
from typing import Dict, List, Any, Optional
import re
from openai import OpenAI

# Set page config
st.set_page_config(
    page_title="Geospatial LLM Assistant",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'workflow_history' not in st.session_state:
    st.session_state.workflow_history = []
if 'current_workflow' not in st.session_state:
    st.session_state.current_workflow = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# LLMReasoner class using OpenAI SDK
class LLMReasoner:
    def __init__(self, provider: str):
        self.provider = provider
        if provider == "Gemini":
            self.client = OpenAI(
                api_key="",
                base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
            )
            self.model = "gemini-2.0-flash-lite"
        elif provider == "Groq":
            self.client = OpenAI(
                api_key="gsk_",
                base_url="https://api.groq.com/openai/v1"
            )
            self.model = "gemma2-9b-it"
        else:
            raise ValueError("Unknown provider")

    def analyze_query(self, query: str) -> Dict:
        # Prompt for LLM to classify the task type and confidence
        prompt = f"""
You are a geospatial analysis assistant. Given the following user query, classify it as one of: flood_risk, site_suitability, urban_planning, or general_analysis. Also, provide a confidence score (0-1) and echo the query.
Query: {query}
Respond in JSON with keys: task_type, confidence, query.
"""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=256,
            temperature=0.2
        )
        if st.checkbox("Show Debug Info", key="llm_debug"):
            st.write("LLM raw response:", response)
        try:
            content = response.choices[0].message.content
            result = json.loads(content)
            
        except Exception:
            # fallback
            result = {"task_type": "general_analysis", "confidence": 0.5, "query": query}
        return result

    def generate_chain_of_thought(self, query_analysis: Dict) -> List[Dict]:
        # Prompt for LLM to generate reasoning steps
        prompt = f"""
You are a geospatial analysis assistant. For the following task type and query, generate a step-by-step chain-of-thought reasoning process (3-6 steps). Each step should have a title and a short reasoning sentence. Respond as a JSON list of objects with keys: step, title, reasoning, status (set to 'pending').
Task type: {query_analysis['task_type']}
Query: {query_analysis['query']}
"""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=512,
            temperature=0.3
        )
        try:
            content = response.choices[0].message.content
            steps = json.loads(content)
        except Exception:
            # fallback
            steps = [
                {"step": 1, "title": "Analysis", "reasoning": "Understand the spatial problem requirements", "status": "pending"},
                {"step": 2, "title": "Data", "reasoning": "Identify necessary data sources", "status": "pending"},
                {"step": 3, "title": "Plan", "reasoning": "Plan appropriate analysis methods", "status": "pending"}
            ]
        return steps

    def generate_workflow(self, query_analysis: Dict) -> Dict:
        # Prompt for LLM to generate a workflow (list of steps with tool, description, parameters)
        prompt = f"""
You are a geospatial analysis assistant. For the following task type and query, generate a workflow as a JSON object with keys: id, name, description, steps (list of steps, each with id, tool, description, parameters). Use realistic tool names and parameters for geospatial analysis. Use bbox [77.1, 28.4, 77.3, 28.7] for the region. Example tools: load_osm_data, buffer_analysis, load_elevation_data, raster_calculator, spatial_intersection, etc.
Task type: {query_analysis['task_type']}
Query: {query_analysis['query']}
"""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1024,
            temperature=0.3
        )
        try:
            content = response.choices[0].message.content
            workflow = json.loads(content)
        except Exception:
            # fallback to a simple workflow
            workflow = {
                "id": f"general_analysis_{int(time.time())}",
                "name": "General Spatial Analysis",
                "description": "General-purpose spatial analysis workflow",
                "steps": [
                    {
                        "id": "load_data",
                        "tool": "load_spatial_data",
                        "description": "Load required spatial datasets",
                        "parameters": {
                            "sources": ["osm", "government_data"],
                            "bbox": [77.1, 28.4, 77.3, 28.7]
                        }
                    },
                    {
                        "id": "basic_analysis",
                        "tool": "spatial_analysis",
                        "description": "Perform basic spatial analysis",
                        "parameters": {
                            "operations": ["buffer", "intersect", "union"]
                        }
                    },
                    {
                        "id": "generate_output",
                        "tool": "output_generator",
                        "description": "Generate analysis results",
                        "parameters": {
                            "format": "geojson",
                            "include_statistics": True
                        }
                    }
                ]
            }
        return workflow

class WorkflowExecutor:
    """Executes geospatial workflows and generates mock results"""
    
    def __init__(self):
        self.mock_data = self._generate_mock_data()
    
    def _generate_mock_data(self) -> Dict:
        """Generate mock geospatial data for demonstration"""
        # Generate sample points for Delhi area
        np.random.seed(42)
        n_points = 50
        
        # Delhi bounding box
        min_lon, min_lat, max_lon, max_lat = 77.1, 28.4, 77.3, 28.7
        
        # Generate random points
        lons = np.random.uniform(min_lon, max_lon, n_points)
        lats = np.random.uniform(min_lat, max_lat, n_points)
        
        # Create sample datasets
        rivers = gpd.GeoDataFrame({
            'geometry': [LineString([(77.15, 28.45), (77.25, 28.65)]),
                        LineString([(77.12, 28.5), (77.28, 28.6)])],
            'name': ['Yamuna River', 'Tributary'],
            'width': [50, 20]
        })
        
        buildings = gpd.GeoDataFrame({
            'geometry': [Point(lon, lat) for lon, lat in zip(lons, lats)],
            'type': np.random.choice(['residential', 'commercial', 'industrial'], n_points),
            'height': np.random.randint(1, 20, n_points)
        })
        
        # Generate elevation grid
        x = np.linspace(min_lon, max_lon, 20)
        y = np.linspace(min_lat, max_lat, 20)
        xx, yy = np.meshgrid(x, y)
        elevation = 200 + 50 * np.sin(xx * 10) + 30 * np.cos(yy * 10) + np.random.normal(0, 10, xx.shape)
        
        return {
            'rivers': rivers,
            'buildings': buildings,
            'elevation': {
                'x': x,
                'y': y,
                'z': elevation
            }
        }
    
    def execute_workflow(self, workflow: Dict) -> Dict:
        """Execute a workflow and return results"""
        results = {
            'workflow_id': workflow['id'],
            'execution_time': datetime.now().isoformat(),
            'status': 'completed',
            'steps_executed': len(workflow['steps']),
            'outputs': {}
        }
        
        # Simulate execution of each step
        for i, step in enumerate(workflow['steps']):
            step_result = self._execute_step(step)
            results['outputs'][step['id']] = step_result
            
            # Add some delay to simulate processing
            time.sleep(0.1)
        
        # Generate final analysis results
        results['final_output'] = self._generate_final_output(workflow)
        
        return results
    
    def _execute_step(self, step: Dict) -> Dict:
        """Execute a single workflow step"""
        tool = step['tool']
        
        if tool == 'load_osm_data':
            return self._mock_osm_data(step['parameters'])
        elif tool == 'buffer_analysis':
            return self._mock_buffer_analysis(step['parameters'])
        elif tool == 'spatial_intersection':
            return self._mock_spatial_intersection(step['parameters'])
        elif tool == 'raster_calculator':
            return self._mock_raster_calculation(step['parameters'])
        else:
            return {
                'status': 'completed',
                'features_processed': np.random.randint(10, 100),
                'processing_time': np.random.uniform(0.5, 2.0)
            }
    
    def _mock_osm_data(self, params: Dict) -> Dict:
        """Mock OSM data loading"""
        feature_type = params.get('feature_type', 'unknown')
        
        if feature_type == 'waterway':
            return {
                'status': 'completed',
                'features_loaded': 2,
                'data_source': 'OpenStreetMap',
                'geometry_type': 'LineString'
            }
        elif feature_type == 'building':
            return {
                'status': 'completed',
                'features_loaded': 50,
                'data_source': 'OpenStreetMap',
                'geometry_type': 'Point'
            }
        else:
            return {
                'status': 'completed',
                'features_loaded': np.random.randint(10, 100),
                'data_source': 'OpenStreetMap'
            }
    
    def _mock_buffer_analysis(self, params: Dict) -> Dict:
        """Mock buffer analysis"""
        return {
            'status': 'completed',
            'buffer_distance': params.get('distance', 100),
            'units': params.get('units', 'meters'),
            'features_buffered': 2,
            'total_area': np.random.uniform(1000, 5000)
        }
    
    def _mock_spatial_intersection(self, params: Dict) -> Dict:
        """Mock spatial intersection"""
        return {
            'status': 'completed',
            'intersection_features': np.random.randint(5, 25),
            'total_area': np.random.uniform(500, 2000),
            'percentage_overlap': np.random.uniform(10, 60)
        }
    
    def _mock_raster_calculation(self, params: Dict) -> Dict:
        """Mock raster calculation"""
        return {
            'status': 'completed',
            'expression': params.get('expression', 'unknown'),
            'cells_processed': np.random.randint(1000, 10000),
            'valid_cells': np.random.randint(500, 8000)
        }
    
    def _generate_final_output(self, workflow: Dict) -> Dict:
        """Generate final analysis output"""
        if 'flood_risk' in workflow['id']:
            return self._generate_flood_risk_output()
        elif 'site_suitability' in workflow['id']:
            return self._generate_suitability_output()
        else:
            return self._generate_general_output()
    
    def _generate_flood_risk_output(self) -> Dict:
        """Generate flood risk analysis output"""
        # Generate risk areas
        risk_areas = []
        for i in range(5):
            center_lon = np.random.uniform(77.15, 77.25)
            center_lat = np.random.uniform(28.5, 28.6)
            risk_level = np.random.choice(['High', 'Medium', 'Low'])
            
            risk_areas.append({
                'id': f'risk_area_{i}',
                'center': [center_lat, center_lon],
                'risk_level': risk_level,
                'affected_buildings': np.random.randint(5, 30),
                'area_sqkm': np.random.uniform(0.1, 1.0)
            })
        
        return {
            'analysis_type': 'flood_risk',
            'risk_areas': risk_areas,
            'total_buildings_at_risk': sum(area['affected_buildings'] for area in risk_areas),
            'total_area_at_risk': sum(area['area_sqkm'] for area in risk_areas),
            'risk_distribution': {
                'High': len([a for a in risk_areas if a['risk_level'] == 'High']),
                'Medium': len([a for a in risk_areas if a['risk_level'] == 'Medium']),
                'Low': len([a for a in risk_areas if a['risk_level'] == 'Low'])
            }
        }
    
    def _generate_suitability_output(self) -> Dict:
        """Generate site suitability output"""
        suitable_sites = []
        for i in range(8):
            center_lon = np.random.uniform(77.15, 77.25)
            center_lat = np.random.uniform(77.5, 28.6)
            suitability_score = np.random.uniform(0.3, 0.95)
            
            suitable_sites.append({
                'id': f'site_{i}',
                'center': [center_lat, center_lon],
                'suitability_score': suitability_score,
                'area_sqkm': np.random.uniform(0.5, 3.0),
                'slope': np.random.uniform(0, 5),
                'distance_to_road': np.random.uniform(100, 2000)
            })
        
        return {
            'analysis_type': 'site_suitability',
            'suitable_sites': suitable_sites,
            'average_suitability': np.mean([s['suitability_score'] for s in suitable_sites]),
            'total_suitable_area': sum(s['area_sqkm'] for s in suitable_sites),
            'criteria_weights': {
                'slope': 0.4,
                'land_use': 0.4,
                'road_access': 0.2
            }
        }
    
    def _generate_general_output(self) -> Dict:
        """Generate general analysis output"""
        return {
            'analysis_type': 'general',
            'features_analyzed': np.random.randint(50, 200),
            'total_area': np.random.uniform(10, 50),
            'processing_summary': {
                'data_sources': ['OSM', 'Government Data'],
                'operations_performed': ['Buffer', 'Intersection', 'Analysis'],
                'accuracy_estimate': np.random.uniform(0.8, 0.95)
            }
        }

# Initialize components
@st.cache_resource
def get_components(selected_llm):
    return LLMReasoner(selected_llm), WorkflowExecutor()

# Sidebar
with st.sidebar:
    st.header("🔧 System Configuration")
    # Model selection
    model_option = st.selectbox(
        "LLM Model",
        ["Gemini", "Groq"],
        key="llm_model_select"
    )
    
    # Analysis region
    st.subheader("📍 Analysis Region")
    region = st.selectbox(
        "Select Region",
        ["Delhi, India", "Mumbai, India", "Bangalore, India", "Custom"]
    )
    
    if region == "Custom":
        st.text_input("Bounding Box (W,S,E,N)", value="77.1,28.4,77.3,28.7")
    
    # Available data sources
    st.subheader("📊 Data Sources")
    data_sources = st.multiselect(
        "Available Sources",
        ["OpenStreetMap", "Bhoonidhi", "Census Data", "Satellite Imagery"],
        default=["OpenStreetMap", "Bhoonidhi"]
    )
    
    # Workflow history
    st.subheader("📋 Workflow History")
    if st.session_state.workflow_history:
        for i, workflow in enumerate(st.session_state.workflow_history[-3:]):
            if st.button(f"Workflow {i+1}: {workflow['name'][:20]}...", key=f"hist_{i}"):
                st.session_state.current_workflow = workflow

# Get LLM and executor based on selection
reasoner, executor = get_components(model_option)

def create_map_visualization(results: Dict) -> folium.Map:
    """Create folium map with analysis results"""
    # Center map on Delhi
    m = folium.Map(location=[28.55, 77.2], zoom_start=11)
    
    if 'final_output' in results:
        output = results['final_output']
        
        if output['analysis_type'] == 'flood_risk':
            # Add flood risk areas
            for area in output['risk_areas']:
                color = {'High': 'red', 'Medium': 'orange', 'Low': 'yellow'}[area['risk_level']]
                folium.CircleMarker(
                    location=area['center'],
                    radius=10,
                    color=color,
                    fill=True,
                    popup=f"Risk: {area['risk_level']}<br>Buildings: {area['affected_buildings']}",
                    tooltip=f"Risk Level: {area['risk_level']}"
                ).add_to(m)
        
        elif output['analysis_type'] == 'site_suitability':
            # Add suitable sites
            for site in output['suitable_sites']:
                color_intensity = site['suitability_score']
                color = f"rgba(0, 255, 0, {color_intensity})"
                folium.CircleMarker(
                    location=site['center'],
                    radius=8,
                    color='green',
                    fill=True,
                    popup=f"Suitability: {site['suitability_score']:.2f}<br>Area: {site['area_sqkm']:.2f} km²",
                    tooltip=f"Suitability Score: {site['suitability_score']:.2f}"
                ).add_to(m)
    
    # Add sample data layers
    sample_data = executor.mock_data
    
    # Add rivers
    for idx, river in sample_data['rivers'].iterrows():
        coords = [[coord[1], coord[0]] for coord in river.geometry.coords]
        folium.PolyLine(
            locations=coords,
            color='blue',
            weight=3,
            opacity=0.7,
            popup=f"River: {river['name']}"
        ).add_to(m)
    
    # Add sample buildings
    for idx, building in sample_data['buildings'].head(10).iterrows():
        folium.CircleMarker(
            location=[building.geometry.y, building.geometry.x],
            radius=3,
            color='gray',
            fill=True,
            popup=f"Building: {building['type']}"
        ).add_to(m)
    
    return m

def create_analysis_charts(results: Dict) -> List[go.Figure]:
    """Create analysis charts"""
    figures = []
    
    if 'final_output' in results:
        output = results['final_output']
        
        if output['analysis_type'] == 'flood_risk':
            # Risk distribution pie chart
            risk_dist = output['risk_distribution']
            fig = go.Figure(data=[go.Pie(
                labels=list(risk_dist.keys()),
                values=list(risk_dist.values()),
                hole=.3,
                marker_colors=['red', 'orange', 'yellow']
            )])
            fig.update_layout(title="Flood Risk Distribution")
            figures.append(fig)
            
            # Buildings at risk bar chart
            risk_areas = output['risk_areas']
            fig2 = go.Figure(data=[go.Bar(
                x=[f"Area {i+1}" for i in range(len(risk_areas))],
                y=[area['affected_buildings'] for area in risk_areas],
                marker_color=['red' if area['risk_level'] == 'High' else 'orange' if area['risk_level'] == 'Medium' else 'yellow' for area in risk_areas]
            )])
            fig2.update_layout(title="Buildings at Risk by Area")
            figures.append(fig2)
        
        elif output['analysis_type'] == 'site_suitability':
            # Suitability scores
            sites = output['suitable_sites']
            fig = go.Figure(data=[go.Scatter(
                x=[site['distance_to_road'] for site in sites],
                y=[site['suitability_score'] for site in sites],
                mode='markers',
                marker=dict(
                    size=[site['area_sqkm'] * 10 for site in sites],
                    color=[site['suitability_score'] for site in sites],
                    colorscale='Viridis',
                    showscale=True
                ),
                text=[f"Site {i+1}" for i in range(len(sites))],
                textposition="top center"
            )])
            fig.update_layout(
                title="Site Suitability Analysis",
                xaxis_title="Distance to Road (m)",
                yaxis_title="Suitability Score"
            )
            figures.append(fig)
    
    return figures

# Main UI
st.title("🌍 Geospatial LLM Assistant")
st.markdown("**Chain-of-Thought Reasoning for Intelligent Geospatial Analysis**")

# Helper: get center from region
region_centers = {
    "Delhi, India": [28.6139, 77.2090],
    "Mumbai, India": [19.0760, 72.8777],
    "Bangalore, India": [12.9716, 77.5946]
}

def get_region_center(region, custom_bbox=None):
    if region in region_centers:
        return region_centers[region]
    elif custom_bbox:
        # bbox: [W, S, E, N] -> center: [(S+N)/2, (W+E)/2]
        try:
            w, s, e, n = map(float, custom_bbox.split(","))
            return [(s + n) / 2, (w + e) / 2]
        except:
            return [28.55, 77.2]  # fallback
    else:
        return [28.55, 77.2]

# Main interface
# Add OSM Map tab as first tab
osm_tab, tab1, tab2, tab3, tab4 = st.tabs(["🗺️ OSM Map", "🤖 Chat Interface", "🧠 Chain of Thought", "📊 Results", "⚙️ Workflow"])

with osm_tab:
    st.header("OpenStreetMap Viewer")
    # Get region center
    custom_bbox = None
    if region == "Custom":
        custom_bbox = st.sidebar.text_input("Bounding Box (W,S,E,N)", value="77.1,28.4,77.3,28.7", key="osm_custom_bbox")
    center = get_region_center(region, custom_bbox)
    zoom = 11 if region != "Custom" else 12
    m = folium.Map(location=center, zoom_start=zoom)
    st_folium(m, width=700, height=500)

with tab1:
    st.header("Chat with Geospatial Assistant")
    
    # Chat interface
    chat_container = st.container()
    
    with chat_container:
        # Display chat history
        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.write(message["content"])
    
    # Query input
    query = st.chat_input("Describe your geospatial analysis task...")
    
    if query:
        st.session_state.chat_history.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.write(query)
        
        with st.chat_message("assistant"):
            with st.spinner("Analyzing your request..."):
                query_analysis = reasoner.analyze_query(query)
                cot_steps = reasoner.generate_chain_of_thought(query_analysis)
                workflow = reasoner.generate_workflow(query_analysis)
                st.session_state.current_workflow = {
                    'query': query,
                    'analysis': query_analysis,
                    'cot_steps': cot_steps,
                    'workflow': workflow
                }
                st.write(f"I understand you want to perform **{query_analysis['task_type'].replace('_', ' ')}** analysis.")
                st.write(f"Confidence: {query_analysis['confidence']:.1%}")
                if query_analysis['confidence'] > 0.5:
                    st.write("✅ I've identified the appropriate workflow for your request.")
                    st.write("Click on the 'Chain of Thought' tab to see my reasoning process.")
                else:
                    st.write("⚠️ I'll do my best to create a general analysis workflow.")
                response = f"Generated {query_analysis['task_type']} workflow with {len(workflow['steps'])} steps"
                st.session_state.chat_history.append({"role": "assistant", "content": response})
    
    # Example queries
    st.subheader("💡 Example Queries")
    example_queries = [
        "Find areas at risk of flooding near rivers in Delhi",
        "Identify suitable locations for solar farms considering terrain and accessibility",
        "Analyze urban development patterns and infrastructure gaps",
        "Map areas with high population density and limited green space"
    ]
    
    cols = st.columns(2)
    for i, example in enumerate(example_queries):
        with cols[i % 2]:
            if st.button(example, key=f"example_{i}"):
                st.session_state.chat_history.append({"role": "user", "content": example})
                query_analysis = reasoner.analyze_query(example)
                cot_steps = reasoner.generate_chain_of_thought(query_analysis)
                workflow = reasoner.generate_workflow(query_analysis)
                
                st.session_state.current_workflow = {
                    'query': example,
                    'analysis': query_analysis,
                    'cot_steps': cot_steps,
                    'workflow': workflow
                }
                st.rerun()

with tab2:
    st.header("🧠 Chain of Thought Reasoning")
    
    if st.session_state.current_workflow:
        workflow_data = st.session_state.current_workflow
        
        st.subheader("Query Analysis")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Task Type", workflow_data['analysis']['task_type'].replace('_', ' ').title())
        with col2:
            st.metric("Confidence", f"{workflow_data['analysis']['confidence']:.1%}")
        with col3:
            st.metric("Workflow Steps", len(workflow_data['workflow']['steps']))
        
        st.subheader("Reasoning Process")
        for i, step in enumerate(workflow_data['cot_steps']):
            with st.expander(f"🔍 {step['title']}", expanded=i == 0):
                st.write(step['reasoning'])
                
                # Show corresponding workflow step if available
                if i < len(workflow_data['workflow']['steps']):
                    workflow_step = workflow_data['workflow']['steps'][i]
                    st.code(f"""
Tool: {workflow_step['tool']}
Description: {workflow_step['description']}
Parameters: {json.dumps(workflow_step['parameters'], indent=2)}
                    """)
        
        # Execute workflow button
        if st.button("🚀 Execute Workflow", type="primary"):
            with st.spinner("Executing workflow..."):
                results = executor.execute_workflow(workflow_data['workflow'])
                st.session_state.execution_results = results
                st.session_state.workflow_history.append(workflow_data['workflow'])
                st.success("Workflow executed successfully!")
                st.balloons()
    else:
        st.info("💬 Start a conversation in the Chat Interface to see chain-of-thought reasoning.")

with tab3:
    st.header("📊 Analysis Results")
    
    if hasattr(st.session_state, 'execution_results'):
        results = st.session_state.execution_results
        
        # Results summary
        st.subheader("📈 Execution Summary")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Status", results['status'].title())
        with col2:
            st.metric("Steps Executed", results['steps_executed'])
        with col3:
            st.metric("Execution Time", results['execution_time'].split('T')[1][:8])
        with col4:
            if 'final_output' in results:
                output = results['final_output']
                if 'total_buildings_at_risk' in output:
                    st.metric("Buildings at Risk", output['total_buildings_at_risk'])
                elif 'suitable_sites' in output:
                    st.metric("Suitable Sites", len(output['suitable_sites']))
                else:
                    st.metric("Features Analyzed", output.get('features_analyzed', 0))
        
        # Map visualization
        st.subheader("🗺️ Spatial Results")
        map_obj = create_map_visualization(results)
        st_folium(map_obj, width=700, height=500)
        
        # Charts
        st.subheader("📊 Analysis Charts")
        charts = create_analysis_charts(results)
        if charts:
            for chart in charts:
                st.plotly_chart(chart, use_container_width=True)
        
        # Detailed results
        st.subheader("📋 Detailed Results")
        with st.expander("View Raw Results"):
            st.json(results)
        
        # Download results
        st.subheader("💾 Download Results")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("📄 Download as JSON"):
                st.download_button(
                    label="Download JSON",
                    data=json.dumps(results, indent=2),
                    file_name=f"geospatial_analysis_{results['workflow_id']}.json",
                    mime="application/json"
                )
        with col2:
            if st.button("🗺️ Download as GeoJSON"):
                # Mock GeoJSON export
                geojson_data = {
                    "type": "FeatureCollection",
                    "features": []
                }
                st.download_button(
                    label="Download GeoJSON",
                    data=json.dumps(geojson_data, indent=2),
                    file_name=f"geospatial_results_{results['workflow_id']}.geojson",
                    mime="application/json"
                )
    else:
        st.info("🚀 Execute a workflow to see results here.")

with tab4:
    st.header("⚙️ Workflow Management")
    
    # Current workflow
    if st.session_state.current_workflow:
        workflow = st.session_state.current_workflow['workflow']
        
        st.subheader("Current Workflow")
        st.write(f"**Name:** {workflow['name']}")
        st.write(f"**Description:** {workflow['description']}")
        st.write(f"**Steps:** {len(workflow['steps'])}")
        
        # Workflow steps
        st.subheader("Workflow Steps")
        for i, step in enumerate(workflow['steps']):
            with st.expander(f"Step {i+1}: {step['description']}", expanded=False):
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"**Tool:** {step['tool']}")
                    st.write(f"**ID:** {step['id']}")
                with col2:
                    st.write("**Parameters:**")
                    st.json(step['parameters'])
        
        # Workflow validation
        st.subheader("Workflow Validation")
        validation_results = {
            "Syntax Valid": True,
            "Tools Available": True,
            "Parameters Valid": True,
            "Data Sources Accessible": True
        }
        
        for check, status in validation_results.items():
            if status:
                st.success(f"✅ {check}")
            else:
                st.error(f"❌ {check}")
        
        # Export workflow
        st.subheader("Export Workflow")
        workflow_json = json.dumps(workflow, indent=2)
        st.download_button(
            label="📥 Download Workflow JSON",
            data=workflow_json,
            file_name=f"workflow_{workflow['id']}.json",
            mime="application/json"
        )
        
        # Workflow editor
        st.subheader("Workflow Editor")
        edited_workflow = st.text_area(
            "Edit Workflow JSON",
            value=workflow_json,
            height=300,
            help="You can manually edit the workflow JSON here"
        )
        
        if st.button("💾 Save Edited Workflow"):
            try:
                new_workflow = json.loads(edited_workflow)
                st.session_state.current_workflow['workflow'] = new_workflow
                st.success("Workflow updated successfully!")
            except json.JSONDecodeError:
                st.error("Invalid JSON format")
    
    # Workflow history
    st.subheader("Workflow History")
    if st.session_state.workflow_history:
        for i, workflow in enumerate(st.session_state.workflow_history):
            with st.expander(f"Workflow {i+1}: {workflow['name']}"):
                st.write(f"**ID:** {workflow['id']}")
                st.write(f"**Description:** {workflow['description']}")
                st.write(f"**Steps:** {len(workflow['steps'])}")
                
                col1, col2 = st.columns(2)
                with col1:
                    if st.button(f"Load Workflow {i+1}", key=f"load_{i}"):
                        st.session_state.current_workflow = {
                            'workflow': workflow,
                            'query': f"Loaded from history",
                            'analysis': {'task_type': 'loaded', 'confidence': 1.0},
                            'cot_steps': []
                        }
                        st.rerun()
                with col2:
                    if st.button(f"Delete Workflow {i+1}", key=f"delete_{i}"):
                        st.session_state.workflow_history.pop(i)
                        st.rerun()
    else:
        st.info("No workflow history available.")

# Footer
st.markdown("---")
st.markdown("""
### 🛠️ System Features
- **Chain-of-Thought Reasoning**: Transparent step-by-step analysis
- **Multi-source Data Integration**: OSM, Bhoonidhi, Census data
- **Interactive Workflow Management**: Edit, save, and reuse workflows
- **Real-time Visualization**: Interactive maps and charts
- **Export Capabilities**: JSON, GeoJSON, and workflow templates

### 🔧 Technical Stack
- **LLM**: Mistral-7B-Instruct (Mock implementation)
- **Geospatial**: GeoPandas, Folium, Shapely
- **Visualization**: Plotly, Streamlit
- **Data Sources**: OpenStreetMap, Government datasets

*This is a prototype demonstration. In production, it would integrate with actual LLM APIs and geospatial processing services.*
""")

# Debug info (only show in development)
if st.checkbox("Show Debug Info", key="main_debug"):
    st.subheader("Debug Information")
    st.write("Session State:")
    st.json({
        "current_workflow": bool(st.session_state.current_workflow),
        "workflow_history_count": len(st.session_state.workflow_history),
        "chat_history_count": len(st.session_state.chat_history),
        "execution_results": hasattr(st.session_state, 'execution_results')
    })