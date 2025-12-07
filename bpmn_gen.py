import openai
import pm4py
from pm4py.objects.bpmn.obj import BPMN
import json
import tempfile
import os
import pandas as pd
import streamlit as st
from datetime import datetime

def generate_agenda_bpmn(agenda_text, api_key):
    """
    Generates a BPMN visualization from a meeting agenda text using OpenAI and pm4py.
    
    Args:
        agenda_text (str): The text of the meeting agenda.
        api_key (str): OpenAI API key.
        
    Returns:
        graphviz.Digraph: The visual representation of the BPMN model.
    """
    if not agenda_text or not api_key:
        return None

    try:
        client = openai.OpenAI(api_key=api_key)
        
        prompt = f"""
        Extract a sequential list of activities/topics from the following meeting agenda.
        Return ONLY a JSON array of strings, where each string is a concise activity label.
        Ensure the order matches the agenda.
        
        Agenda:
        {agenda_text}
        """
        
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that extracts process activities from text."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0
        )
        
        content = response.choices[0].message.content
        # robust parsing logic in case of extra text
        start_idx = content.find('[')
        end_idx = content.rfind(']') + 1
        if start_idx == -1 or end_idx == 0:
            raise ValueError("Could not find JSON array in response")
            
        activities = json.loads(content[start_idx:end_idx])
        
        # Build BPMN
        bpmn_graph = BPMN()
        process = BPMN.Process(id="process_1")
        bpmn_graph.set_process(process)
    
        start_event = BPMN.StartEvent(id="start", name="Start")
        process.append(start_event)
    
        end_event = BPMN.EndEvent(id="end", name="End")
        process.append(end_event)
    
        previous_node = start_event
        
        for i, act_label in enumerate(activities):
            task = BPMN.Task(id=f"task_{i}", name=act_label)
            process.append(task)
            
            flow = BPMN.SequenceFlow(previous_node, task)
            process.append(flow)
            
            previous_node = task
    
        # Connect last task to end event
        final_flow = BPMN.SequenceFlow(previous_node, end_event)
        process.append(final_flow)

        # layout top-to-bottom
        # pm4py's view_bpmn usually displays it. To get the graphviz object we can use the visualizer directly.
        from pm4py.visualization.bpmn import visualizer as bpmn_visualizer
        
        # Note: pm4py might default to left-to-right. We can try to adjust parameters if needed.
        # But 'rankdir': 'TB' is standard graphviz.
        parameters = {bpmn_visualizer.Variants.CLASSIC.value.Parameters.FORMAT: "svg"}
        gviz = bpmn_visualizer.apply(bpmn_graph, parameters=parameters)
        
        # Hack to force Top-to-Bottom if pm4py doesn't expose it easily in high level API
        # but typically Graphviz defaults are okay or we can modify the body.
        gviz.attr(rankdir='TB') 
        
        return gviz

    except Exception as e:
        print(f"Error generating BPMN: {e}")
        # Return a simple error graph or None
        import graphviz
        err_g = graphviz.Digraph()
        err_g.node('Error', f"Error generating BPMN: {str(e)}")
        return err_g

def convert_to_event_log(df):
    """
    Converts a pandas DataFrame with 'timestamp' and 'activity_name' to a PM4Py Event Log.
    Assumes all events belong to a single case 'Meeting_1'.
    """
    if df.empty:
        return None

    # Create a copy to avoid modifying original
    log_df = df.copy()
    
    # Assign a single case ID
    log_df['case:concept:name'] = 'Meeting_1'
    
    # Map columns to PM4Py standard names
    log_df['concept:name'] = log_df['activity_name']
    
    # Convert timestamp string (MM:SS) to datetime objects for PM4Py
    # We use a dummy date for reference
    def parse_time(t_str):
        try:
            return datetime.strptime(f"2023-01-01 00:{t_str}", "%Y-%m-%d %H:%M:%S")
        except:
            return datetime.now()
            
    log_df['time:timestamp'] = log_df['timestamp'].apply(parse_time)
    
    return log_df

def generate_discovered_bpmn(log_df):
    """
    Generates a BPMN graph from an event log using Inductive Miner.
    """
    if log_df is None or log_df.empty:
        return None
        
    try:
        # Discover BPMN model
        process_tree = pm4py.discover_process_tree_inductive(log_df)
        bpmn_model = pm4py.convert_to_bpmn(process_tree)
        
        # Visualize
        # pm4py.view_bpmn(bpmn_model) # This opens a window, we want graphviz object
        
        # Streamlit doesn't support direct PM4Py BPMN object rendering easily without conversion
        # But st.graphviz_chart takes a DOT string or source.
        # PM4Py's visualization returns a graphviz Digraph object
        gviz = pm4py.visualization.bpmn.visualizer.apply(bpmn_model)
        return gviz
    except Exception as e:
        st.error(f"Error generating BPMN: {e}")
        return None
