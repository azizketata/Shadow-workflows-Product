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
        tuple: (graphviz.Digraph, list of activities, BPMN object)
    """
    if not agenda_text or not api_key:
        return None, [], None

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
        
        # Build BPMN using PM4Py 2.7.x API
        # In recent versions, we add nodes directly to the graph using add_node() and add_flow()
        bpmn_graph = BPMN()
        
        # Create nodes
        start_event = BPMN.StartEvent(name="Start")
        bpmn_graph.add_node(start_event)
        
        end_event = BPMN.EndEvent(name="End")
        bpmn_graph.add_node(end_event)
        
        previous_node = start_event
        
        for i, act_label in enumerate(activities):
            task = BPMN.Task(name=act_label)
            bpmn_graph.add_node(task)
            
            # Add flow from previous node to current task
            flow = BPMN.Flow(previous_node, task)
            bpmn_graph.add_flow(flow)
            
            previous_node = task
        
        # Connect last task to end event
        final_flow = BPMN.Flow(previous_node, end_event)
        bpmn_graph.add_flow(final_flow)

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
        
        return gviz, activities, bpmn_graph

    except Exception as e:
        print(f"Error generating BPMN: {e}")
        # Return a simple error graph or None
        import graphviz
        err_g = graphviz.Digraph()
        err_g.node('Error', f"Error generating BPMN: {str(e)}")
        # Return None for the BPMN object so we don't crash downstream
        return err_g, [], None

def generate_colored_bpmn(bpmn_graph, alignments):
    """
    Generates a color-coded BPMN visualization based on alignment diagnostics.
    
    Args:
        bpmn_graph (pm4py.objects.bpmn.obj.BPMN): The reference BPMN model.
        alignments (list): List of alignment dictionaries from pm4py.
        
    Returns:
        graphviz.Digraph: The colored graph.
    """
    from pm4py.visualization.bpmn import visualizer as bpmn_visualizer
    import graphviz
    
    if not bpmn_graph or not alignments:
        return None

    # 1. Analyze Alignments to count moves
    # Structure of alignment['alignment'] is a list of tuples: (log_move, model_move)
    # log_move: activity name or None (>> in pm4py usually)
    # model_move: activity name or None (>> in pm4py usually)
    
    model_move_counts = {} # Activity -> Count of "Model Move Only" (Skipped)
    log_move_counts = {}   # Activity -> Count of "Log Move Only" (Deviation)
    sync_move_counts = {}  # Activity -> Count of "Sync" (Match)
    
    for align in alignments:
        trace_alignment = align['alignment']
        for log_move, model_move in trace_alignment:
            # Normalize None/Skipped representation
            # PM4Py uses '>>' string often in alignment printing, but strictly (str, str) or (str, None) or (None, str) in object?
            # Let's check typical pm4py output format. Usually it is a tuple of labels.
            # '>>' is used for "no move".
            
            is_log_skip = (log_move is None or log_move == '>>')
            is_model_skip = (model_move is None or model_move == '>>')
            
            if not is_log_skip and not is_model_skip:
                if log_move == model_move:
                    sync_move_counts[model_move] = sync_move_counts.get(model_move, 0) + 1
            elif is_log_skip and not is_model_skip:
                # Model Move Only -> Skipped in reality
                model_move_counts[model_move] = model_move_counts.get(model_move, 0) + 1
            elif not is_log_skip and is_model_skip:
                # Log Move Only -> Shadow activity
                log_move_counts[log_move] = log_move_counts.get(log_move, 0) + 1
                
    # 2. Generate Base Graphviz Object
    # We use pm4py's visualizer to get the structure, then modify attributes
    parameters = {bpmn_visualizer.Variants.CLASSIC.value.Parameters.FORMAT: "svg"}
    gviz = bpmn_visualizer.apply(bpmn_graph, parameters=parameters)
    
    # 3. Apply Colors
    # We need to map BPMN Node IDs (from the visualizer) to Activity Names
    # This is tricky because visualizer creates its own IDs.
    # However, graphviz nodes usually have 'label' attribute.
    
    # Parse graphviz source or modify nodes directly?
    # graphviz.Digraph object allows iterating body? No, it's a list of strings.
    # Better approach: Re-build or use string replacement on the source if simple,
    # OR use pm4py's frequency decoration if possible (but we want specific colors).
    
    # Let's iterate through the gviz.body to find node definitions and modify styles.
    # Typical line: 123 [label="Activity Name", shape=box, ...];
    
    new_body = []
    for line in gviz.body:
        if 'label=' in line and 'shape=' in line:
            # It's a node
            # Extract Label
            try:
                # simplified parsing
                start_quote = line.find('label="') + 7
                end_quote = line.find('"', start_quote)
                label = line[start_quote:end_quote]
                
                # Determine Color
                fill_color = "white"
                pen_width = "1"
                color = "black"
                
                # Logic:
                # If mostly Sync -> Green
                # If mostly Skipped -> Grey/Dashed
                # If mostly Log Move (This is harder because Log Moves might not correspond to Model Nodes)
                # But wait, Log Moves (Shadow) often are NOT in the model, so we can't color a model node for them!
                # We can only color existing nodes as "Skipped" or "Executed".
                # Shadow activities would need NEW nodes added, which is hard with just coloring.
                # For RQ2 visualization, coloring the model shows CONFORMANCE (what matched/skipped).
                # Shadow activities (Log Moves) are best shown in a list or by adding dummy nodes.
                # Let's focus on coloring the Reference Model first.
                
                syncs = sync_move_counts.get(label, 0)
                skips = model_move_counts.get(label, 0)
                
                total = syncs + skips
                if total > 0:
                    if skips > syncs:
                        # Mostly Skipped
                        fill_color = "#f0f0f0" # light grey
                        color = "grey"
                        pen_width = "2"
                        style = "dashed,filled"
                    else:
                        # Mostly Executed
                        fill_color = "#e6ffe6" # light green
                        color = "green"
                        pen_width = "2"
                        style = "filled"
                else:
                    style = "filled" # Default
                    
                # Inject style
                # Replace the closing bracket '];' with our styles
                line = line.rstrip(';\n')
                line = line.rstrip(']')
                line += f', style="{style}", fillcolor="{fill_color}", color="{color}", penwidth="{pen_width}"];'
                
            except:
                pass
                
        new_body.append(line)
        
    gviz.body = new_body
    gviz.attr(rankdir='TB')
    
    # NOTE: To visualize "Log Moves" (Shadow Steps) that are NOT in the model,
    # we would ideally inject new nodes into the graph. 
    # For MVP, we will return a separate list of "Shadow Activities" to display in UI.
    
    return gviz

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
