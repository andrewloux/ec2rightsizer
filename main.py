import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

class AWSInstanceSelector:
    def __init__(self, csv_path):
        self.df = pd.read_csv(csv_path)
        self.preprocess_data()

    def preprocess_data(self):
        """Clean and preprocess the CSV data"""
        # Extract numeric memory value
        self.df['memoryGiB'] = self.df['Instance Memory'].str.extract(r'(\d+\.?\d*)').astype(float)
        
        # Extract numeric vCPU value (ignore burst info)
        self.df['vCPU'] = self.df['vCPUs'].str.extract(r'(\d+)').astype(int)
        
        # Clean price fields (remove "$" and "hourly", convert to float)
        self.df['pricePerHour'] = self.df['On Demand'].str.extract(r'\$(\d+\.?\d*)').astype(float)
        
        # Filter out instances with no pricing (unavailable)
        self.df = self.df[self.df['On Demand'] != 'unavailable']
        
        # Determine instance type from API Name
        self.df['type'] = self.df['API Name'].apply(self._get_instance_type)
        
        # Clean up instance type for display
        self.df['instanceType'] = self.df['API Name']

    def _get_instance_type(self, api_name):
        """Determine instance type from API name"""
        if pd.isna(api_name) or not isinstance(api_name, str):
            return 'Unknown'
        
        # Extract instance family (first character(s) before number)
        try:
            # Get the instance family (e.g., t4g, m6g, c6i)
            instance_family = ''.join(c for c in api_name.split('.')[0] if not c.isdigit())
            
            # Map common instance families to types
            type_mapping = {
                't': 'BURST',
                'm': 'GENERAL',
                'c': 'COMPUTE',
                'r': 'MEMORY',
                'i': 'STORAGE',
                'g': 'GPU',
                'p': 'GPU',
                'x': 'MEMORY',
                'd': 'STORAGE',
                'f': 'COMPUTE',
                'z': 'MEMORY',
                'a': 'ARM'
            }
            
            # Check if it's a metal instance
            if 'metal' in api_name.lower():
                return 'METAL'
            
            # Get the base type from the first letter of the family
            base_type = type_mapping.get(instance_family[0].lower(), 'OTHER')
            
            return base_type
            
        except (AttributeError, IndexError):
            return 'Unknown'

    def calculate_efficiency_score(self, memory_weight=0.5):
        """
        Calculate efficiency score based on memory/CPU weight ratio
        memory_weight: 0.0 (all CPU) to 1.0 (all memory)
        """
        cpu_weight = 1 - memory_weight
        
        # Calculate individual efficiency metrics
        cpu_efficiency = self.df['vCPU'] / self.df['pricePerHour']
        memory_efficiency = self.df['memoryGiB'] / self.df['pricePerHour']
        
        # Normalize the metrics
        cpu_efficiency_norm = (cpu_efficiency - cpu_efficiency.min()) / (cpu_efficiency.max() - cpu_efficiency.min())
        memory_efficiency_norm = (memory_efficiency - memory_efficiency.min()) / (memory_efficiency.max() - memory_efficiency.min())
        
        # Calculate weighted score
        return (memory_efficiency_norm * memory_weight + 
                cpu_efficiency_norm * cpu_weight)

    def get_best_instances(self, memory_weight=0.5, instance_types=None, family_prefix=None, required_memory=None, required_vcpus=None, top_n=5, max_instances=4):
        """Get best instances based on requirements and preferences, including combinations"""
        working_df = self.df.copy()
        
        # Apply filters
        if instance_types:
            working_df = working_df[working_df['type'].isin(instance_types)]
        if family_prefix:
            working_df = working_df[working_df['API Name'].str.startswith(family_prefix, na=False)]
        
        # Calculate base efficiency score
        working_df['efficiency_score'] = self.calculate_efficiency_score(memory_weight)
        
        # If no requirements, return single instances sorted by efficiency
        if not (required_memory and required_vcpus):
            return working_df.nlargest(top_n, 'efficiency_score')
        
        # Create combinations dataframe
        combinations = []
        
        # Consider single instances and combinations
        for count in range(1, max_instances + 1):
            for _, instance in working_df.iterrows():
                total_memory = instance['memoryGiB'] * count
                total_vcpu = instance['vCPU'] * count
                total_cost = instance['pricePerHour'] * count
                
                # Check if combination meets requirements
                if total_memory >= required_memory and total_vcpu >= required_vcpus:
                    memory_overhead = (total_memory - required_memory) / required_memory
                    cpu_overhead = (total_vcpu - required_vcpus) / required_vcpus
                    
                    # Calculate fit score
                    fit_score = 1 / (1 + memory_overhead + cpu_overhead)
                    
                    # Calculate cost efficiency (resources per dollar)
                    cost_efficiency = (total_memory * memory_weight + total_vcpu * (1-memory_weight)) / total_cost
                    
                    combinations.append({
                        'instanceType': instance['instanceType'],
                        'count': count,
                        'total_memory': total_memory,
                        'total_vcpu': total_vcpu,
                        'total_cost': total_cost,
                        'fit_score': fit_score,
                        'cost_efficiency': cost_efficiency,
                        'final_score': fit_score * cost_efficiency
                    })
        
        # Convert to dataframe and sort
        combinations_df = pd.DataFrame(combinations)
        if combinations_df.empty:
            return pd.DataFrame()  # Return empty if no valid combinations
            
        return combinations_df.nlargest(top_n, 'final_score')

# Create the Streamlit web app
def main():
    st.title("AWS Instance Optimizer")
    st.sidebar.header("Filters")

    # Load data
    try:
        selector = AWSInstanceSelector('aws_instances.csv')
    except Exception as e:
        st.error(f"Error loading CSV: {str(e)}")
        return

    # Sidebar controls
    memory_weight = st.sidebar.slider(
        "Memory vs CPU Preference",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        help="Slide left for CPU preference, right for memory preference",
        label_visibility="visible"
    )
    
    # Add labels for the slider
    col1, col2 = st.sidebar.columns(2)
    with col1:
        st.caption("← CPU Priority")
    with col2:
        st.caption("Memory Priority →")

    # Add cluster requirements section
    st.sidebar.markdown("---")  # Add separator
    st.sidebar.subheader("Cluster Requirements")
    
    required_memory = st.sidebar.number_input(
        "Required Memory (GiB)",
        min_value=1,
        value=48,
        help="Required memory in GiB (including overhead)"
    )
    
    required_vcpus = st.sidebar.number_input(
        "Required vCPUs",
        min_value=1,
        value=24,
        help="Required vCPUs (including overhead)"
    )

    # Instance type filter
    available_types = selector.df['type'].unique()
    selected_types = st.sidebar.multiselect(
        "Filter Instance Types",
        options=available_types,
        default=available_types
    )

    # Add instance family prefix filter
    family_prefix = st.sidebar.text_input(
        "Instance Family Prefix (e.g., m6g, c6i)",
        help="Filter instances by family prefix (leave empty for all)",
        value=""
    )

    # Number of results
    top_n = st.sidebar.number_input(
        "Number of results",
        min_value=1,
        max_value=50,
        value=10
    )

    # Add max instances selector
    max_instances = st.sidebar.number_input(
        "Maximum instances per combination",
        min_value=1,
        max_value=10,
        value=4,
        help="Maximum number of identical instances to consider in combinations"
    )

    # Add a prominent Calculate button
    st.sidebar.markdown("---")
    calculate_button = st.sidebar.button("Calculate Recommendations", type="primary")

    if calculate_button:
        # Get results
        results = selector.get_best_instances(
            memory_weight=memory_weight,
            instance_types=selected_types,
            family_prefix=family_prefix if family_prefix else None,
            required_memory=required_memory,
            required_vcpus=required_vcpus,
            max_instances=max_instances,
            top_n=top_n
        )
    else:
        results = pd.DataFrame()  # Empty results until Calculate is clicked

    # Display results in full width
    st.subheader("Top Instance Recommendations")
    
    if results.empty:
        st.warning("Click 'Calculate Recommendations' to see results")
    else:
        # Modified table for combinations
        if 'count' in results.columns:  # If we're showing combinations
            st.dataframe(
                results[[
                    'instanceType', 'count', 'total_memory', 
                    'total_vcpu', 'total_cost', 'final_score'
                ]].style.format({
                    'total_memory': '{:.1f} GiB',
                    'total_cost': '${:.3f}/hr',
                    'final_score': '{:.3f}'
                })
            )
            
            # Full width visualizations for combinations
            st.subheader("Cost vs Resources")
            fig = px.scatter(
                results,
                x='total_cost',
                y='final_score',
                size='count',
                hover_data=['instanceType', 'total_memory', 'total_vcpu'],
                title='Cost Efficiency of Instance Combinations'
            )
            st.plotly_chart(fig, use_container_width=True)
            
            st.subheader("Resource Distribution")
            fig2 = px.bar(
                results.head(),
                x='instanceType',
                y=['total_memory', 'total_vcpu'],
                title='Resource Distribution by Instance Type',
                barmode='group'
            )
            st.plotly_chart(fig2, use_container_width=True)

    # Add price efficiency analysis for single instances
    if not results.empty and 'count' not in results.columns:
        st.subheader("Price Efficiency Analysis")
        fig3 = go.Figure()
        
        fig3.add_trace(go.Scatter(
            x=results['pricePerHour'],
            y=results['vCPU'] / results['pricePerHour'],
            mode='markers',
            name='CPU Efficiency',
            text=results['instanceType']
        ))
        
        fig3.add_trace(go.Scatter(
            x=results['pricePerHour'],
            y=results['memoryGiB'] / results['pricePerHour'],
            mode='markers',
            name='Memory Efficiency',
            text=results['instanceType']
        ))
        
        fig3.update_layout(
            title='Price Efficiency (Higher is Better)',
            xaxis_title='Price per Hour ($)',
            yaxis_title='Resources per Dollar'
        )
        
        st.plotly_chart(fig3)

if __name__ == "__main__":
    main()
