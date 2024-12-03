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
        self.df['pricePerHour'] = pd.to_numeric(
            self.df['On Demand'].str.extract(r'\$(\d+\.?\d*)')[0], 
            errors='coerce'
        )
        
        # Determine instance type from API Name
        self.df['type'] = self.df['API Name'].apply(self._get_instance_type)
        
        # Clean up instance type for display
        self.df['instanceType'] = self.df['API Name']
        
        # Print summary of available instances
        print(f"Total instances: {len(self.df)}")
        print(f"Instances with valid prices: {self.df['pricePerHour'].notna().sum()}")
        print("Instance types available:", self.df['type'].unique())

    def _get_instance_type(self, api_name):
        """Determine instance type from API name"""
        if pd.isna(api_name) or not isinstance(api_name, str):
            return 'Unknown'
        
        # Extract instance family (first character(s) before number)
        try:
            instance_family = api_name.split('.')[0].lower()
            
            # Map common instance families to types
            type_mapping = {
                't': 'BURST',  # Changed to catch all T-series instances
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
            if 'metal' in instance_family:
                return 'METAL'
            
            # Get the base type from the first letter
            base_type = type_mapping.get(instance_family[0], 'OTHER')
            
            return base_type
            
        except (AttributeError, IndexError):
            return 'Unknown'

    def calculate_efficiency_score(self, memory_weight=0.5):
        """Calculate efficiency score based on memory/CPU weight ratio"""
        # Calculate resource per dollar metrics
        cpu_efficiency = self.df['vCPU'] / self.df['pricePerHour']
        memory_efficiency = self.df['memoryGiB'] / self.df['pricePerHour']
        
        # Calculate z-scores to identify outliers
        cpu_zscore = (cpu_efficiency - cpu_efficiency.mean()) / cpu_efficiency.std()
        memory_zscore = (memory_efficiency - memory_efficiency.mean()) / memory_efficiency.std()
        
        # Cap extreme values
        cpu_zscore = cpu_zscore.clip(-3, 3)
        memory_zscore = memory_zscore.clip(-3, 3)
        
        # Normalize to 0-1 range
        cpu_norm = (cpu_zscore - cpu_zscore.min()) / (cpu_zscore.max() - cpu_zscore.min())
        memory_norm = (memory_zscore - memory_zscore.min()) / (memory_zscore.max() - memory_zscore.min())
        
        # Return weighted score
        return (memory_norm * memory_weight + cpu_norm * (1-memory_weight))

    def get_best_instances(self, memory_weight=0.5, instance_types=None, family_prefix=None, required_memory=None, required_vcpus=None, max_instances=4, top_n=5):
        """Get best instances based on requirements and preferences, including combinations"""
        working_df = self.df.copy()
        
        # Filter out instances with unavailable pricing
        working_df = working_df[working_df['pricePerHour'].notna()]
        
        # Apply filters
        if instance_types:
            working_df = working_df[working_df['type'].isin(instance_types)]
        if family_prefix:
            working_df = working_df[working_df['API Name'].str.startswith(family_prefix, na=False)]
            
        # If no requirements, return single instances sorted by efficiency
        if not (required_memory and required_vcpus):
            working_df['efficiency_score'] = self.calculate_efficiency_score(memory_weight)
            return working_df.nlargest(top_n, 'efficiency_score')
        
        # Create combinations dataframe
        combinations = []
        
        # Consider combinations of instances
        for _, instance in working_df.iterrows():
            for count in range(1, max_instances + 1):
                total_memory = instance['memoryGiB'] * count
                total_vcpu = instance['vCPU'] * count
                total_cost = instance['pricePerHour'] * count
                
                # Skip if resources are insufficient
                if total_memory < required_memory * 0.95 or total_vcpu < required_vcpus * 0.95:
                    continue
                
                # Calculate utilization efficiency
                memory_util = required_memory / total_memory
                vcpu_util = required_vcpus / total_vcpu
                
                # Skip if utilization is too low
                if memory_util < 0.6 or vcpu_util < 0.6:
                    continue
                
                # Calculate resource efficiency score
                resource_efficiency = (
                    memory_util * memory_weight +
                    vcpu_util * (1 - memory_weight)
                )
                
                # Calculate cost efficiency (resources per dollar)
                cost_efficiency = (
                    min(total_memory, required_memory * 1.2) * memory_weight +
                    min(total_vcpu, required_vcpus * 1.2) * (1 - memory_weight)
                ) / total_cost
                
                combinations.append({
                    'instanceType': instance['instanceType'],
                    'count': count,
                    'total_memory': total_memory,
                    'total_vcpu': total_vcpu,
                    'total_cost': total_cost,
                    'memory_util': memory_util * 100,  # Convert to percentage
                    'cpu_util': vcpu_util * 100,      # Convert to percentage
                    'resource_efficiency': resource_efficiency,
                    'cost_efficiency': cost_efficiency
                })
        
        # Convert to dataframe
        combinations_df = pd.DataFrame(combinations)
        if combinations_df.empty:
            return pd.DataFrame()
        
        # Normalize cost efficiency
        combinations_df['cost_efficiency_norm'] = (
            combinations_df['cost_efficiency'] - combinations_df['cost_efficiency'].min()
        ) / (
            combinations_df['cost_efficiency'].max() - combinations_df['cost_efficiency'].min()
        )
        
        # Calculate final score
        combinations_df['final_score'] = (
            combinations_df['resource_efficiency'] * 0.6 +  # Prioritize good resource utilization
            combinations_df['cost_efficiency_norm'] * 0.4   # Consider cost efficiency
        )
        
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

    # Initialize session state for results if not exists
    if 'results_calculated' not in st.session_state:
        st.session_state.results_calculated = False

    if calculate_button:
        st.session_state.results_calculated = True
        with st.spinner('Calculating recommendations...'):
            results = selector.get_best_instances(
                memory_weight=memory_weight,
                instance_types=selected_types if selected_types else None,
                family_prefix=family_prefix if family_prefix else None,
                required_memory=required_memory if required_memory > 0 else None,
                required_vcpus=required_vcpus if required_vcpus > 0 else None,
                max_instances=max_instances,
                top_n=top_n
            )
    elif not st.session_state.results_calculated:
        st.info("👆 Configure your requirements and click 'Calculate Recommendations' to see results")
        return

    if results.empty:
        st.warning("No instances found matching your criteria. Try adjusting the filters.")
    else:
        # 1. Top Recommendations Table
        st.subheader("Top Instance Recommendations")
        if 'count' in results.columns:  # If showing combinations
            st.dataframe(
                results[[
                    'instanceType', 'count', 'total_memory', 
                    'total_vcpu', 'total_cost', 'memory_util',
                    'cpu_util', 'final_score'
                ]].style.format({
                    'total_memory': '{:.1f} GiB',
                    'total_cost': '${:.3f}/hr',
                    'memory_util': '{:.1f}%',
                    'cpu_util': '{:.1f}%',
                    'final_score': '{:.3f}'
                })
            )

        # 2. Memory vs CPU Scatter Plot
        st.subheader("Memory vs CPU Comparison")
        if 'count' in results.columns:  # For combinations
            fig1 = px.scatter(
                results,
                x='total_vcpu',
                y='total_memory',
                size='total_cost',
                color='final_score',
                hover_data=['instanceType', 'count', 'memory_util', 'cpu_util'],
                labels={
                    'total_vcpu': 'Total vCPUs',
                    'total_memory': 'Total Memory (GiB)',
                    'total_cost': 'Total Cost per Hour ($)',
                    'final_score': 'Final Score',
                    'memory_util': 'Memory Utilization (%)',
                    'cpu_util': 'CPU Utilization (%)'
                }
            )
        else:  # For single instances
            fig1 = px.scatter(
                results,
                x='vCPU',
                y='memoryGiB',
                size='pricePerHour',
                color='efficiency_score',
                hover_data=['instanceType', 'pricePerHour'],
                labels={
                    'vCPU': 'vCPUs',
                    'memoryGiB': 'Memory (GiB)',
                    'pricePerHour': 'Price per Hour ($)',
                    'efficiency_score': 'Efficiency Score'
                }
            )
        st.plotly_chart(fig1)

        # 3. Efficiency Score Bar Chart
        st.subheader("Instance Scores")
        if 'count' in results.columns:
            fig2 = px.bar(
                results,
                x='instanceType',
                y='final_score',
                color='count',
                labels={
                    'instanceType': 'Instance Type',
                    'final_score': 'Final Score',
                    'count': 'Instance Count'
                }
            )
        else:
            fig2 = px.bar(
                results,
                x='instanceType',
                y='efficiency_score',
                color='type',
                labels={
                    'instanceType': 'Instance Type',
                    'efficiency_score': 'Efficiency Score',
                    'type': 'Instance Family'
                }
            )
        fig2.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig2)

        # Only show these visualizations for single instances
        if 'count' not in results.columns:
            # 4. Price Efficiency Analysis
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

        # 5. Cost Distribution
        st.subheader("Cost Distribution")
        if 'count' in results.columns:
            fig4 = px.box(
                results,
                x='instanceType',
                y='total_cost',
                points="all",
                labels={
                    'instanceType': 'Instance Type',
                    'total_cost': 'Total Cost per Hour ($)'
                }
            )
        else:
            fig4 = px.box(
                results,
                x='type',
                y='pricePerHour',
                points="all",
                labels={
                    'type': 'Instance Type',
                    'pricePerHour': 'Price per Hour ($)'
                }
            )
        st.plotly_chart(fig4)

        # 6. Memory/CPU Ratio Analysis
        st.subheader("Resource Ratio Analysis")
        if 'count' in results.columns:
            results['mem_cpu_ratio'] = results['total_memory'] / results['total_vcpu']
            fig5 = px.scatter(
                results,
                x='total_cost',
                y='mem_cpu_ratio',
                color='count',
                size='total_vcpu',
                hover_data=['instanceType', 'memory_util', 'cpu_util'],
                labels={
                    'total_cost': 'Total Cost per Hour ($)',
                    'mem_cpu_ratio': 'Memory (GiB) per vCPU',
                    'count': 'Instance Count',
                    'memory_util': 'Memory Utilization (%)',
                    'cpu_util': 'CPU Utilization (%)'
                }
            )
        else:
            results['mem_cpu_ratio'] = results['memoryGiB'] / results['vCPU']
            fig5 = px.scatter(
                results,
                x='pricePerHour',
                y='mem_cpu_ratio',
                color='type',
                size='vCPU',
                hover_data=['instanceType'],
                labels={
                    'pricePerHour': 'Price per Hour ($)',
                    'mem_cpu_ratio': 'Memory (GiB) per vCPU',
                    'type': 'Instance Family'
                }
            )
        st.plotly_chart(fig5)

if __name__ == "__main__":
    main()
