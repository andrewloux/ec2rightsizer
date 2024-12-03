import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import chi2


class AWSInstanceSelector:
    def __init__(self, csv_path):
        self.df = pd.read_csv(csv_path)
        self.preprocess_data()

    def preprocess_data(self):
        """Clean and preprocess the CSV data"""
        # Extract numeric memory value
        self.df['memoryGiB'] = self.df['Instance Memory'].str.extract(
            r'(\d+\.?\d*)').astype(float)

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

    def calculate_node_combinations(
            self,
            target_memory,
            target_vcpu,
            max_nodes=5):
        """
        Calculate possible node combinations that meet requirements while optimizing for cost
        """
        combinations = []

        # Filter instances based on price availability
        valid_instances = self.df[self.df['pricePerHour'].notna()].copy()

        for _, instance in valid_instances.iterrows():
            for num_nodes in range(1, max_nodes + 1):
                total_memory = instance['memoryGiB'] * num_nodes
                total_vcpu = instance['vCPU'] * num_nodes
                total_cost = instance['pricePerHour'] * num_nodes

                # Only consider if meets minimum requirements
                if total_memory >= target_memory and total_vcpu >= target_vcpu:
                    # Calculate utilization percentages
                    memory_utilization = (target_memory / total_memory) * 100
                    cpu_utilization = (target_vcpu / total_vcpu) * 100

                    # Calculate reliability score
                    reliability_score = self.calculate_reliability_score(
                        num_nodes)

                    combinations.append({
                        'instance_type': instance['API Name'],
                        'num_nodes': num_nodes,
                        'total_memory': total_memory,
                        'total_vcpu': total_vcpu,
                        'total_cost': total_cost,
                        'memory_utilization': memory_utilization,
                        'cpu_utilization': cpu_utilization,
                        'cost_per_usable_memory': (total_cost / target_memory),
                        'cost_per_usable_vcpu': (total_cost / target_vcpu),
                        'reliability_score': reliability_score
                    })

        return pd.DataFrame(combinations)

    def calculate_reliability_score(self, num_nodes):
        """
        Calculate reliability score based on node count and high-availability patterns
        """
        reliability_map = {
            1: 0.4,   # No HA
            2: 0.85,  # Basic HA
            3: 1.0,
            4: 0.95,
            5: 0.98,  # Improved failure tolerance
        }
        return reliability_map.get(num_nodes, 0.90)  # 6+ nodes

    def score_combinations(self, combinations_df, weight_config=None):
        """
        Score combinations based on cost, utilization, and reliability
        """
        if weight_config is None:
            weight_config = {
                'cost': 0.40,
                'resource_utilization': 0.35,
                'reliability': 0.25
            }

        # Normalize cost scores (lower is better)
        min_cost = combinations_df['total_cost'].min()
        cost_scores = min_cost / combinations_df['total_cost']

        # Calculate resource utilization score (closer to 100% is better)
        def score_utilization(util_pct):
            if util_pct > 100:
                return 0  # Severely penalize under-provisioning
            elif util_pct >= 80:
                return 1.0  # Optimal range
            else:
                return util_pct / 80  # Linear penalty for over-provisioning

        utilization_scores = (
            combinations_df['memory_utilization'].apply(score_utilization) *
            0.5 +
            combinations_df['cpu_utilization'].apply(score_utilization) *
            0.5)

        # Combine all scores with weights
        final_scores = (
            cost_scores * weight_config['cost'] +
            utilization_scores * weight_config['resource_utilization'] +
            combinations_df['reliability_score'] * weight_config['reliability']
        )

        return final_scores

    def get_optimal_combinations(
            self,
            target_memory,
            target_vcpu,
            max_nodes=5,
            weight_config=None):
        """
        Get optimal node combinations with detailed analysis
        """
        if weight_config is None:
            weight_config = {
                'cost': 0.40,
                'resource_utilization': 0.35,
                'reliability': 0.25
            }

        # Get all possible combinations
        combinations = self.calculate_node_combinations(
            target_memory,
            target_vcpu,
            max_nodes
        )

        if combinations.empty:
            return pd.DataFrame()

        # Score combinations
        combinations['final_score'] = self.score_combinations(
            combinations,
            weight_config
        )

        # Sort by final score
        return combinations.sort_values('final_score', ascending=False)

# Create the Streamlit web app


def main():
    st.title("AWS Instance Optimizer")
    st.sidebar.header("Configuration")

    # Load data
    try:
        selector = AWSInstanceSelector('aws_instances.csv')
    except Exception as e:
        st.error(f"Error loading CSV: {str(e)}")
        return

    # Sidebar controls for requirements
    st.sidebar.subheader("Resource Requirements")
    required_memory = st.sidebar.number_input(
        "Required Memory (GiB)",
        min_value=1,
        value=48,
        help="Required memory in GiB"
    )

    required_vcpus = st.sidebar.number_input(
        "Required vCPUs",
        min_value=1,
        value=24,
        help="Required vCPUs"
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
        help="Filter instances by family prefix (leave empty for all)"
    )

    # Scoring weights configuration
    st.sidebar.subheader("Scoring Weights")

    cost_weight = st.sidebar.slider(
        "Cost Importance",
        min_value=0.0,
        max_value=1.0,
        value=0.4,
        help="Weight given to cost optimization"
    )

    utilization_weight = st.sidebar.slider(
        "Resource Utilization Importance",
        min_value=0.0,
        max_value=1.0,
        value=0.35,
        help="Weight given to efficient resource utilization"
    )

    reliability_weight = st.sidebar.slider(
        "Reliability Importance",
        min_value=0.0,
        max_value=1.0,
        value=0.25,
        help="Weight given to high availability and reliability"
    )

    # Normalize weights to sum to 1
    total_weight = cost_weight + utilization_weight + reliability_weight
    weight_config = {
        'cost': cost_weight / total_weight,
        'resource_utilization': utilization_weight / total_weight,
        'reliability': reliability_weight / total_weight
    }

    # Maximum number of nodes
    max_nodes = st.sidebar.number_input(
        "Maximum Nodes",
        min_value=1,
        max_value=10,
        value=5,
        help="Maximum number of nodes to consider in combinations"
    )

    # Number of recommendations
    top_n = st.sidebar.number_input(
        "Number of Recommendations",
        min_value=1,
        max_value=20,
        value=5
    )

    # Calculate button
    if st.sidebar.button("Calculate Recommendations", type="primary"):
        with st.spinner('Calculating optimal configurations...'):
            # Get all possible combinations
            combinations = selector.get_optimal_combinations(
                target_memory=required_memory,
                target_vcpu=required_vcpus,
                max_nodes=max_nodes
            )

            if len(combinations) == 0:
                st.warning(
                    "No valid configurations found. Try adjusting your requirements.")
                return

            # Score combinations
            combinations['final_score'] = selector.score_combinations(
                combinations,
                weight_config=weight_config
            )

            # Sort by final score
            combinations = combinations.sort_values(
                'final_score', ascending=False)

            # Get top recommendations
            top_recommendations = combinations.head(top_n)

            # Display results
            st.subheader("Top Recommendations")

            # Format the results for display
            display_df = top_recommendations[[
                'instance_type', 'num_nodes', 'total_memory',
                'total_vcpu', 'total_cost', 'memory_utilization',
                'cpu_utilization', 'reliability_score', 'final_score'
            ]].copy()

            # Format columns
            display_df['total_memory'] = display_df['total_memory'].round(
                1).astype(str) + ' GiB'
            display_df['total_cost'] = '$' + \
                display_df['total_cost'].round(3).astype(str) + '/hr'
            display_df['memory_utilization'] = display_df['memory_utilization'].round(
                1).astype(str) + '%'
            display_df['cpu_utilization'] = display_df['cpu_utilization'].round(
                1).astype(str) + '%'
            display_df['final_score'] = display_df['final_score'].round(3)
            display_df['reliability_score'] = display_df['reliability_score'].round(
                3)

            st.dataframe(display_df)

            # Visualization
            st.subheader("Cost vs. Performance")
            fig = px.scatter(
                top_recommendations,
                x='total_cost',
                y='final_score',
                size='num_nodes',
                color='reliability_score',
                hover_data=[
                    'instance_type',
                    'memory_utilization',
                    'cpu_utilization'],
                labels={
                    'total_cost': 'Total Cost per Hour ($)',
                    'final_score': 'Final Score',
                    'num_nodes': 'Number of Nodes',
                    'reliability_score': 'Reliability Score'})
            st.plotly_chart(fig)


if __name__ == "__main__":
    main()
