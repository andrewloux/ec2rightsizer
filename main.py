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

        # Extract numeric vCPU value
        self.df['vCPU'] = self.df['vCPUs'].str.extract(r'(\d+)').astype(int)

        # Clean price fields
        self.df['pricePerHour'] = pd.to_numeric(
            self.df['On Demand'].str.extract(r'\$(\d+\.?\d*)')[0],
            errors='coerce'
        )

        # Determine instance type from API Name
        self.df['type'] = self.df['API Name'].apply(self._get_instance_type)

        # Clean up instance type for display
        self.df['instanceType'] = self.df['API Name']

    def _get_instance_type(self, api_name):
        """Determine instance type from API name"""
        if pd.isna(api_name) or not isinstance(api_name, str):
            return 'Unknown'

        try:
            instance_family = api_name.split('.')[0].lower()

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

            if 'metal' in instance_family:
                return 'METAL'

            return type_mapping.get(instance_family[0], 'OTHER')

        except (AttributeError, IndexError):
            return 'Unknown'

    def calculate_reliability_score(self, num_nodes):
        """
        Calculate reliability score based on node count and high-availability patterns

        Reliability scoring logic:
        1 node  = 0.4   (No HA, single point of failure)
        2 nodes = 0.85  (Basic HA with primary/backup)
        3 nodes = 1.0   (Optimal for consensus/quorum based systems)
        4 nodes = 0.95  (Reduced score - doesn't improve quorum, wastes resources)
        5 nodes = 0.98  (Slight improvement in failure tolerance)
        6+ nodes = 0.90 (Diminishing returns, increased coordination overhead)

        This follows common patterns where:
        - Single node has no redundancy
        - Two nodes provide basic failover but can't prevent split-brain
        - Three nodes optimal for consensus (2n+1 for quorum)
        - Four nodes don't improve quorum math over three
        - Five nodes allow for two simultaneous failures
        - Beyond five nodes provides minimal benefit for most use cases

        Example use cases:
        - Web servers behind load balancer: 2+ nodes optimal
        - Database cluster: 3 nodes optimal (primary + 2 secondaries)
        - Kafka/ZK: 3 or 5 nodes optimal for quorum
        """
        reliability_map = {
            1: 0.4,   # No HA
            2: 0.85,  # Basic HA
            3: 1.0,   # Optimal quorum
            4: 0.95,  # Suboptimal - doesn't improve quorum
            5: 0.98,  # Improved failure tolerance
        }
        return reliability_map.get(num_nodes, 0.90)  # 6+ nodes

    def get_optimal_combinations(
            self,
            target_memory,
            target_vcpu,
            max_nodes=5):
        """
        Get optimal node combinations considering resources, cost, and reliability

        Example:
        Required: 32 GiB RAM, 8 vCPU

        Possible combinations:
        1. 1x r6g.2xlarge (32 GiB, 8 vCPU)
           - Cost: $0.8064/hr
           - Reliability: 0.4
           - Utilization: 100% RAM, 100% CPU

        2. 2x r6g.xlarge (16 GiB, 4 vCPU each)
           - Cost: $0.4032/hr * 2 = $0.8064/hr
           - Reliability: 0.85
           - Utilization: 100% RAM, 100% CPU

        3. 3x t3.large (8 GiB, 2 vCPU each)
           - Cost: $0.0832/hr * 3 = $0.2496/hr
           - Reliability: 1.0
           - Utilization: 89% RAM, 89% CPU
        """
        combinations = []

        for instance_idx, instance in self.df.iterrows():
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

    def score_combinations(self, combinations_df, weight_config=None):
        """
        Score combinations based on cost, utilization, and reliability

        Default weights:
        - Cost: 40% (primary driver)
        - Resource Utilization: 35% (efficient use of resources)
        - Reliability: 25% (HA consideration)
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
        # Penalize over-provisioning more than slight under-provisioning
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

    def calculate_mahalanobis_distance(self, target_memory, target_vcpu):
        """
        Calculate Mahalanobis distance to measure how many standard deviations a data point is from the mean
        of our distribution, accounting for correlation between variables.

        The Mahalanobis distance helps us identify optimal instances by considering both memory and vCPU
        requirements simultaneously, while accounting for their relationship.

        Example Covariance Matrix Calculation:
        Consider these instances:
        Instance   Memory(GiB)   vCPU
        t3.small      2          2
        t3.medium     4          2
        t3.large      8          2
        r6g.large     16         2

        Memory mean = (2 + 4 + 8 + 16)/4 = 7.5
        vCPU mean = (2 + 2 + 2 + 2)/4 = 2

        Memory variance = mean((x - mean)²)
            = ((2-7.5)² + (4-7.5)² + (8-7.5)² + (16-7.5)²)/4
            = (30.25 + 12.25 + 0.25 + 72.25)/4
            ≈ 28.75

        vCPU variance = 0 (all instances have 2 vCPUs)

        Covariance(memory,vcpu) = mean((memory - memory_mean)(vcpu - vcpu_mean))
            = 0 (since vCPU doesn't vary)

        Resulting covariance matrix:
        [[ 28.75  0    ]    # This indicates:
         [ 0      0    ]]   # - Memory variance = 28.75
                            # - vCPU variance = 0
                            # - No covariance (0) because vCPU doesn't vary

        A more typical real-world example with varying vCPUs:
        Instance   Memory(GiB)   vCPU
        t3.small      2          2
        t3.medium     4          2
        r6g.large     16         4
        r6g.xlarge    32         8

        Now we'd see non-zero covariance because as memory increases,
        vCPUs tend to increase too.
        """
        try:
            features = self.df[['memoryGiB', 'vCPU']].values
            target = np.array([target_memory, target_vcpu])

            print(f"Features shape: {features.shape}")
            print(f"Target values: {target}")

            # Calculate covariance matrix
            cov_matrix = np.cov(features.T)
            print(f"Covariance matrix:\n{cov_matrix}")

            # Add small regularization term to ensure matrix is invertible
            epsilon = 1e-6
            cov_matrix = cov_matrix + np.eye(2) * epsilon

            # Inverse of covariance matrix
            inv_covmat = np.linalg.inv(cov_matrix)
            print(f"Inverse covariance matrix:\n{inv_covmat}")

            # Calculate difference between each point and target
            diff = features - target

            # Calculate Mahalanobis distance
            distances = np.sqrt(
                np.sum(
                    np.dot(
                        diff,
                        inv_covmat) *
                    diff,
                    axis=1))

            print(f"Distance range: {distances.min()} to {distances.max()}")
            return distances

        except Exception as e:
            print(f"Error in Mahalanobis calculation: {str(e)}")
            return np.zeros(len(self.df))

    def calculate_efficiency_score(
            self,
            memory_weight=0.5,
            target_memory=None,
            target_vcpu=None):
        """
        Calculate efficiency score using statistical methods to provide more robust scoring.

        We use chi-square distribution because Mahalanobis distance follows chi-square
        distribution with degrees of freedom equal to number of variables (2 in our case:
        memory and vCPU).

        Args:
            memory_weight (float): Weight for memory importance (0-1)
            target_memory (float): Required memory in GiB
            target_vcpu (int): Required vCPUs

        Returns:
            float: Final efficiency score (0-1)
        """
        try:
            if target_memory is None or target_vcpu is None:
                return 0

            print(
                f"\nCalculating efficiency score for target: {target_memory}GB, {target_vcpu} vCPUs")

            # Calculate Mahalanobis distances
            mahalanobis_dist = self.calculate_mahalanobis_distance(
                target_memory, target_vcpu)

            # Convert distances to probability scores
            probability_scores = 1 - chi2.cdf(mahalanobis_dist, df=2)
            print(
                f"Probability scores range: {
                    probability_scores.min():.4f} to {
                    probability_scores.max():.4f}")

            # Transform prices using log
            cost_efficiency = -np.log(self.df['pricePerHour'] + 1e-10)
            cost_efficiency = (cost_efficiency - cost_efficiency.min()) / \
                (cost_efficiency.max() - cost_efficiency.min())
            print(
                f"Cost efficiency range: {
                    cost_efficiency.min():.4f} to {
                    cost_efficiency.max():.4f}")

            # Calculate utilization ratios
            memory_util = target_memory / self.df['memoryGiB']
            vcpu_util = target_vcpu / self.df['vCPU']

            def calculate_utilization_penalty(util):
                return 1 - np.abs(1 - util)

            # Combine utilization scores
            utilization_score = (
                calculate_utilization_penalty(memory_util) * memory_weight +
                calculate_utilization_penalty(vcpu_util) * (1 - memory_weight)
            )
            print(
                f"Utilization score range: {
                    utilization_score.min():.4f} to {
                    utilization_score.max():.4f}")

            # Calculate final score
            final_score = (
                probability_scores ** 0.4 *     # Statistical fit
                cost_efficiency ** 0.3 *        # Cost efficiency
                utilization_score ** 0.3        # Resource utilization
            )

            print(
                f"Final score range: {
                    final_score.min():.4f} to {
                    final_score.max():.4f}")

            return final_score

        except Exception as e:
            print(f"Error in efficiency score calculation: {str(e)}")
            return 0

    def get_best_instances(
            self,
            memory_weight=0.5,
            instance_types=None,
            family_prefix=None,
            required_memory=None,
            required_vcpus=None,
            max_instances=None,
            top_n=5):
        """
        Get best instances based purely on scoring without filtering
        """
        print("\nStarting instance selection...")
        print(
            f"Requirements: {required_memory}GB memory, {required_vcpus} vCPUs")

        # Create a working copy
        working_df = self.df.copy()
        print(f"Initial instances: {len(working_df)}")

        # Filter out instances with unavailable pricing
        working_df = working_df[working_df['pricePerHour'].notna()]
        print(f"After price filtering: {len(working_df)}")

        # Apply type filtering if specified
        if instance_types:
            working_df = working_df[working_df['type'].isin(instance_types)]
            print(f"After type filtering: {len(working_df)}")

        # Apply family prefix filtering if specified
        if family_prefix:
            working_df = working_df[working_df['API Name'].str.startswith(
                family_prefix + '.', na=False)]
            print(f"After family prefix filtering: {len(working_df)}")

        if working_df.empty:
            print("No instances available after filtering")
            return pd.DataFrame()

        # Calculate efficiency score
        working_df['efficiency_score'] = self.calculate_efficiency_score(
            memory_weight=memory_weight,
            target_memory=required_memory,
            target_vcpu=required_vcpus
        )

        # Print some debugging info
        print(
            f"\nInstances with non-zero scores: {(working_df['efficiency_score'] > 0).sum()}")
        print(
            f"Score range: {
                working_df['efficiency_score'].min():.4f} to {
                working_df['efficiency_score'].max():.4f}")

        # Add useful metrics for display
        working_df['memory_ratio'] = working_df['memoryGiB'] / \
            required_memory if required_memory else 1
        working_df['vcpu_ratio'] = working_df['vCPU'] / \
            required_vcpus if required_vcpus else 1
        working_df['mem_per_vcpu'] = working_df['memoryGiB'] / \
            working_df['vCPU']
        working_df['price_per_vcpu'] = working_df['pricePerHour'] / \
            working_df['vCPU']
        working_df['price_per_gib'] = working_df['pricePerHour'] / \
            working_df['memoryGiB']

        # Get top results
        result = working_df.nlargest(top_n, 'efficiency_score')

        # Print top results for debugging
        print("\nTop instances found:")
        for _, row in result.iterrows():
            print(f"{row['API Name']}: score={row['efficiency_score']:.3f}, "
                  f"mem={row['memoryGiB']}GB, vCPU={row['vCPU']}, "
                  f"price=${row['pricePerHour']}/hr")

        return result

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

            if combinations.empty:
                st.warning(
                    "No valid configurations found. Try adjusting your requirements.")
                return

            # Score combinations
            combinations['final_score'] = selector.score_combinations(
                combinations,
                weight_config=weight_config
            )

            # Get top recommendations
            top_recommendations = combinations.nlargest(top_n, 'final_score')

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
