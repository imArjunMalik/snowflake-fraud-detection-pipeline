import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from graphviz import Digraph

# Load the dataset
df = pd.read_csv('synthetic_account_opening_data.csv')

# Filter out successful transactions
df_successful = df[df['Attempt_Status'] == 'Successful']

# Count the number of attempts for each SSN to measure how many attempts were made for account creation
df_attempt_counts = df.groupby('SSN').size().reset_index(name='Attempts')

# Merge this back with the original dataframe to get attempts per transaction
df_successful = pd.merge(df_successful, df_attempt_counts, on='SSN')

# Split the dataset into benign and fraudulent successful transactions
benign_attempts = df_successful[df_successful['Label'] == 0]['Attempts']
fraudulent_attempts = df_successful[df_successful['Label'] == 1]['Attempts']

# Define the bins with equal width: 0-5, 6-10, 11-15, 16-20, and 20+
bins = [0, 5, 10, 15, 20, float('inf')]
bin_labels = ['0-5', '6-10', '11-15', '16-20', '20+']

# Plot histogram for benign attempts
plt.figure(figsize=(10, 6))
plt.hist(benign_attempts, bins=bins, edgecolor='black', label='Benign', color='blue', alpha=0.7)
plt.title('Number of Attempts for Benign Transactions', fontsize=16)
plt.xlabel('Number of Attempts', fontsize=14)
plt.ylabel('Frequency', fontsize=14)
plt.xticks([2.5, 7.5, 12.5, 17.5, 25], bin_labels)  # Adjust the x-ticks to match the new bin ranges
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()

# Show the first histogram in a separate window
plt.show()

# Plot histogram for fraudulent attempts
plt.figure(figsize=(10, 6))
plt.hist(fraudulent_attempts, bins=bins, edgecolor='black', label='Fraudulent', color='red', alpha=0.7)
plt.title('Number of Attempts for Fraudulent Transactions', fontsize=16)
plt.xlabel('Number of Attempts', fontsize=14)
plt.ylabel('Frequency', fontsize=14)
plt.xticks([2.5, 7.5, 12.5, 17.5, 25], bin_labels)  # Adjust the x-ticks to match the new bin ranges
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()

# Show the second histogram in a separate window
plt.show()


############################# PIE CHART ######################################

# Count the number of benign and fraudulent transactions
benign_count = len(df_successful[df_successful['Label'] == 0])
fraudulent_count = len(df_successful[df_successful['Label'] == 1])

# Define the data for the pie chart
labels = ['Benign', 'Fraudulent']
sizes = [benign_count, fraudulent_count]
colors = ['#4CAF50', '#FF6347']  # Green for benign, red for fraudulent

# Create the pie chart with a smaller figure size
fig, ax = plt.subplots(figsize=(6, 6))  # Adjust figure size to fit properly in the window

# Create the pie chart with equal aspect ratio (to ensure a circle)
patches, texts, autotexts = ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, colors=colors)

# Set a title
ax.set_title('Percentage of Benign vs Fraudulent Successful Openings', fontsize=14)

# Adding a legend (key)
plt.legend(patches, labels, loc="best", title="Transaction Type", fontsize=10)

# Ensure the pie chart is a perfect circle
plt.axis('equal')

# Save the pie chart as an image
#plt.savefig('benign_vs_fraudulent_pie_chart.png', bbox_inches='tight', dpi=300)

# Display the pie chart
plt.tight_layout()
plt.show()

################################ IN REVIEW ####################################

# Create a histogram for Model_Risk_Score in specified bins for 'In Review' transactions

# Load the dataset with model risk scores
df_in_review = pd.read_csv('in_review_transactions_with_risk_scores.csv')

# Define bins and labels
bins = [0, 10, 30, 50, 70, 90, 100]
bin_labels = ['0-10', '10-30', '30-50', '50-70', '70-90', '90-100']

# Plot histogram of Model_Risk_Score
plt.figure(figsize=(10, 6))
plt.hist(df_in_review['Model_Risk_Score'], bins=bins, edgecolor='black', color='purple', alpha=0.7)

# Add labels and title
plt.title('Model Risk Scores for In Review Transactions', fontsize=16)
plt.xlabel('Model Risk Score', fontsize=14)
plt.ylabel('Count of Transactions', fontsize=14)
plt.xticks([5, 20, 40, 60, 80, 95], bin_labels)  # Adjust x-ticks to match the bin ranges

# Display grid for clarity
plt.grid(True, linestyle='--', alpha=0.7)

# Adjust layout for better spacing
plt.tight_layout()

# Show the histogram
plt.show()


above_95 = df_in_review[df_in_review['Model_Risk_Score'] >= 95].shape[0]
below_95 = df_in_review[df_in_review['Model_Risk_Score'] < 95].shape[0]

# Data for the pie chart
labels = ['Risk Score >= 95', 'Risk Score < 95']
sizes = [above_95, below_95]
colors = ['#ff9999','#66b3ff']  # Custom colors for the pie chart

# Generate the pie chart
plt.figure(figsize=(6, 6))
plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90, wedgeprops={'edgecolor': 'black'})

# Equal aspect ratio ensures the pie is drawn as a circle
plt.axis('equal')

# Title of the pie chart
plt.title('Percentage of In-Review Transactions by Model Risk Score')

# Show the pie chart
plt.show()

################################ COST SAVINGS ##################################

# Assumptions
# agents = 50
# hours_per_year_per_agent = 2000
# hourly_wage = 30
# reviews_per_agent_per_year = 1000
# manual_reviews_reduction_percentage = 0.5  # 50% fewer reviews
# fraud_catch_rate_with_model = 0.9
# fraud_catch_rate_without_model = 0.7

# Parameters for calculation
annual_labor_cost_without_model = 1_000_000  # Assuming $1 million in agent labor costs per year
reduction_percentage = 0.28  # 28% reduction in the number of transactions needing manual review
reduced_costs = annual_labor_cost_without_model * (1 - reduction_percentage)  # Reduced costs per year
inflation_rate = 0.029  # 2.9% inflation per year

# Years for plotting
years = np.arange(1, 6)

# Apply inflation rate to calculate original costs over 5 years
original_costs = [annual_labor_cost_without_model * (1 + inflation_rate) ** (i - 1) for i in years]

# Reduced costs remain the same since we assume the percentage reduction is constant, but apply inflation too
reduced_costs_over_time = [reduced_costs * (1 + inflation_rate) ** (i - 1) for i in years]

# Plotting the original costs (red line) and reduced costs (green line)
plt.figure(figsize=(10, 6))

# Plot the original costs with inflation (red line)
plt.plot(years, original_costs, marker='o', linestyle='-', color='r', label="Original Costs (No Model, With Inflation)", linewidth=2)

# Plot the reduced costs (green line)
plt.plot(years, reduced_costs_over_time, marker='o', linestyle='-', color='g', label="Reduced Costs (With Model)", linewidth=2)

# Adding labels and title
plt.title("Cost Comparison: Original vs Reduced Costs with Model", fontsize=16)
plt.xlabel("Years", fontsize=12)
plt.ylabel("Amount (in USD)", fontsize=12)
plt.grid(True)

# Adjusting y-limits to bring the two lines closer
plt.ylim(0.6 * 10**6, 1.2 * 10**6)

# Annotating the points for each line with adjusted positioning
for i, value in enumerate(original_costs):
    plt.text(years[i], original_costs[i] + 20_000, f"${value/1_000_000:.1f}M", ha='center', color='r', fontsize=10)
for i, value in enumerate(reduced_costs_over_time):
    plt.text(years[i], reduced_costs_over_time[i] - 40_000, f"${value/1_000_000:.1f}M", ha='center', color='g', fontsize=10)

# Show legend
plt.legend(loc="upper left", fontsize=12)

# Show the plot
plt.tight_layout()
plt.show()

######################## CUMULATIVE SAVINGS ##################################

# Data points for cumulative savings over 5 years
years = [1, 2, 3, 4, 5]
cumulative_savings = [0.3, 0.6, 0.8, 1.1, 1.4]  # In millions

# Plotting the cumulative savings
plt.figure(figsize=(10, 6))
plt.plot(years, cumulative_savings, marker='o', linestyle='-', color='blue', label="Cumulative Savings")
plt.fill_between(years, cumulative_savings, color='lightblue', alpha=0.5)

# Adding labels and title
for i, saving in enumerate(cumulative_savings):
    plt.text(years[i], saving, f"${saving:.1f}M", ha='center', fontsize=10)

plt.title("Cumulative Savings Over Time by Implementing Fraud Detection Model", fontsize=14)
plt.xlabel("Years", fontsize=12)
plt.ylabel("Cumulative Savings (in USD)", fontsize=12)
plt.grid(True)

# Show the cumulative savings plot
plt.tight_layout()
plt.show()


############################# FLOW CHART ######################################

# Create a new Digraph object
flow_chart = Digraph("Transaction Workflow", format="png")

# Manual Review (Pre-Model)
flow_chart.node("A", "Transactions Enter System")
flow_chart.node("B", "All Transactions Sent for Manual Review")
flow_chart.node("C", "Manual Agent Review")
flow_chart.node("D", "Accepted")
flow_chart.node("E", "Rejected")

# Manual Review workflow (Pre-Model)
flow_chart.edge("A", "B")
flow_chart.edge("B", "C")
flow_chart.edge("C", "D", label="Low Risk")
flow_chart.edge("C", "E", label="High Risk")

# Automated Workflow (Post-Model)
flow_chart.node("F", "Transactions Enter System")
flow_chart.node("G", "Transactions Sent to Fraud Model for Risk Scoring")
flow_chart.node("H", "Model Auto-Rejects High-Risk Transactions")
flow_chart.node("I", "Manual Agent Review for Remaining")
flow_chart.node("J", "Accepted (After Review)")
flow_chart.node("K", "Rejected (After Review)")

# Post-Model workflow
flow_chart.edge("F", "G")
flow_chart.edge("G", "H", label="Risk Score >= 95")
flow_chart.edge("G", "I", label="Risk Score < 95")
flow_chart.edge("I", "J", label="Low Risk")
flow_chart.edge("I", "K", label="High Risk")

# Save and render the diagram
flow_chart.render('workflow_comparison_diagram', view=True)


######################### STACKED BAR CHART ####################################

low_risk = df_in_review[df_in_review['Model_Risk_Score'] < 50]
medium_risk = df_in_review[(df_in_review['Model_Risk_Score'] >= 50) & (df_in_review['Model_Risk_Score'] < 75)]
high_risk = df_in_review[df_in_review['Model_Risk_Score'] >= 75]

# Step 3: Estimate time spent (in minutes) per transaction in each risk category
time_spent_low = len(low_risk) * 5  # 5 minutes for low-risk transactions
time_spent_medium = len(medium_risk) * 10  # 10 minutes for medium-risk transactions
time_spent_high = len(high_risk) * 20  # 20 minutes for high-risk transactions

# Step 4: Create a stacked bar chart
categories = ['Low Risk', 'Medium Risk', 'High Risk']
time_spent = [time_spent_low, time_spent_medium, time_spent_high]

# Plot the stacked bar chart
fig, ax = plt.subplots()
ax.bar('In Review Transactions', time_spent_low, label='Low Risk', color='#8BC34A')
ax.bar('In Review Transactions', time_spent_medium, bottom=time_spent_low, label='Medium Risk', color='#FFEB3B')
ax.bar('In Review Transactions', time_spent_high, bottom=[time_spent_low + time_spent_medium], label='High Risk', color='#F44336')

# Add labels, title, and legend
ax.set_ylabel('Total Time Spent (in Minutes)')
ax.set_title('Time Spent per Transaction Category for In Review Transactions')
ax.legend()

# Display the chart
plt.show()