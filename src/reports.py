import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import shap

# Load the dataset
# Use the absolute path if necessary
df = pd.read_csv('derived_risk_attributes_with_model_risk_score.csv')

# Function to create a correlation heatmap
def plot_correlation_heatmap(df):
    # Select only numeric columns for correlation
    numeric_df = df.select_dtypes(include=[float, int])
    
    plt.figure(figsize=(10, 8))
    corr = numeric_df.corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
    plt.title('Correlation Heatmap')
    plt.savefig('correlation_heatmap.png')
    plt.show()

# Function to create box plots for different features by fraud label
def plot_box_plots_by_label(df, features, label_column='Label'):
    for feature in features:
        plt.figure(figsize=(8, 6))
        sns.boxplot(x=label_column, y=feature, data=df)
        plt.title(f'{feature} Distribution by Fraud Label')
        plt.savefig(f'{feature}_boxplot_by_label.png')
        plt.show()

# Function to create a pairplot for selected features
def plot_pairplot(df, features, label_column='Label'):
    sns.pairplot(df[features + [label_column]], hue=label_column, diag_kind='kde')
    plt.savefig('pairplot.png')
    plt.show()

# Function to plot feature importance from a RandomForestClassifier
def plot_feature_importance(df, features, label_column='Label'):
    X = df[features]
    y = df[label_column]
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale the features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Train the model
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    
    # Feature importance
    feature_importances = pd.Series(rf_model.feature_importances_, index=features)
    feature_importances.sort_values(ascending=False).plot(kind='bar', figsize=(10, 6))
    plt.title('Feature Importance')
    plt.ylabel('Importance Score')
    plt.xlabel('Features')
    plt.savefig('feature_importance.png')
    plt.show()

# Function to generate SHAP summary plot
def plot_shap_summary(df, features, label_column='Label'):
    X = df[features]
    y = df[label_column]
    
    # Train a RandomForest model and use SHAP for explanation
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X, y)
    
    explainer = shap.TreeExplainer(rf_model)
    shap_values = explainer.shap_values(X)
    
    shap.summary_plot(shap_values, X, feature_names=features)
    plt.savefig('shap_summary_plot.png')
    plt.show()

# Function to create KDE plots for fraud vs non-fraud for selected features
def plot_kde_by_label(df, features, label_column='Label'):
    for feature in features:
        plt.figure(figsize=(8, 6))
        sns.kdeplot(df[df[label_column] == 1][feature], label='Fraud', shade=True)
        sns.kdeplot(df[df[label_column] == 0][feature], label='Non-Fraud', shade=True)
        plt.title(f'KDE Plot of {feature} by Fraud Label')
        plt.xlabel(feature)
        plt.ylabel('Density')
        plt.savefig(f'{feature}_kde_by_label.png')
        plt.show()

# List of features to analyze (adjust as necessary)
selected_features = [
    'SSN_24hr_Velocity',
    'SSN_72hr_Velocity',
    'Phone_24hr_Velocity',
    'Phone_72hr_Velocity',
    'Email_Gibberish_Score',
    'KYC_Score'
]

# Generate the reports
plot_correlation_heatmap(df)
plot_box_plots_by_label(df, selected_features)
plot_pairplot(df, selected_features)
plot_feature_importance(df, selected_features)
plot_shap_summary(df, selected_features)
plot_kde_by_label(df, selected_features)
