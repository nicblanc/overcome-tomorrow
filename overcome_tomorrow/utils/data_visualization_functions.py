# data_visualization_functions.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_activities_per_month(activity_df):
    """
    Plot the total number of activities per month.

    Parameters:
        activity_df (DataFrame): DataFrame containing activity data with a 'timestamp' column.
    """
    # Convert the 'timestamp' column to datetime format
    activity_df['timestamp'] = pd.to_datetime(activity_df['timestamp'])

    # Extract the month from the 'timestamp' column
    activity_df['month'] = activity_df['timestamp'].dt.to_period('M')

    # Group the data by month
    activities_per_month = activity_df.groupby('month').size()

    # Plot the graph
    plt.figure(figsize=(10, 6))
    activities_per_month.plot(kind='line', color='skyblue', marker='o')

    # Add title and axis labels
    plt.title('Total Activities per Month')
    plt.xlabel('Month')
    plt.ylabel('Number of Activities')

    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha='right')

    # Add gridlines for better

def visualize_sport_distribution(activity_df):
    """
    Visualize the distribution of sports based on the number of activities per sport.

    Parameters:
        activity_df (DataFrame): DataFrame containing activity data with a 'sport' column.
    """
    # Count the number of activities per sport
    sport_counts = activity_df['sport'].value_counts(ascending=False)

    # Visualization of the distribution of sports
    plt.figure(figsize=(10, 6))
    sns.countplot(data=activity_df, x='sport', order=sport_counts.index)
    plt.title('Distribution of Sports (Sorted in Descending Order)')
    plt.xlabel('Sport')
    plt.ylabel("Number of Activities")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

# Example usage:
# Assuming activity_df is your DataFrame containing activity data
# visualize_sport_distribution(activity_df)


def plot_monthly_distance_over_years(activity_df):
    """
    Plot the total distance covered each month over multiple years.

    Parameters:
        activity_df (DataFrame): DataFrame containing activity data with 'timestamp' and 'total_distance' columns.
    """
    # Convert the 'timestamp' column to datetime format and extract month and year
    activity_df['timestamp'] = pd.to_datetime(activity_df['timestamp'])
    activity_df['month'] = activity_df['timestamp'].dt.month
    activity_df['year'] = activity_df['timestamp'].dt.year

    # Calculate the sum of total distance covered for each month over each year
    monthly_distance = activity_df.groupby(['year', 'month'])['total_distance'].sum()

    # Plot the line graph of total distance covered each month over multiple years
    plt.figure(figsize=(12, 6))
    for year in activity_df['year'].unique():
        monthly_distance[year].plot(kind='line', marker='o', label=str(year))

    plt.title('Total Distance Covered Each Month Over Years')
    plt.xlabel('Month')
    plt.ylabel('Total Distance (meters)')
    plt.xticks(range(1, 13), ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
    plt.legend(title='Year')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Example usage:
# Assuming activity_df is your DataFrame containing activity data
# plot_monthly_distance_over_years(activity_df)


def plot_monthly_distance(year, month, activity_df):
    """
    Plot the total distance covered for each day of a specific month of a specific year.
    
    Parameters:
        year (int): The year.
        month (int): The month (1-12).
        activity_df (DataFrame): DataFrame containing activity data.
    """
    # Convert the 'start_time' column to datetime format
    activity_df['start_time'] = pd.to_datetime(activity_df['start_time'])
    
    # Filter the DataFrame for the specified year and month
    monthly_data = activity_df[(activity_df['start_time'].dt.year == year) & (activity_df['start_time'].dt.month == month)]
    
    # Aggregate the data on total distance covered for each day
    aggregated_data = monthly_data.groupby(monthly_data['start_time'].dt.day)['total_distance'].sum()
    
    # Plot the total distance covered for each day of the specified month and year
    plt.figure(figsize=(12, 6))
    plt.plot(aggregated_data.index, aggregated_data.values, marker='o', color='b')
    plt.title(f'Total Distance Covered in {pd.Timestamp(year=year, month=month, day=1).strftime("%B %Y")}')
    plt.xlabel('Day')
    plt.ylabel('Total Distance (meters)')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Example usage:
# plot_monthly_distance(2023, 6, activity_df)


def plot_scatter_average_cadence_vs_speed(activity_df):
    """
    Create a scatter plot of average cadence vs average speed colored by sport.

    Parameters:
        activity_df (DataFrame): DataFrame containing activity data with 'avg_cadence', 'enhanced_avg_speed', and 'sport' columns.
    """
    # Extract columns of interest
    cadence = activity_df['avg_cadence']
    speed = activity_df['enhanced_avg_speed']
    sport = activity_df["sport"]

    # Create a scatter plot
    plt.figure(figsize=(10, 6))

    # Loop through each unique sport to color points based on sport
    sports = activity_df['sport'].unique()
    for s in sports:
        plt.scatter(cadence[activity_df['sport'] == s], 
                    speed[activity_df['sport'] == s], 
                    alpha=0.5, label=s)

    # Add labels and a title
    plt.title('Scatter Plot: Average Cadence vs Average Speed')
    plt.xlabel('Average Cadence')
    plt.ylabel('Average Speed')
    plt.legend()

    # Show the plot
    plt.grid(True)
    plt.show()

# Example usage:
# Assuming activity_df is your DataFrame containing activity data
# plot_scatter_average_cadence_vs_speed(activity_df)

import pandas as pd
import matplotlib.pyplot as plt

def plot_monthly_calorie_and_distance(activity_df):
    """
    Create bar plot for monthly calorie expenditure and line plot for monthly distance,
    with a legend to differentiate between the two.

    Parameters:
        activity_df (DataFrame): DataFrame containing activity data with 'month', 'total_calories', and 'total_distance' columns.
    """
    # Assuming the data is already grouped by month and the total calories are summed up
    # If not, you may need to perform a groupby operation to aggregate the total calories by month
    monthly_calories = activity_df.groupby('month')['total_calories'].sum()

    # Extracting months and total calories
    months = monthly_calories.index
    total_calories = monthly_calories.values

    # Convert total distance from meters to kilometers
    activity_df['total_distance_km'] = activity_df['total_distance'] / 1000

    # Creating a bar plot for total calorie expenditure per month
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.bar(months, total_calories, color='skyblue')
    ax1.set_xlabel('Month')
    ax1.set_ylabel('Total Calories', color='skyblue')
    ax1.tick_params(axis='y', labelcolor='skyblue')
    ax1.set_xticks(months)

    # Plotting the line plot for total distance covered per month on the same figure
    ax2 = ax1.twinx()
    ax2.plot(months, activity_df.groupby('month')['total_distance_km'].sum(), marker='o', color='orange')
    ax2.set_ylabel('Total Distance (km)', color='orange')
    ax2.tick_params(axis='y', labelcolor='orange')

    # Adding legend and title
    plt.title('Monthly Calorie Expenditure and Distance Covered')
    plt.xticks(months, ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])

    # Adding legend
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, ['Calories', 'Distance'], loc='upper left')

    # Display the plot
    plt.grid(True)
    plt.show()

# Example usage:
# Assuming activity_df is your DataFrame containing activity data
# plot_monthly_calorie_and_distance(activity_df)


def plot_average_speed_over_time(activity_df):
    """
    Create a line plot showing the evolution of average speed over time.

    Parameters:
        activity_df (DataFrame): DataFrame containing activity data with 'timestamp' and 'enhanced_avg_speed' columns.
    """
    # Convert the 'timestamp' column to datetime format if not already done
    activity_df['timestamp'] = pd.to_datetime(activity_df['timestamp'])

    # Sort the data by timestamp
    activity_df.sort_values(by='timestamp', inplace=True)

    # Create the line plot
    plt.figure(figsize=(10, 6))
    plt.plot(activity_df['timestamp'], activity_df['enhanced_avg_speed'], color='blue')

    # Add labels and title
    plt.title('Evolution of Average Speed over Time')
    plt.xlabel('Date')
    plt.ylabel('Average Speed')
    plt.grid(True)

    # Display the plot
    plt.show()

# Example usage:
# Assuming activity_df is your DataFrame containing activity data
# plot_average_speed_over_time(activity_df)


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def analyze_last_activities(activity_df, num_days, sport_type):
    """
    Analyze the distance and average heart rate for the last X days of a specific sport type.

    Parameters:
        activity_df (DataFrame): DataFrame containing activity data with 'sport', 'total_distance', 'avg_heart_rate', and 'date' columns.
        num_days (int): Number of last days to consider for the specific sport type.
        sport_type (str): Type of sport to include in the analysis.

    Returns:
        None (displays a barplot)
    """
    # Filter the DataFrame to keep only the last X days' activities for the specific sport type
    last_activities = activity_df[activity_df['sport'] == sport_type].tail(num_days)

    # Convert total distance from meters to kilometers
    last_activities['total_distance_km'] = last_activities['total_distance'] / 1000

    # Create an index range for the last X days
    days_range = np.arange(1, num_days + 1)

    # Plotting the barplot
    plt.figure(figsize=(10, 6))
    sns.barplot(data=last_activities, x=days_range, y='total_distance_km', color='skyblue', label='Total Distance')
    ax2 = plt.twinx()
    sns.lineplot(data=last_activities, x=days_range, y='avg_heart_rate', marker='o', color='orange', ax=ax2, label='Average Heart Rate')

    # Adding labels and title
    plt.title(f'Analysis of Last {num_days} Days of {sport_type.capitalize()} Activities')
    plt.xlabel('Days')
    plt.ylabel('Total Distance (kilometers)', color='skyblue')
    ax2.set_ylabel('Average Heart Rate', color='orange')

    # Displaying the legend
    plt.legend(loc='upper left', bbox_to_anchor=(0.7, 1))

    # Show the plot
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

# Example usage:
# analyze_last_activities(activity_df, 20, "running")