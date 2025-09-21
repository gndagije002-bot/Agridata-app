import streamlit as st
import pandas as pd
import numpy as np

import matplotlib
matplotlib.use('Agg')  
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import plotly.express as px
import plotly.graph_objects as go

plt.style.use('default')
sns.set_palette("husl")

st.set_page_config(
    page_title="Lebanon Agriculture Analytics",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #2E8B57;
        margin: 0.5rem 0;
    }
    .insight-box {
        background-color: #e8f5e8;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #ffc107;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    try:
        try:
            data = pd.read_csv("agriculture-and-rural-development_lbn.csv")
        except FileNotFoundError:
            data_path = r"agriculture-and-rural-development_lbn.csv"
            data = pd.read_csv(data_path)
    except FileNotFoundError:
        st.error("⚠️ CSV file not found. Using sample data for demonstration.")
        years = list(range(1960, 2021))
        np.random.seed(42)
        base_values = np.random.uniform(80, 200, len(years))
        outlier_indices = np.random.choice(len(years), size=3, replace=False)
        base_values[outlier_indices] = np.random.uniform(400, 800, 3)
        
        sample_data = {
            'Year': years,
            'Value': base_values,
            'Indicator Name': ['Fertilizer consumption (% of fertilizer production)'] * len(years),
            'Country Name': ['Lebanon'] * len(years),
            'Country Code': ['LBN'] * len(years)
        }
        data = pd.DataFrame(sample_data)
    
    data["Value"] = pd.to_numeric(data["Value"], errors="coerce")
    data = data.dropna(subset=["Value"])
    return data

data = load_data()

try:
    fertilizer_data = data[data["Indicator Name"].str.contains("Fertilizer", case=False, na=False)]
    if fertilizer_data.empty:
        fertilizer_data = data
except:
    fertilizer_data = data

fertilizer_data["Year"] = pd.to_numeric(fertilizer_data["Year"], errors="coerce")
fertilizer_data = fertilizer_data.dropna(subset=["Year"])
fertilizer_data["Year"] = fertilizer_data["Year"].astype(int)
fertilizer_data = fertilizer_data.sort_values("Year")

st.title("Trend in Agriculture - Lebanon")
st.subheader("Fertilizer Consumption Analysis (1960-2020)")

tab1, tab2, tab3, tab4 = st.tabs(["📊 Interactive Analysis", "🔍 Detailed Statistics", "📈 Advanced Analytics", "📋 Data Explorer"])

with tab1:
    # Sidebar controls
    st.sidebar.header("🎛️ Interactive Controls")
    st.sidebar.markdown("---")

    if not fertilizer_data.empty:
        min_year = int(fertilizer_data["Year"].min())
        max_year = int(fertilizer_data["Year"].max())
        
        year_range = st.sidebar.slider(
            "📅 Select Year Range for Analysis",
            min_value=min_year,
            max_value=max_year,
            value=(min_year, max_year),
            help="Drag to filter data by specific year range"
        )
        
        viz_type = st.sidebar.selectbox(
            "📊 Choose Primary Visualization Type",
            ["Line Chart", "Area Chart", "Histogram", "Box Plot"],
            help="Select different chart types to explore data perspectives"
        )
        
        outlier_handling = st.sidebar.radio(
            "🎯 Outlier Treatment",
            ["Include All Data", "Remove Extreme Outliers", "Highlight Outliers"],
            help="Choose how to handle extreme values in the analysis"
        )
        
        st.sidebar.markdown("### 📊 Analysis Options")
        show_trend = st.sidebar.checkbox("Show Trend Line", value=True)
        show_confidence = st.sidebar.checkbox("Show Confidence Interval", value=False)
        show_moving_avg = st.sidebar.checkbox("Show Moving Average", value=False)
        
        if show_moving_avg:
            window_size = st.sidebar.slider("Moving Average Window", 3, 10, 5)
        
        filtered_data = fertilizer_data[
            (fertilizer_data["Year"] >= year_range[0]) & 
            (fertilizer_data["Year"] <= year_range[1])
        ].copy()
        
        outliers = pd.DataFrame()
        if outlier_handling == "Remove Extreme Outliers" and len(filtered_data) > 5:
            Q1 = filtered_data['Value'].quantile(0.25)
            Q3 = filtered_data['Value'].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 2.5 * IQR
            upper_bound = Q3 + 2.5 * IQR
            original_count = len(filtered_data)
            outliers = filtered_data[(filtered_data['Value'] < lower_bound) | 
                                   (filtered_data['Value'] > upper_bound)]
            filtered_data = filtered_data[(filtered_data['Value'] >= lower_bound) & 
                                        (filtered_data['Value'] <= upper_bound)]
            removed_count = original_count - len(filtered_data)
            if removed_count > 0:
                st.sidebar.info(f"Removed {removed_count} outlier(s)")

        st.sidebar.markdown("### 📈 Quick Stats")
        if not filtered_data.empty:
            st.sidebar.write(f"**Years in Range:** {year_range[1] - year_range[0] + 1}")
            st.sidebar.write(f"**Data Points:** {len(filtered_data)}")
            st.sidebar.write(f"**Average Consumption:** {filtered_data['Value'].mean():.2f}%")
            st.sidebar.write(f"**Peak Consumption:** {filtered_data['Value'].max():.2f}%")
            
            if len(filtered_data) > 2:
                slope, intercept, r_value, p_value, std_err = stats.linregress(filtered_data['Year'], filtered_data['Value'])
                trend_direction = "↗️ Upward" if slope > 0 else "↘️ Downward"
                st.sidebar.write(f"**Trend Direction:** {trend_direction}")
                st.sidebar.write(f"**R² Value:** {r_value**2:.3f}")

        if not filtered_data.empty:
            col1, col2 = st.columns([2.5, 1.5])
            
            with col1:
                st.markdown(f"### 📊 Primary Analysis: {viz_type}")
                
                try:
                    if viz_type == "Line Chart":
                        fig, ax = plt.subplots(figsize=(12, 6))
                        
                        ax.plot(filtered_data['Year'], filtered_data['Value'], 
                               marker='o', linewidth=2, markersize=4, color='#2E8B57',
                               label='Fertilizer Consumption')
                        
                        if show_trend and len(filtered_data) > 2:
                            z = np.polyfit(filtered_data['Year'], filtered_data['Value'], 1)
                            p = np.poly1d(z)
                            ax.plot(filtered_data['Year'], p(filtered_data['Year']), 
                                   "--", alpha=0.8, color='red', linewidth=2, label='Trend Line')
                        
                        if show_moving_avg and len(filtered_data) >= window_size:
                            moving_avg = filtered_data['Value'].rolling(window=window_size, center=True).mean()
                            ax.plot(filtered_data['Year'], moving_avg, 
                                   color='orange', linewidth=2, alpha=0.7, 
                                   label=f'{window_size}-Year Moving Average')
                        
                        if outlier_handling == "Highlight Outliers":
                            Q1 = filtered_data['Value'].quantile(0.25)
                            Q3 = filtered_data['Value'].quantile(0.75)
                            IQR = Q3 - Q1
                            outliers = filtered_data[(filtered_data['Value'] < Q1 - 1.5*IQR) | 
                                                    (filtered_data['Value'] > Q3 + 1.5*IQR)]
                            if not outliers.empty:
                                ax.scatter(outliers['Year'], outliers['Value'], 
                                         color='red', s=100, alpha=0.7, label='Outliers', marker='x')
                        
                        ax.set_xlabel('Year', fontsize=12)
                        ax.set_ylabel('Consumption (% of production)', fontsize=12)
                        ax.set_title('Fertilizer Consumption Trend Over Time', fontsize=14, fontweight='bold')
                        ax.grid(True, alpha=0.3)
                        
                        handles, labels = ax.get_legend_handles_labels()
                        if len(handles) > 1:
                            ax.legend(loc='best')
                        
                        plt.tight_layout()
                        st.pyplot(fig)
                        plt.close(fig)
                        
                    elif viz_type == "Area Chart":
                        fig, ax = plt.subplots(figsize=(12, 6))
                        
                        ax.fill_between(filtered_data['Year'], filtered_data['Value'], 
                                       alpha=0.6, color='#2E8B57', label='Fertilizer Consumption')
                        ax.plot(filtered_data['Year'], filtered_data['Value'], 
                               color='#1B5E20', linewidth=2)
                        
                        ax.set_xlabel('Year', fontsize=12)
                        ax.set_ylabel('Consumption (% of production)', fontsize=12)
                        ax.set_title('Fertilizer Consumption Area Chart', fontsize=14, fontweight='bold')
                        ax.grid(True, alpha=0.3)
                        
                        plt.tight_layout()
                        st.pyplot(fig)
                        plt.close(fig)
                        
                    elif viz_type == "Box Plot":
                        fig, ax = plt.subplots(figsize=(10, 6))
                        
                        filtered_data_copy = filtered_data.copy()
                        filtered_data_copy['Decade'] = (filtered_data_copy['Year'] // 10) * 10
                        decades = sorted(filtered_data_copy['Decade'].unique())
                        
                        if len(decades) > 1:
                            decade_data = [filtered_data_copy[filtered_data_copy['Decade'] == decade]['Value'].values 
                                         for decade in decades]
                            decade_labels = [f"{int(decade)}s" for decade in decades]
                            
                            box_plot = ax.boxplot(decade_data, labels=decade_labels, patch_artist=True)
                            for patch in box_plot['boxes']:
                                patch.set_facecolor('#2E8B57')
                                patch.set_alpha(0.7)
                            
                            ax.set_xlabel('Decade', fontsize=12)
                            ax.set_ylabel('Consumption (% of production)', fontsize=12)
                            ax.set_title('Distribution by Decade', fontsize=14, fontweight='bold')
                        else:
                            box_plot = ax.boxplot(filtered_data['Value'], patch_artist=True)
                            for patch in box_plot['boxes']:
                                patch.set_facecolor('#2E8B57')
                                patch.set_alpha(0.7)
                            ax.set_ylabel('Consumption (% of production)', fontsize=12)
                            ax.set_title('Overall Distribution', fontsize=14, fontweight='bold')
                            ax.set_xticklabels(['All Data'])
                            
                        ax.grid(True, alpha=0.3)
                        plt.tight_layout()
                        st.pyplot(fig)
                        plt.close(fig)
                        
                    else:  # Histogram
                        fig, ax = plt.subplots(figsize=(10, 6))
                        
                        n, bins, patches = ax.hist(filtered_data['Value'], bins=15, alpha=0.7, 
                                                  color='skyblue', edgecolor='black', linewidth=0.5)
                        
                        q10 = filtered_data['Value'].quantile(0.1)
                        q90 = filtered_data['Value'].quantile(0.9)
                        
                        for i, (patch, bin_start, bin_end) in enumerate(zip(patches, bins[:-1], bins[1:])):
                            bin_center = (bin_start + bin_end) / 2
                            if bin_center > q90:
                                patch.set_facecolor('red')
                                patch.set_alpha(0.8)
                            elif bin_center < q10:
                                patch.set_facecolor('orange')
                                patch.set_alpha(0.8)
                        
                        mean_val = filtered_data['Value'].mean()
                        median_val = filtered_data['Value'].median()
                        
                        ax.axvline(mean_val, color='red', linestyle='--', 
                                 linewidth=2, label=f'Mean: {mean_val:.1f}%')
                        ax.axvline(median_val, color='green', linestyle='--', 
                                 linewidth=2, label=f'Median: {median_val:.1f}%')
                        
                        ax.set_xlabel('Consumption (%)', fontsize=12)
                        ax.set_ylabel('Frequency', fontsize=12)
                        ax.set_title('Distribution of Consumption Values', fontsize=14, fontweight='bold')
                        ax.legend()
                        ax.grid(True, alpha=0.3)
                        
                        plt.tight_layout()
                        st.pyplot(fig)
                        plt.close(fig)
                        
                except Exception as e:
                    st.error(f"Error creating visualization: {str(e)}")
                    st.info("Please try a different visualization type or check your data.")
            
            with col2:
                st.markdown("### 📊 Enhanced Data Summary")
                
                if not filtered_data.empty:
                    col_a, col_b = st.columns(2)
                    
                    with col_a:
                        st.write(f"**📊 Mean:** {filtered_data['Value'].mean():.2f}%")
                        st.write(f"**📈 Maximum:** {filtered_data['Value'].max():.2f}%")
                        st.write(f"**🎯 Median:** {filtered_data['Value'].median():.2f}%")
                    
                    with col_b:
                        st.write(f"**📉 Minimum:** {filtered_data['Value'].min():.2f}%")
                        st.write(f"**📐 Std Dev:** {filtered_data['Value'].std():.2f}%")
                        st.write(f"**📏 Range:** {filtered_data['Value'].max() - filtered_data['Value'].min():.2f}%")
                    
                    st.markdown("### 🔬 Advanced Metrics")
                    skewness = stats.skew(filtered_data['Value'])
                    kurtosis = stats.kurtosis(filtered_data['Value'])
                    cv = (filtered_data['Value'].std() / filtered_data['Value'].mean()) * 100
                    
                    st.write(f"**📊 Skewness:** {skewness:.3f}")
                    st.write(f"**📊 Kurtosis:** {kurtosis:.3f}")
                    st.write(f"**📊 Coeff. of Variation:** {cv:.1f}%")
                    
                    if len(filtered_data) > 1:
                        trend = "📈 Increasing" if filtered_data["Value"].iloc[-1] > filtered_data["Value"].iloc[0] else "📉 Decreasing"
                        change_pct = ((filtered_data["Value"].iloc[-1] - filtered_data["Value"].iloc[0]) / 
                                    filtered_data["Value"].iloc[0] * 100)
                        st.success(f"{trend} ({change_pct:+.1f}%)")

with tab2:
    st.markdown("## 🔍 Detailed Statistical Analysis")
    
    if not fertilizer_data.empty and not filtered_data.empty:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 📊 Descriptive Statistics")
            desc_stats = filtered_data['Value'].describe()
            st.dataframe(desc_stats.round(2))
            
            st.markdown("### 📊 Quartile Analysis")
            quartiles = {
                'Mean': filtered_data['Value'].mean(),
                'Q1 (25th percentile)': filtered_data['Value'].quantile(0.25),
                'Q2 (Median)': filtered_data['Value'].quantile(0.5),
                'Q3 (75th percentile)': filtered_data['Value'].quantile(0.75),
                'IQR': filtered_data['Value'].quantile(0.75) - filtered_data['Value'].quantile(0.25)
            }
            for key, value in quartiles.items():
                st.write(f"**{key}:** {value:.2f}%")
        
        with col2:
            st.markdown("### 📈 Time Series Analysis")
            
            if len(filtered_data) > 10:
                filtered_data_sorted = filtered_data.sort_values('Year')
                filtered_data_sorted['YoY_Change'] = filtered_data_sorted['Value'].pct_change() * 100
                
                avg_growth = filtered_data_sorted['YoY_Change'].mean()
                max_growth = filtered_data_sorted['YoY_Change'].max()
                min_growth = filtered_data_sorted['YoY_Change'].min()
                
                st.write(f"**Average YoY Change:** {avg_growth:.2f}%")
                st.write(f"**Max YoY Growth:** {max_growth:.2f}%")
                st.write(f"**Max YoY Decline:** {min_growth:.2f}%")
                
                high_volatility_years = filtered_data_sorted[
                    abs(filtered_data_sorted['YoY_Change']) > filtered_data_sorted['YoY_Change'].std()
                ]['Year'].tolist()
                
                if high_volatility_years:
                    st.markdown("**High Volatility Years:**")
                    st.write(", ".join(map(str, high_volatility_years[:5])))

with tab3:
    st.markdown("## 📈 Advanced Analytics & Insights")
    
    if not fertilizer_data.empty and len(filtered_data) > 5:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 🔬 Statistical Tests")
            
            if len(filtered_data) >= 3:
                try:
                    shapiro_stat, shapiro_p = stats.shapiro(filtered_data['Value'])
                    st.markdown(f"**Shapiro-Wilk Normality Test:**")
                    st.write(f"Statistic: {shapiro_stat:.4f}, p-value: {shapiro_p:.4f}")
                    if shapiro_p > 0.05:
                        st.success("Data appears normally distributed")
                    else:
                        st.warning("Data does not appear normally distributed")
                except Exception as e:
                    st.warning("Cannot perform normality test on this dataset")
            
            Q1 = filtered_data['Value'].quantile(0.25)
            Q3 = filtered_data['Value'].quantile(0.75)
            IQR = Q3 - Q1
            outliers = filtered_data[(filtered_data['Value'] < Q1 - 1.5*IQR) | 
                                   (filtered_data['Value'] > Q3 + 1.5*IQR)]
            
            st.markdown(f"**Outlier Analysis:**")
            st.write(f"Detected {len(outliers)} outliers ({len(outliers)/len(filtered_data)*100:.1f}% of data)")
            
        with col2:
            st.markdown("### 🎯 Key Insights")
            
            data_quality_score = 100
            issues = []
            
            if len(outliers) / len(filtered_data) > 0.1:
                data_quality_score -= 20
                issues.append("High outlier percentage")
            
            if filtered_data['Value'].std() / filtered_data['Value'].mean() > 0.5:
                data_quality_score -= 15
                issues.append("High coefficient of variation")
            
            if filtered_data['Value'].isnull().sum() > 0:
                data_quality_score -= 25
                issues.append("Missing values present")
            
            st.markdown(f"**Data Quality Score:** {data_quality_score}/100")
            
            if issues:
                st.markdown("**Data Quality Issues:**")
                for issue in issues:
                    st.write(f"• {issue}")
            else:
                st.success("No major data quality issues detected")

with tab4:
    st.markdown("## 📋 Data Explorer")
    
    if not filtered_data.empty:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("### 🗃️ Raw Data View")
            
            display_data = filtered_data[['Year', 'Value']].copy()
            display_data['Value'] = display_data['Value'].round(2)
            
            if len(display_data) > 1:
                display_data = display_data.sort_values('Year')
                display_data['YoY_Change'] = display_data['Value'].pct_change() * 100
                display_data['YoY_Change'] = display_data['YoY_Change'].round(2)
            
            sort_by = st.selectbox("Sort by:", ["Year", "Value", "YoY_Change"] if 'YoY_Change' in display_data.columns else ["Year", "Value"])
            sort_order = st.radio("Order:", ["Ascending", "Descending"], horizontal=True)
            
            if sort_order == "Descending":
                display_data = display_data.sort_values(sort_by, ascending=False)
            else:
                display_data = display_data.sort_values(sort_by, ascending=True)
            
            st.dataframe(display_data, use_container_width=True, height=400)
        
        with col2:
            st.markdown("### 📊 Data Export Options")
            
            csv_data = display_data.to_csv(index=False)
            
            st.download_button(
                label="📥 Download Filtered Data (CSV)",
                data=csv_data,
                file_name=f"lebanon_fertilizer_{year_range[0]}_{year_range[1]}.csv",
                mime="text/csv"
            )
            
            summary_stats = filtered_data['Value'].describe().round(2)
            summary_csv = summary_stats.to_csv()
            
            st.download_button(
                label="📊 Download Summary Statistics",
                data=summary_csv,
                file_name=f"lebanon_fertilizer_summary_{year_range[0]}_{year_range[1]}.csv",
                mime="text/csv"
            )
            
            st.markdown("### ℹ️ Data Information")
            if not fertilizer_data.empty:
                st.info(f"**Dataset Period:** {min_year} - {max_year}")
                st.info(f"**Total Records:** {len(fertilizer_data)}")
                st.info(f"**Filtered Records:** {len(filtered_data)}")

st.markdown("---")
st.markdown("## 💡 Comprehensive Data Insights & Analysis")

if not filtered_data.empty:
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="insight-box">
        <h4>🔍 Context & Background</h4>
        <p>This comprehensive analysis examines Lebanon's agricultural fertilizer consumption patterns, 
        providing insights into agricultural modernization, food security efforts, and economic development 
        indicators in the Lebanese agricultural sector.</p>
        </div>
        """, unsafe_allow_html=True)
        
    with col2:
        if len(filtered_data) > 2:
            correlation = np.corrcoef(filtered_data["Year"], filtered_data["Value"])[0,1]
            trend_strength = "Strong" if abs(correlation) > 0.7 else "Moderate" if abs(correlation) > 0.4 else "Weak"
            trend_dir = "positive" if correlation > 0 else "negative"
            
            st.markdown(f"""
            <div class="insight-box">
            <h4>📈 Advanced Trend Analysis</h4>
            <p><strong>{trend_strength} {trend_dir} correlation</strong> (r = {correlation:.3f}) between years and consumption. 
            The analysis reveals {'an upward' if correlation > 0 else 'a downward'} trajectory with 
            {abs(correlation)**2*100:.1f}% of variance explained by time trends.</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="insight-box">
            <h4>📈 Trend Analysis</h4>
            <p>Select a broader year range to enable meaningful trend analysis and statistical correlations.</p>
            </div>
            """, unsafe_allow_html=True)
    
    with col3:
        volatility = filtered_data["Value"].std()
        stability_level = "Low" if volatility < 30 else "Moderate" if volatility < 80 else "High"
        
        cv = (volatility / filtered_data["Value"].mean()) * 100
        volatility_interpretation = "stable" if cv < 25 else "moderately volatile" if cv < 50 else "highly volatile"
        
        st.markdown(f"""
        <div class="insight-box">
        <h4>🎯 Enhanced Statistical Summary</h4>
        <p><strong>Volatility Level: {stability_level}</strong><br>
        Standard Deviation: {volatility:.2f}%<br>
        Coefficient of Variation: {cv:.1f}%<br>
        The data shows {volatility_interpretation} patterns with significant implications for agricultural planning.</p>
        </div>
        """, unsafe_allow_html=True)

    if 'outliers' in locals() and len(outliers) > 0:
        st.markdown(f"""
        <div class="warning-box">
        <h4>⚠️ Data Quality Alert</h4>
        <p>Detected {len(outliers)} extreme outliers in the dataset. These may represent:</p>
        <ul>
        <li>Economic crisis periods or policy changes</li>
        <li>Data collection or measurement errors</li>
        <li>Exceptional agricultural or political circumstances</li>
        </ul>
        <p>Consider investigating years: {', '.join(map(str, outliers['Year'].tolist()[:5])) if not outliers.empty else 'None'}</p>
        </div>
        """, unsafe_allow_html=True)

# Footer
st.markdown("---")

st.markdown("🌱 **Lebanon Agricultural Development Analytics** | Enhanced Interactive Dashboard | Built with Streamlit & Matplotlib")
