import streamlit as st
import pandas as pd
import numpy as np
import base64
import io
import os
import zipfile
from datetime import datetime
from prediction import generate_prediction, calculate_confidence
from analysis import analyze_historical_data

def get_download_link(df, filename="aviator_predictions.csv", text="Download Predictions CSV"):
    """
    Generate a download link for a pandas DataFrame
    """
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">{text}</a>'
    return href

def get_excel_download_link(df, filename="aviator_predictions.xlsx", text="Download Predictions Excel"):
    """
    Generate a download link for a pandas DataFrame as Excel
    """
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, sheet_name='Predictions', index=False)
    
    excel_data = output.getvalue()
    b64 = base64.b64encode(excel_data).decode()
    href = f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download="{filename}">{text}</a>'
    return href

def get_txt_download_link(text_content, filename="aviator_predictions.txt", link_text="Download Predictions TXT"):
    """
    Generate a download link for text content
    """
    b64 = base64.b64encode(text_content.encode()).decode()
    href = f'<a href="data:file/txt;base64,{b64}" download="{filename}">{link_text}</a>'
    return href

def get_zip_download_link(filename="aviator-predictor.zip", link_text="Download Complete Project (ZIP)"):
    """
    Generate a download link for a zip file of all project files
    """
    # Create a temporary ZIP file
    memory_file = io.BytesIO()
    
    with zipfile.ZipFile(memory_file, 'w', zipfile.ZIP_DEFLATED) as zf:
        # Get all Python files and other important files
        for root, dirs, files in os.walk('.'):
            # Skip hidden directories and cache
            if any(part.startswith('.') for part in root.split(os.sep)) or '__pycache__' in root:
                continue
                
            for file in files:
                # Skip temporary files, hidden files and large data files
                if (file.startswith('.') or file.endswith(('.pyc', '.git', '.DS_Store', '.log')) or 
                    'cache' in file.lower() or 'tmp' in file.lower()):
                    continue
                    
                file_path = os.path.join(root, file)
                # Add file to zip with proper path
                arc_name = file_path[2:] if file_path.startswith('./') else file_path
                zf.write(file_path, arc_name)
    
    # Get the ZIP file data
    memory_file.seek(0)
    zip_data = memory_file.getvalue()
    
    # Create download link
    b64 = base64.b64encode(zip_data).decode()
    href = f'<a href="data:application/zip;base64,{b64}" download="{filename}">{link_text}</a>'
    return href

def generate_prediction_report(data, analysis_method, time_window, confidence_level):
    """
    Generate a full prediction report for download
    """
    # Generate predictions
    predictions = generate_prediction(
        data, 
        n_predictions=10, 
        method=analysis_method, 
        time_window=time_window
    )
    
    confidence = calculate_confidence(
        data, 
        predictions, 
        confidence_level=confidence_level/100
    )
    
    # Get statistical analysis
    stats = analyze_historical_data(data, time_window)
    
    # Create predictions DataFrame
    pred_df = pd.DataFrame({
        'Prediction': range(1, len(predictions) + 1),
        'Multiplier': [f"{p:.2f}x" for p in predictions],
        'Confidence': [f"{c:.1f}%" for c in confidence]
    })
    
    # Create a report in text format
    report = []
    report.append("=== AVIATOR PREDICTION REPORT ===")
    report.append(f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append(f"Analysis Method: {analysis_method}")
    report.append(f"Time Window: {time_window} games")
    report.append(f"Confidence Level: {confidence_level}%")
    report.append("\n--- STATISTICAL ANALYSIS ---")
    report.append(f"Average Multiplier: {stats['mean']:.2f}x")
    report.append(f"Median Multiplier: {stats['median']:.2f}x")
    report.append(f"Minimum Multiplier: {stats['min']:.2f}x")
    report.append(f"Maximum Multiplier: {stats['max']:.2f}x")
    report.append(f"Standard Deviation: {stats['std']:.2f}")
    report.append("\n--- PREDICTION RESULTS ---")
    
    for i, (pred, conf) in enumerate(zip(predictions, confidence)):
        report.append(f"Prediction {i+1}: {pred:.2f}x (Confidence: {conf:.1f}%)")
    
    # Add recommendations
    report.append("\n--- RECOMMENDATIONS ---")
    
    # Basic recommendations based on statistical analysis
    if stats['mean'] < 2.0:
        report.append("- The average multiplier is relatively low. Consider conservative betting strategies.")
    elif stats['mean'] > 5.0:
        report.append("- The average multiplier is high. This period might be favorable for slightly riskier strategies.")
    
    if stats['std'] > 3.0:
        report.append("- High volatility detected. Be cautious with bet timing.")
    else:
        report.append("- Volatility is moderate. More predictable patterns may be present.")
    
    # Add disclaimer
    report.append("\n--- DISCLAIMER ---")
    report.append("This prediction is based on statistical analysis of historical data and does not guarantee future results.")
    report.append("The Aviator game operates on a random number generator, and past patterns do not necessarily predict future outcomes.")
    report.append("Use these predictions responsibly and gamble within your limits.")
    
    return pred_df, "\n".join(report)

def add_download_section():
    """
    Add download section to the Streamlit app with clean, intuitive design
    """
    st.header("📥 Download Predictions")
    
    if 'data' in st.session_state and st.session_state.data is not None:
        # Create tabs for download options
        download_tab1, download_tab2 = st.tabs(["Quick Download", "Advanced Settings"])
        
        with download_tab1:
            # Quick download section with minimal options
            st.markdown("### Quick Download")
            
            # Add simple explanation
            st.markdown("""
            Instantly download predictions in your preferred format with just one click.
            Perfect for mobile use - quick and simple!
            """)
            
            # Simple selection for format
            quick_format = st.radio(
                "Choose format:",
                ["CSV", "Excel", "Text Report"],
                horizontal=True
            )
            
            # Prominent download button with icon
            quick_download_button = st.button("⬇️ Download Predictions", use_container_width=True, type="primary")
            
            if quick_download_button:
                with st.spinner("Preparing your download..."):
                    # Generate predictions with sensible defaults
                    quick_predictions = generate_prediction(
                        st.session_state.data, 
                        n_predictions=10, 
                        method="Moving Average", 
                        time_window=100
                    )
                    
                    quick_confidence = calculate_confidence(
                        st.session_state.data, 
                        quick_predictions, 
                        confidence_level=0.9  # 90% confidence
                    )
                    
                    # Create DataFrame for download
                    quick_df = pd.DataFrame({
                        'Prediction': range(1, len(quick_predictions) + 1),
                        'Multiplier': quick_predictions,
                        'Confidence': quick_confidence
                    })
                    
                    # Generate prediction report
                    _, report_text = generate_prediction_report(
                        st.session_state.data,
                        "Moving Average",
                        100,  # Time window
                        90  # Confidence level
                    )
                    
                    # Show download based on selected format with improved button styling
                    st.success("✅ Your prediction file is ready!")
                    
                    # Style the download link as a button
                    button_style = """
                    <style>
                    .download-button {
                        display: inline-block;
                        padding: 10px 20px;
                        background-color: #4CAF50;
                        color: white;
                        text-align: center;
                        text-decoration: none;
                        font-size: 16px;
                        border-radius: 8px;
                        width: 100%;
                        margin: 10px 0;
                        font-weight: bold;
                        transition: background-color 0.3s;
                    }
                    .download-button:hover {
                        background-color: #45a049;
                    }
                    </style>
                    """
                    
                    if quick_format == "CSV":
                        download_link = get_download_link(quick_df, text="📊 DOWNLOAD CSV FILE")
                        st.markdown(button_style + download_link.replace('<a ', '<a class="download-button" '), unsafe_allow_html=True)
                        
                    elif quick_format == "Excel":
                        download_link = get_excel_download_link(quick_df, text="📊 DOWNLOAD EXCEL FILE")
                        st.markdown(button_style + download_link.replace('<a ', '<a class="download-button" '), unsafe_allow_html=True)
                    
                    elif quick_format == "Text Report":
                        download_link = get_txt_download_link(report_text, link_text="📝 DOWNLOAD TEXT REPORT")
                        st.markdown(button_style + download_link.replace('<a ', '<a class="download-button" '), unsafe_allow_html=True)
                    
                    # Show a small preview
                    st.caption("Preview of your predictions:")
                    if quick_format in ["CSV", "Excel"]:
                        st.dataframe(quick_df, use_container_width=True)
                    else:
                        with st.expander("Text Report Preview"):
                            st.text_area("", report_text, height=200)
        
        with download_tab2:
            # Advanced download settings for power users
            st.markdown("### Advanced Settings")
            st.markdown("Customize every aspect of your prediction downloads.")
            
            col1, col2 = st.columns(2)
            
            with col1:
                download_format = st.selectbox(
                    "Select download format:",
                    ["CSV", "Excel", "Text Report"]
                )
                
                predictions_count = st.slider(
                    "Number of predictions to download", 
                    min_value=5, 
                    max_value=50, 
                    value=10, 
                    step=5
                )
            
            with col2:
                download_analysis_method = st.selectbox(
                    "Analysis Method for Download",
                    ["Basic Statistics", "Pattern Recognition", "Moving Average", "Random Forest"],
                    index=2  # Default to moving average for downloads
                )
                
                download_confidence = st.slider(
                    "Confidence Level for Download (%)", 
                    min_value=50, 
                    max_value=99, 
                    value=90, 
                    step=1
                )
            
            # Generate download button
            if st.button("Generate Custom Download", type="secondary", use_container_width=True):
                with st.spinner("Preparing customized download..."):
                    # Generate predictions for download
                    download_predictions = generate_prediction(
                        st.session_state.data, 
                        n_predictions=predictions_count, 
                        method=download_analysis_method, 
                        time_window=100  # Use default time window
                    )
                    
                    download_confidence = calculate_confidence(
                        st.session_state.data, 
                        download_predictions, 
                        confidence_level=download_confidence/100
                    )
                    
                    # Create DataFrame for download
                    download_df = pd.DataFrame({
                        'Prediction': range(1, len(download_predictions) + 1),
                        'Multiplier': download_predictions,
                        'Confidence': download_confidence
                    })
                    
                    # Generate prediction report
                    pred_df, report_text = generate_prediction_report(
                        st.session_state.data,
                        download_analysis_method,
                        100,  # Time window
                        download_confidence
                    )
                    
                    # Show success message
                    st.success("✅ Your custom prediction file is ready!")
                    
                    # Create the same styled button for consistency
                    button_style = """
                    <style>
                    .download-button {
                        display: inline-block;
                        padding: 10px 20px;
                        background-color: #4CAF50;
                        color: white;
                        text-align: center;
                        text-decoration: none;
                        font-size: 16px;
                        border-radius: 8px;
                        width: 100%;
                        margin: 10px 0;
                        font-weight: bold;
                        transition: background-color 0.3s;
                    }
                    .download-button:hover {
                        background-color: #45a049;
                    }
                    </style>
                    """
                    
                    # Show download links based on format
                    if download_format == "CSV":
                        download_link = get_download_link(download_df, text="📊 DOWNLOAD CUSTOM CSV")
                        st.markdown(button_style + download_link.replace('<a ', '<a class="download-button" '), unsafe_allow_html=True)
                        
                    elif download_format == "Excel":
                        download_link = get_excel_download_link(download_df, text="📊 DOWNLOAD CUSTOM EXCEL")
                        st.markdown(button_style + download_link.replace('<a ', '<a class="download-button" '), unsafe_allow_html=True)
                    
                    elif download_format == "Text Report":
                        download_link = get_txt_download_link(report_text, link_text="📝 DOWNLOAD CUSTOM REPORT")
                        st.markdown(button_style + download_link.replace('<a ', '<a class="download-button" '), unsafe_allow_html=True)
                    
                    # Show preview
                    st.subheader("Preview")
                    
                    if download_format in ["CSV", "Excel"]:
                        st.dataframe(download_df)
                    else:
                        with st.expander("Full Text Report Preview"):
                            st.text_area("", report_text, height=300)
    
    else:
        # Mobile-friendly message when no data is loaded
        st.warning("⚠️ Please load data first to enable downloads.")
        
        col1, col2 = st.columns([1, 3])
        
        with col1:
            st.markdown("👈")
        
        with col2:
            st.markdown("Use the sidebar to load Aviator data first")
    
    # Add project download section (always available)
    st.header("🗂️ Download Full Project")
    st.markdown("Download the entire Aviator Predictor project with all source files.")
    
    project_download_btn = st.button("⬇️ Create Project ZIP File", use_container_width=True)
    
    if project_download_btn:
        with st.spinner("Creating ZIP archive of all project files..."):
            # Create download button styled the same way
            button_style = """
            <style>
            .download-button {
                display: inline-block;
                padding: 10px 20px;
                background-color: #4CAF50;
                color: white;
                text-align: center;
                text-decoration: none;
                font-size: 16px;
                border-radius: 8px;
                width: 100%;
                margin: 10px 0;
                font-weight: bold;
                transition: background-color 0.3s;
            }
            .download-button:hover {
                background-color: #45a049;
            }
            </style>
            """
            
            # Get download link for ZIP
            download_link = get_zip_download_link(
                filename="aviator-predictor-complete.zip",
                link_text="📦 DOWNLOAD COMPLETE PROJECT"
            )
            
            # Show success message
            st.success("✅ Project ZIP file is ready!")
            st.markdown(button_style + download_link.replace('<a ', '<a class="download-button" '), unsafe_allow_html=True)
            
            # Show content information
            st.markdown("""
            **What's included in the ZIP file:**
            - All Python source code files (prediction algorithms, utilities, visualizations)
            - Sample data files for demonstration
            - Configuration files for the application
            - Complete documentation within the code
            
            You can run this project on your local machine with Python and the required dependencies.
            """)
    
    # Info box with usage tips
    with st.expander("📱 Mobile Usage Tips"):
        st.markdown("""
        **Using on Mobile:**
        1. Download the prediction file to your device
        2. Open the file in any spreadsheet or text app
        3. Refer to the predictions during your Aviator game sessions
        
        **Recommendation:** For the best mobile experience, use the "Quick Download" option.
        """)
        
    # Add small separator
    st.markdown("---")