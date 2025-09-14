### Instructions for Running Streamlit in Local Environment (Dev)  

1. Clone the repository to your local machine.  
    ***using ssh***  
    ```bash
    git clone git@github.com:roscoelai/Team1_IT5006_Healthcare_Analytics_AY2526.git
    ```
    ***using https***  
    ```bash
    git clone https://github.com/roscoelai/Team1_IT5006_Healthcare_Analytics_AY2526.git
    ```  
2. Navigate to the project directory
    ```bash
    cd Team1_IT5006_Healthcare_Analytics_AY2526
    ```
3. Create and activate a virtual environment (Optional):
    ```bash
    python -m venv .venv
    source .venv/bin/activate  # On Windows use `.venv\Scripts\activate`
    ```
4. Install the required dependencies for streamlit  
    ```bash
    pip install -r dashboard_app/requirements.txt
    ```
5. Start up the streamlit application in your local environment  
    ```bash
    PYTHONPATH=. streamlit run dashboard_app/app.py  
    ```
6. By default, the application will be accessible at http://localhost:8501 Open this URL in your web browser to view the application.  
