import app as st
import pandas as pd
import pickle

def load_model():
    with open('pijar_rf_model.pkl', 'rb') as file:
        model = pickle.load(file)
    return model

model = load_model()

def predict_claim(input_data):
    input_df = pd.DataFrame([input_data], columns=[
        'Duration', 'Sales', 'Commision', 'Age', 
        'Type_Travel Agency', 'Distribution_Online', 
        'Agency_encoded', 'Product_encoded', 
        'Destination_encoded'
    ])
    
    prediction = model.predict(input_df)
    
    return 'Claim Approved' if prediction[0] == 1 else 'No Claim'


def main():
    
    st.markdown(
        """
        <div style='display: flex; align-items: center; margin-bottom: 20px;'>
            <img src='https://blogger.googleusercontent.com/img/b/R29vZ2xl/AVvXsEi1wySGm9lFze7qvDIteXSCMFxZJKvPVGc4Cy0tBLYf5Ajcl-uJsJ8RkUgqCjhzwB-K0cy_lvEMLYuCh7huiC4JJzo9byGv5PBusHm1hEKLuWN0JiRa7x2rNPV1l6o7MfX8g_w_4UaXh-UolBnD1Ke0jJyhRVUafrGJLP6cs1qbCMdw6TiqpaCgu3VARA/s320/sigma.png' style='width: 100px; height: 100px;'>
            <h1 style='margin-left: 20px;'>Insurance Claim Prediction</h1>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    # Input 
    Duration = st.number_input('Duration (normalized)', format="%.6f")
    Sales = st.number_input('Sales (normalized)', format="%.6f")
    Commision = st.number_input('Commision (normalized)', format="%.6f")
    Age = st.number_input('Age (normalized)')
    Type_Travel_Agency = st.number_input('Type_Travel_Agency (0 or 1)', min_value=0.0, max_value=1.0, step=0.01)
    Distribution_Online = st.number_input('Distribution_Online (0 or 1)', min_value=0.0, max_value=1.0, step=0.01)
    Agency_encoded = st.number_input('Agency (encoded)', min_value=0.0, max_value=1.0, step=0.01)
    Product_encoded = st.number_input('Product (encoded)', min_value=0.0, max_value=1.0, step=0.01)
    Destination_encoded = st.number_input('Destination (encoded)', min_value=0.0, max_value=1.0, step=0.01)

    # predict
    if st.button('Predict Claim Status'):
        prediction = predict_claim([
            Duration, Sales, Commision, Age, 
            Type_Travel_Agency, Distribution_Online, 
            Agency_encoded, Product_encoded, 
            Destination_encoded
        ])
        st.success(prediction)

if __name__ == '__main__':
    main()
