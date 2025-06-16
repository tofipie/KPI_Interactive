import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sentence_transformers import SentenceTransformer
from scipy import spatial
import streamlit as st
from utils import grop_generation
from groq import Groq

GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
client = Groq(api_key=GROQ_API_KEY)
prompt = """
"Translate the following [DOCUMENT] into a complete English. Give me only the final text"

"""

st.title("Interactive KPI Dashboard and Text Similarity app 💬")
st.header('Made by Noa Cohen')

st.subheader(" מודל לחישוב דמיון טקסטואלי בין תיאורי פריטים", divider="green") 
st.subheader('sentence-transformers/all-mpnet-base-v2',divider="green")


files = ['purchase_orders','goods_receipts','vendor_invoices',
'material_master','vendor_master','invoice_approvals']
st.sidebar.title("App Description")

with st.sidebar:
    st.write("קבצים שנמצאים ב DB:")
    for file in files:
        st.markdown("- " + file)  

report_df = pd.read_csv('report_df.csv')



var_mapping = {'קבוצת מוצר':'MaterialGroup',
               'דמיון טקסטואלי':'Similarity_Category',
               'התאמה מדויקת - כמות פריטים':'Qty_Mismatch_Category',
               'התאמה מדויקת - מחיר':'Price_Mismatch_Category',
               'שם ספק':'VendorName'}
    
# Load Transformers model
model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

user_input1 = st.text_input("תיאור פריט פנימי")
user_input2 = st.text_input("תיאור פריט ספק")
button = st.button("חשב")
if user_input1 and user_input2 and button:
    #translate to english if theres hebrew description using LLM
    
    translated1 = groq_generation(prompt, user_input1)
    translated2 = groq_generation(prompt, user_input)
   
    emb1 = model.encode(translated1)
    emb2 = model.encode(translated2)
    similarity_score = 1 - spatial.distance.cosine(emb1, emb2)
    st.write(f"Similarity Score: {"{:.2f}".format(similarity_score)}") # "{:.2f}".format(x)

st.subheader("חישוב שיעור אישור חשבוניות לפי פרמטר", divider="blue")

custom_names = list(var_mapping.keys())
selected_custom_name = st.selectbox("בחר פרמטר לחישוב KPI", custom_names)
selected_actual_name = var_mapping.get(selected_custom_name)



titles_dict = {'MaterialGroup':'Approval Rate by MaterialGroup',
'Similarity_Category':'Approval Rate by Text Similarity Score',
'Qty_Mismatch_Category':'Approval Rate by Quantity',
'Price_Mismatch_Category':'Approval Rate by Price',
'VendorName':'Approval Rate by VendorName'}


title = titles_dict.get(selected_actual_name)
from PIL import Image

def generate_kpi_report(df, group_col, target_col='IsApproved', title=""):

    kpi_df = df.groupby(group_col)[target_col].agg(
        Total_Lines=('size'),
        Approved_Lines=('sum'),
        Rejection_Lines=(lambda x: (x == 0).sum())
    ).reset_index()
    kpi_df['Approval_Rate'] = (kpi_df['Approved_Lines'] / kpi_df['Total_Lines']) * 100
    kpi_df = kpi_df.sort_values(by='Approval_Rate', ascending=False)

    # Visualization
    fig = plt.figure(figsize=(7, 3))
    sns.barplot(x='Approval_Rate', y=group_col, data=kpi_df, palette='viridis')
    plt.title(title)
    plt.xlabel('Approval Rate')
    plt.ylabel(group_col)
   # plt.show()
   # st.pyplot(fig) # instead of plt.show()
        
    fig.savefig("figure_name.png")
    image = Image.open('figure_name.png')
    st.image(image)
    return kpi_df

kpi_result = generate_kpi_report(report_df,selected_actual_name,title = title)
st.write(kpi_result)




