import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sentence_transformers import SentenceTransformer
from scipy import spatial
import streamlit as st
#from utils import groq_generation
from groq import Groq
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate


llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)

def get_response(prompt):
    """Helper function to get response from the language model."""
    return llm.invoke(prompt).content

def create_chain(prompt_template):
    """
    Create a LangChain chain with the given prompt template.

    Args:
        prompt_template (str): The prompt template string.

    Returns:
        LLMChain: A LangChain chain object.
    """
    prompt = PromptTemplate.from_template(prompt_template)
    return prompt | llm

direct_task_prompt = """
Translate the text into a complete English in case it contains Hebrew. Give me only the final text. In case it does not contain Hebrew, return only the text"

Text: {text}

Translation:

"""

direct_task_chain = create_chain(direct_task_prompt)

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
report_df['VendorName'] = report_df['VendorName'].apply(lambda x: x[::-1])


var_mapping = {'קבוצת מוצר':'MaterialGroup',
               'דמיון טקסטואלי':'Similarity_Category',
               'התאמה מדויקת - כמות פריטים':'Qty_Mismatch_Category',
               'התאמה מדויקת - מחיר':'Price_Mismatch_Category',
               'שם ספק':'VendorName'}
    
# Load Transformers model
model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

user_input1 = st.text_input("תיאור פריט פנימי")
user_input2 = st.text_input("תיאור פריט ספק")

button = st.button("חשב דמיון טקסט")
if user_input1 and user_input2 and button:
    
    text_1 = direct_task_chain.invoke({user_input1}).content
    text_2 = direct_task_chain.invoke({user_input2}).content

    emb1 = model.encode(text_1)
    emb2 = model.encode(text_2)
    similarity_score = 1 - spatial.distance.cosine(emb1, emb2)
    st.write("{:.3f}".format(similarity_score))
    #st.write(f"Similarity Score: {"{:.2f}".format(similarity_score)}") # "{:.2f}".format(x)

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
    fig = plt.figure(figsize=(7, 5))
    sns.barplot(x='Approval_Rate', y=group_col, data=kpi_df, palette='viridis',legend=False,hue=group_col)
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




