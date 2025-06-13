import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sentence_transformers import SentenceTransformer
from scipy import spatial
import streamlit as st
st.title("Interactive KPI Dashboard and Text Similarity app 💬")

st.subheader(" מודל לחישוב דמיון טקסטואלי בין תיאורי פריטים", divider="green") 

st.sidebar.title("App Description")

var_mapping = {'קבוצת מוצר':'MaterialGroup',
               'דמיון טקסטואלי':'Similarity_Category',
               'התאמה מדויקת - כמות פריטים':'Qty_Mismatch_Category',
               'התאמה מדויקת - מחיר':'Price_Mismatch_Category',
               'שם ספק':'VendorName'}
    
custom_names = list(var_mapping.keys())
#selected_custom_name = st.sidebar.selectbox('בחר פרמטר לחישוב KPI ', ['', *custom_names])

selected_actual_name = var_mapping.get(selected_custom_name)

st.markdown('חישוב דמיון טקסטואלי')

# Load Transformers model
model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

user_input1 = st.text_input("תיאור פריט פנימי")
user_input2 = st.text_input("תיאור פריט ספק")
button = st.button("חשב")

st.subheader("חישוב שיעור אישור חשבוניות לפי פרמטר", divider="blue")
selected_custom_name = st.selectbox("בחר פרמטר לחישוב KPI", custom_names)

if user_input1 and user_input2 and button:
    emb1 = model.encode(user_input1)
    emb2 = model.encode(user_input2)
    similarity_score = 1 - spatial.distance.cosine(emb1, emb2)
    st.write(f"Similarity Score: {"{:.2f}".format(similarity_score)}") # "{:.2f}".format(x)



files = ['purchase_orders','goods_receipts','vendor_invoices',
'material_master','vendor_master','invoice_approvals']

with st.sidebar:
    st.write("קבצים שנמצאים ב DB:")
    st.write('sentence-transformers/all-mpnet-base-v2')
    for file in files:
        st.markdown("- " + file)  
    st.write('Made by Noa Cohen')

report_df = pd.read_csv('report_df.csv')

titles_dict = {'MaterialGroup':'Approval Rate by MaterialGroup',
'Similarity_Category':'Approval Rate by Text Similarity Score',
'Qty_Mismatch_Category':'Approval Rate by Quantity',
'Price_Mismatch_Category':'Approval Rate by Price',
'VendorName':'Approval Rate by VendorName'}


title = titles_dict.get(selected_actual_name)

def generate_kpi_report(df, group_col, target_col='IsApproved', title=""):

    kpi_df = df.groupby(group_col)[target_col].agg(
        Total_Lines=('size'),
        Approved_Lines=('sum'),
        Rejection_Lines=(lambda x: (x == 0).sum())
    ).reset_index()
    kpi_df['Approval_Rate'] = (kpi_df['Approved_Lines'] / kpi_df['Total_Lines']) * 100
    kpi_df = kpi_df.sort_values(by='Approval_Rate', ascending=False)

    # Visualization
    fig = plt.figure(figsize=(2, 2))
    sns.barplot(x='Approval_Rate', y=group_col, data=kpi_df, palette='viridis')
    plt.title(title)
    plt.xlabel('Approval Rate')
    plt.ylabel(group_col)
    plt.show()
    st.pyplot(fig) # instead of plt.show()
    return kpi_df

kpi_result = generate_kpi_report(report_df,selected_actual_name,title = title)
st.write(kpi_result)


